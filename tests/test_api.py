"""API tests. Run `python credit_risk.py train` first, then `pytest -q`."""

import math
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import credit_risk as cr  # noqa: E402

pytestmark = pytest.mark.skipif(not cr.MODEL_PATH.exists(), reason="train the model first")
client = TestClient(cr.app)


def valid_applicant() -> dict:
    """Build a request from the categories the trained model actually knows."""
    categories = cr.load_artifact()["categories"]
    applicant = dict(cr.EXAMPLE_APPLICANT)
    for col, allowed in categories.items():
        if applicant[col] not in allowed:
            applicant[col] = allowed[0]
    return applicant


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_returns_valid_pd_and_score():
    r = client.post("/predict", json=valid_applicant())
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"approve", "decline"}
    assert body["risk_band"] in {"A", "B", "C", "D", "E"}


def test_shap_contributions_add_up_to_prediction():
    body = client.post("/explain", json=valid_applicant()).json()
    artifact = cr.load_artifact()
    import pandas as pd

    _, grouped, base = cr.shap_contributions(artifact, pd.DataFrame([valid_applicant()]))
    log_odds = base[0] + grouped[0].sum()
    assert math.isclose(1 / (1 + math.exp(-log_odds)), body["probability_of_default"], abs_tol=1e-3)


def test_unknown_category_is_rejected():
    bad = valid_applicant() | {"purpose": "space travel"}
    assert client.post("/predict", json=bad).status_code == 422


def test_out_of_range_number_is_rejected():
    bad = valid_applicant() | {"age": 12}
    assert client.post("/predict", json=bad).status_code == 422


def test_longer_loan_never_lowers_risk():
    """Monotonic constraint: holding everything else fixed, PD must not fall as duration rises."""
    pds = [client.post("/predict", json=valid_applicant() | {"duration": d}).json()["probability_of_default"]
           for d in (6, 12, 24, 36, 48, 60)]
    assert all(b >= a - 1e-9 for a, b in zip(pds, pds[1:]))
