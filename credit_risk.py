"""
Credit Risk - Probability of Default (PD) model
================================================

End-to-end credit risk project in a single file:

    1. Load the German Credit dataset (UCI / OpenML "credit-g", 1,000 loan applicants)
    2. Train a Logistic Regression baseline and a monotonic-constrained XGBoost model
    3. Evaluate with banking metrics: ROC AUC, Gini, KS, PR AUC, Brier score
    4. Choose the approval cut-off by minimising expected misclassification cost
    5. Convert PD into a points-based credit score (scorecard scaling)
    6. Explain the model globally and per applicant with SHAP values
    7. Serve PD, score, decision and reason codes through a FastAPI REST API

Usage
-----
    python credit_risk.py train                 # train, evaluate, save model + reports
    python credit_risk.py predict --example     # score a sample applicant from the CLI
    python credit_risk.py predict --json applicant.json
    python credit_risk.py serve                 # start the API on http://127.0.0.1:8000
"""

import argparse
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "german_credit.csv"
MODEL_PATH = ROOT / "models" / "credit_risk_model.joblib"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
N_SEARCH_ITER = 30
TARGET = "default"  # 1 = bad loan (default), 0 = good loan

NUMERIC = [
    "duration",                # loan duration in months
    "credit_amount",           # loan amount (Deutsche Mark)
    "installment_commitment",  # instalment as % of disposable income (1-4 band)
    "residence_since",         # years at current address (1-4 band)
    "age",
    "existing_credits",        # number of existing credits at this bank
    "num_dependents",
]
CATEGORICAL = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "other_parties",
    "property_magnitude",
    "other_payment_plans",
    "housing",
    "job",
    "own_telephone",
]
FEATURES = NUMERIC + CATEGORICAL

# Excluded on fair-lending grounds: personal_status mixes sex with marital status and
# foreign_worker is a proxy for national origin. Neither should drive a credit decision.
EXCLUDED = ["personal_status", "foreign_worker"]

# Business-logic constraints: risk may only go UP as these increase (holding all else
# fixed). This keeps the model intuitive and defensible to model validators.
MONOTONE_INCREASING = ["duration", "credit_amount", "installment_commitment"]

# Cost matrix published with the dataset: approving a borrower who defaults costs 5x
# as much as declining a borrower who would have repaid.
COST_FALSE_NEGATIVE = 5.0  # approved a defaulter
COST_FALSE_POSITIVE = 1.0  # declined a good customer

# Scorecard scaling: 600 points = 50:1 good/bad odds, every 20 points doubles the odds.
BASE_SCORE, BASE_ODDS, PDO = 600, 50.0, 20.0

# PD bands used for the risk grade returned by the API.
RISK_BANDS = [(0.05, "A"), (0.10, "B"), (0.20, "C"), (0.35, "D"), (1.01, "E")]

EXAMPLE_APPLICANT = {
    "checking_status": "no checking",
    "duration": 12,
    "credit_history": "existing paid",
    "purpose": "radio/tv",
    "credit_amount": 2500,
    "savings_status": "<100",
    "employment": "1<=X<4",
    "installment_commitment": 2,
    "other_parties": "none",
    "residence_since": 2,
    "property_magnitude": "car",
    "age": 35,
    "other_payment_plans": "none",
    "housing": "own",
    "existing_credits": 1,
    "job": "skilled",
    "num_dependents": 1,
    "own_telephone": "none",
}

log = logging.getLogger("credit_risk")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the German Credit data, downloading it from OpenML the first time."""
    if path.exists():
        df = pd.read_csv(path)
        log.info("Loaded %s rows from %s", len(df), path.relative_to(ROOT))
    else:
        from sklearn.datasets import fetch_openml

        log.info("Downloading German Credit data (OpenML 'credit-g')...")
        df = fetch_openml("credit-g", version=1, as_frame=True, parser="auto").frame
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        log.info("Saved raw data to %s", path.relative_to(ROOT))

    missing = [c for c in FEATURES + ["class"] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")

    df = df.copy()
    for col in CATEGORICAL + ["class"]:
        df[col] = df[col].astype(str).str.strip().str.strip("'\"")
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col])
    df[TARGET] = (df["class"] == "bad").astype(int)
    return df


def data_quality_report(df: pd.DataFrame) -> dict:
    return {
        "rows": int(len(df)),
        "default_rate": round(float(df[TARGET].mean()), 4),
        "missing_values": int(df[FEATURES].isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "excluded_features": EXCLUDED,
    }


# --------------------------------------------------------------------------------------
# Feature engineering / models
# --------------------------------------------------------------------------------------
def category_vocabulary(df: pd.DataFrame) -> dict[str, list[str]]:
    """Fixed category lists so every CV fold produces the same one-hot columns."""
    return {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL}


def build_preprocessor(categories: dict[str, list[str]], scale_numeric: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    encoder = OneHotEncoder(
        categories=[categories[c] for c in CATEGORICAL],
        handle_unknown="ignore",
        sparse_output=False,
    )
    numeric_step = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        [("num", numeric_step, NUMERIC), ("cat", encoder, CATEGORICAL)],
        remainder="drop",
    )


def column_layout(categories: dict[str, list[str]]):
    """Names of the model's input columns and which original feature each belongs to."""
    names, i = list(NUMERIC), len(NUMERIC)
    groups = [(c, [k]) for k, c in enumerate(NUMERIC)]
    for c in CATEGORICAL:
        n = len(categories[c])
        names += [f"{c}={v}" for v in categories[c]]
        groups.append((c, list(range(i, i + n))))
        i += n
    return names, groups


def monotone_constraints(n_columns: int) -> str:
    flags = [0] * n_columns
    for c in MONOTONE_INCREASING:
        flags[NUMERIC.index(c)] = 1
    return "(" + ",".join(map(str, flags)) + ")"


def build_models(categories: dict[str, list[str]]):
    """Return {name: (pipeline, hyper-parameter search space)}."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier

    n_columns = len(column_layout(categories)[0])
    logit = Pipeline([
        ("pre", build_preprocessor(categories, scale_numeric=True)),
        ("clf", LogisticRegression(max_iter=5000)),
    ])
    logit_space = {"clf__C": [0.005, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]}

    xgb = Pipeline([
        ("pre", build_preprocessor(categories, scale_numeric=False)),
        ("clf", XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            tree_method="hist",
            monotone_constraints=monotone_constraints(n_columns),
            random_state=SEED,
            n_jobs=1,  # parallelism comes from the CV search instead
        )),
    ])
    xgb_space = {
        "clf__n_estimators": [150, 250, 400, 600],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__max_depth": [2, 3, 4],
        "clf__min_child_weight": [1, 3, 5, 10],
        "clf__subsample": [0.6, 0.8, 1.0],
        "clf__colsample_bytree": [0.5, 0.7, 0.9],
        "clf__reg_lambda": [1, 5, 10, 20],
        "clf__gamma": [0, 0.5, 1, 2],
    }
    return {"logistic_regression": (logit, logit_space), "xgboost": (xgb, xgb_space)}


# --------------------------------------------------------------------------------------
# Metrics, cut-off and scorecard
# --------------------------------------------------------------------------------------
def ks_statistic(y_true, pd_hat) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(y_true, pd_hat)
    return float(np.max(tpr - fpr))


def discrimination_metrics(y_true, pd_hat) -> dict:
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

    auc = roc_auc_score(y_true, pd_hat)
    return {
        "roc_auc": round(float(auc), 4),
        "gini": round(float(2 * auc - 1), 4),
        "ks": round(ks_statistic(y_true, pd_hat), 4),
        "pr_auc": round(float(average_precision_score(y_true, pd_hat)), 4),
        "brier": round(float(brier_score_loss(y_true, pd_hat)), 4),
        "log_loss": round(float(log_loss(y_true, pd_hat)), 4),
    }


def expected_cost(y_true, pd_hat, threshold: float) -> float:
    """Average cost per applicant if we decline everyone with PD >= threshold."""
    y_true = np.asarray(y_true)
    decline = np.asarray(pd_hat) >= threshold
    fn = np.sum((~decline) & (y_true == 1))
    fp = np.sum(decline & (y_true == 0))
    return float((COST_FALSE_NEGATIVE * fn + COST_FALSE_POSITIVE * fp) / len(y_true))


def optimal_threshold(y_true, pd_hat) -> tuple[float, np.ndarray, np.ndarray]:
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    costs = np.array([expected_cost(y_true, pd_hat, t) for t in grid])
    return float(grid[int(np.argmin(costs))]), grid, costs


def decision_metrics(y_true, pd_hat, threshold: float) -> dict:
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    decline = (np.asarray(pd_hat) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, decline, labels=[0, 1]).ravel()
    approved = decline == 0
    return {
        "threshold": threshold,
        "approval_rate": round(float(approved.mean()), 4),
        "bad_rate_in_approved": round(float(y_true[approved].mean()), 4) if approved.any() else None,
        "defaults_caught_recall": round(float(tp / (tp + fn)), 4) if (tp + fn) else None,
        "precision_of_declines": round(float(tp / (tp + fp)), 4) if (tp + fp) else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "expected_cost_per_applicant": round(expected_cost(y_true, pd_hat, threshold), 4),
        "cost_if_approve_everyone": round(COST_FALSE_NEGATIVE * float(y_true.mean()), 4),
    }


def pd_to_score(pd_hat) -> np.ndarray:
    """Scorecard scaling: higher score = lower risk."""
    p = np.clip(np.asarray(pd_hat, dtype=float), 1e-6, 1 - 1e-6)
    factor = PDO / np.log(2)
    offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    return np.round(offset + factor * np.log((1 - p) / p)).astype(int)


def risk_band(pd_value: float) -> str:
    return next(band for limit, band in RISK_BANDS if pd_value < limit)


# --------------------------------------------------------------------------------------
# Explainability (TreeSHAP, computed natively by XGBoost)
# --------------------------------------------------------------------------------------
def shap_contributions(artifact: dict, X: pd.DataFrame):
    """Exact TreeSHAP values in log-odds space.

    Returns (per one-hot column values, per original feature values, base value).
    One-hot columns are summed back to their original feature, so each applicant gets
    one contribution per input field - the format reason codes need.
    """
    import xgboost as xgb

    pipe = artifact["model"]
    Xt = pipe.named_steps["pre"].transform(X[FEATURES])
    contribs = pipe.named_steps["clf"].get_booster().predict(
        xgb.DMatrix(np.asarray(Xt, dtype=float)), pred_contribs=True
    )
    columns, base = contribs[:, :-1], contribs[:, -1]
    grouped = np.column_stack([columns[:, idx].sum(axis=1) for _, idx in artifact["groups"]])
    return columns, grouped, base


def reason_codes(feature_names, values, contributions, top_n: int = 4) -> list[dict]:
    order = np.argsort(-np.abs(contributions))[:top_n]
    return [
        {
            "feature": feature_names[i],
            "value": values[i],
            "contribution_log_odds": round(float(contributions[i]), 4),
            "effect": "increases risk" if contributions[i] > 0 else "decreases risk",
        }
        for i in order
    ]


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------
def train() -> dict:
    import matplotlib

    matplotlib.use("Agg")
    from sklearn.base import clone
    from sklearn.model_selection import (RandomizedSearchCV, StratifiedKFold,
                                         cross_val_predict, train_test_split)

    for d in (MODEL_PATH.parent, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    df = load_data()
    quality = data_quality_report(df)
    log.info("Data: %s rows, default rate %.1f%%", quality["rows"], 100 * quality["default_rate"])

    categories = category_vocabulary(df)
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)

    results, fitted, test_pd = {}, {}, {}
    for name, (pipe, space) in build_models(categories).items():
        n_combos = int(np.prod([len(v) for v in space.values()]))
        search = RandomizedSearchCV(
            pipe, space, n_iter=min(N_SEARCH_ITER, n_combos), scoring="roc_auc",
            cv=cv, random_state=SEED, n_jobs=-1, refit=True,
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        # Out-of-fold PDs on the training set are used to pick the cut-off, so the
        # test set is never touched when making that decision.
        oof_pd = cross_val_predict(clone(best), X_train, y_train, cv=cv,
                                   method="predict_proba", n_jobs=-1)[:, 1]
        threshold, grid, costs = optimal_threshold(y_train, oof_pd)
        pd_test = best.predict_proba(X_test)[:, 1]

        results[name] = {
            "best_params": {k.replace("clf__", ""): v for k, v in search.best_params_.items()},
            "cv_roc_auc": round(float(search.best_score_), 4),
            "test": discrimination_metrics(y_test, pd_test),
            "decision_at_cost_optimal_threshold": decision_metrics(y_test, pd_test, threshold),
        }
        fitted[name] = (best, threshold, grid, costs)
        test_pd[name] = pd_test
        log.info("%-20s CV AUC %.3f | test AUC %.3f | Gini %.3f | KS %.3f | cut-off %.2f",
                 name, results[name]["cv_roc_auc"], results[name]["test"]["roc_auc"],
                 results[name]["test"]["gini"], results[name]["test"]["ks"], threshold)

    # XGBoost is the champion model served by the API; logistic regression is the
    # transparent benchmark that a model-risk team would expect to see alongside it.
    champion = "xgboost"
    model, threshold, grid, costs = fitted[champion]
    names, groups = column_layout(categories)
    artifact = {
        "model": model,
        "model_name": champion,
        "threshold": threshold,
        "features": FEATURES,
        "categories": categories,
        "numeric_ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in NUMERIC},
        "column_names": names,
        "groups": groups,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    joblib.dump(artifact, MODEL_PATH)
    log.info("Saved model to %s", MODEL_PATH.relative_to(ROOT))

    unknown = {k: v for k, v in EXAMPLE_APPLICANT.items()
               if k in CATEGORICAL and v not in categories[k]}
    if unknown:
        log.warning("EXAMPLE_APPLICANT uses values not in the data: %s", unknown)

    summary = {
        "data": quality,
        "split": {"train": int(len(X_train)), "test": int(len(X_test)), "cv_folds": CV_FOLDS},
        "costs": {"false_negative": COST_FALSE_NEGATIVE, "false_positive": COST_FALSE_POSITIVE},
        "champion_model": champion,
        "models": results,
        "environment": environment_versions(),
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(summary, indent=2, default=str))
    write_model_comparison(results)

    make_figures(artifact, fitted, X_test, y_test, test_pd)
    log.info("Reports written to %s", REPORTS_DIR.relative_to(ROOT))
    return summary


def environment_versions() -> dict:
    import sklearn
    import xgboost

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "xgboost": xgboost.__version__, "pandas": pd.__version__, "numpy": np.__version__}


def write_model_comparison(results: dict) -> None:
    rows = ["| Model | CV AUC | Test AUC | Gini | KS | Brier | Cut-off | Approval rate | "
            "Bad rate in approved | Cost / applicant |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for name, r in results.items():
        t, d = r["test"], r["decision_at_cost_optimal_threshold"]
        rows.append(f"| {name} | {r['cv_roc_auc']:.3f} | {t['roc_auc']:.3f} | {t['gini']:.3f} | "
                    f"{t['ks']:.3f} | {t['brier']:.3f} | {d['threshold']:.2f} | "
                    f"{d['approval_rate']:.1%} | {d['bad_rate_in_approved']:.1%} | "
                    f"{d['expected_cost_per_applicant']:.3f} |")
    baseline = next(iter(results.values()))["decision_at_cost_optimal_threshold"]["cost_if_approve_everyone"]
    rows.append(f"\nCost of approving every applicant (no model): {baseline:.3f} per applicant.")
    (REPORTS_DIR / "model_comparison.md").write_text("\n".join(rows) + "\n")


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def make_figures(artifact, fitted, X_test, y_test, test_pd) -> None:
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    from sklearn.metrics import roc_curve

    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    labels = {"logistic_regression": "Logistic regression", "xgboost": "XGBoost (monotonic)"}

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in test_pd.items():
        fpr, tpr, _ = roc_curve(y_test, p)
        auc = np.trapezoid(tpr, fpr) if hasattr(np, "trapezoid") else np.trapz(tpr, fpr)
        ax.plot(fpr, tpr, label=f"{labels[name]} (AUC {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve - test set")
    ax.legend(loc="lower right")
    save(fig, "roc_curve.png")

    # Calibration
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, p in test_pd.items():
        frac, mean_pred = calibration_curve(y_test, p, n_bins=8, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", label=labels[name])
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="Perfect calibration")
    ax.set(xlabel="Predicted PD", ylabel="Observed default rate", title="Calibration - test set")
    ax.legend()
    save(fig, "calibration.png")

    # Expected cost vs cut-off (out-of-fold, training data)
    _, threshold, grid, costs = fitted[artifact["model_name"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(grid, costs)
    ax.axvline(threshold, ls="--", color="crimson", label=f"Chosen cut-off = {threshold:.2f}")
    ax.set(xlabel="Decline if PD >= cut-off", ylabel="Expected cost per applicant",
           title="Cost-based cut-off (5:1 cost of a missed default)")
    ax.legend()
    save(fig, "cost_vs_threshold.png")

    # Score distribution by outcome
    scores = pd_to_score(test_pd[artifact["model_name"]])
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(scores.min(), scores.max(), 25)
    ax.hist(scores[np.asarray(y_test) == 0], bins=bins, alpha=0.6, label="Repaid")
    ax.hist(scores[np.asarray(y_test) == 1], bins=bins, alpha=0.6, label="Defaulted")
    ax.set(xlabel="Credit score", ylabel="Applicants", title="Score distribution - test set")
    ax.legend()
    save(fig, "score_distribution.png")

    # SHAP
    columns, grouped, base = shap_contributions(artifact, X_test)
    order = np.argsort(np.abs(grouped).mean(axis=0))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(np.array(FEATURES)[order], np.abs(grouped).mean(axis=0)[order])
    ax.set(xlabel="Mean |SHAP value| (log-odds)", title="Global feature importance")
    save(fig, "shap_importance.png")

    try:
        import shap

        Xt = artifact["model"].named_steps["pre"].transform(X_test[FEATURES])
        exp = shap.Explanation(values=columns, base_values=base, data=np.asarray(Xt, dtype=float),
                               feature_names=artifact["column_names"])
        shap.plots.beeswarm(exp, max_display=15, show=False)
        plt.title("SHAP summary - effect on default log-odds")
        save(plt.gcf(), "shap_beeswarm.png")

        # Explain the riskiest applicant in the test set (a typical "why declined" case)
        i = int(np.argmax(test_pd[artifact["model_name"]]))
        single = shap.Explanation(values=grouped[i], base_values=base[i],
                                  data=X_test[FEATURES].iloc[i].astype(str).values,
                                  feature_names=FEATURES)
        shap.plots.waterfall(single, max_display=12, show=False)
        plt.title("Why this applicant was declined")
        save(plt.gcf(), "shap_waterfall_declined.png")
    except Exception as exc:  # plotting is a nice-to-have; never fail training over it
        log.warning("SHAP plots skipped: %s", exc)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)


# --------------------------------------------------------------------------------------
# Scoring (shared by the CLI and the API)
# --------------------------------------------------------------------------------------
def load_artifact(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run `python credit_risk.py train` first.")
    return joblib.load(path)


def validate_applicant(artifact: dict, applicant: dict) -> list[str]:
    errors = []
    for col in CATEGORICAL:
        allowed = artifact["categories"][col]
        if applicant.get(col) not in allowed:
            errors.append(f"{col}: '{applicant.get(col)}' is not one of {allowed}")
    return errors


def score_applicant(artifact: dict, applicant: dict, explain: bool = False) -> dict:
    X = pd.DataFrame([applicant])[FEATURES]
    pd_value = float(artifact["model"].predict_proba(X)[0, 1])
    threshold = artifact["threshold"]
    result = {
        "probability_of_default": round(pd_value, 4),
        "credit_score": int(pd_to_score([pd_value])[0]),
        "risk_band": risk_band(pd_value),
        "decision": "decline" if pd_value >= threshold else "approve",
        "cutoff": threshold,
        "model": artifact["model_name"],
        "model_trained_at": artifact["trained_at"],
    }
    if explain:
        _, grouped, base = shap_contributions(artifact, X)
        values = [applicant[f] for f in FEATURES]
        result["base_log_odds"] = round(float(base[0]), 4)
        result["top_factors"] = reason_codes(FEATURES, values, grouped[0], top_n=5)
        risk_up = grouped[0] > 0
        result["reasons_for_decline"] = (
            [r["feature"] for r in reason_codes(
                np.array(FEATURES)[risk_up].tolist(),
                np.array(values, dtype=object)[risk_up].tolist(),
                grouped[0][risk_up], top_n=4)]
            if result["decision"] == "decline" else []
        )
    return result


# --------------------------------------------------------------------------------------
# REST API
# --------------------------------------------------------------------------------------
class Applicant(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_APPLICANT]})

    checking_status: str = Field(description="Status of existing current account")
    duration: int = Field(ge=1, le=120, description="Loan duration in months")
    credit_history: str
    purpose: str
    credit_amount: float = Field(gt=0, le=1_000_000, description="Loan amount")
    savings_status: str
    employment: str = Field(description="Years in current employment (band)")
    installment_commitment: int = Field(ge=1, le=4, description="Instalment as % of income (1-4)")
    other_parties: str = Field(description="Co-applicant / guarantor")
    residence_since: int = Field(ge=1, le=4)
    property_magnitude: str
    age: int = Field(ge=18, le=100)
    other_payment_plans: str
    housing: str
    existing_credits: int = Field(ge=1, le=10)
    job: str
    num_dependents: int = Field(ge=0, le=10)
    own_telephone: str


class Factor(BaseModel):
    feature: str
    value: Any
    contribution_log_odds: float
    effect: str


class Prediction(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    probability_of_default: float
    credit_score: int
    risk_band: str
    decision: str
    cutoff: float
    model: str
    model_trained_at: str


class ExplainedPrediction(Prediction):
    base_log_odds: float
    top_factors: list[Factor]
    reasons_for_decline: list[str]


STATE: dict = {}

app = FastAPI(
    title="Credit Risk PD API",
    version="1.0.0",
    description="Probability of default, credit score, decision and SHAP reason codes.",
)


def get_artifact() -> dict:
    if "artifact" not in STATE:
        try:
            STATE["artifact"] = load_artifact()
        except FileNotFoundError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    return STATE["artifact"]


def checked(applicant: Applicant) -> tuple[dict, dict]:
    artifact = get_artifact()
    payload = applicant.model_dump()
    errors = validate_applicant(artifact, payload)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return artifact, payload


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info() -> dict:
    artifact = get_artifact()
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return {
        "model": artifact["model_name"],
        "trained_at": artifact["trained_at"],
        "cutoff": artifact["threshold"],
        "test_metrics": metrics.get("models", {}).get(artifact["model_name"], {}).get("test"),
        "allowed_values": artifact["categories"],
        "numeric_ranges_seen_in_training": artifact["numeric_ranges"],
        "excluded_features": EXCLUDED,
    }


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant) -> dict:
    artifact, payload = checked(applicant)
    return score_applicant(artifact, payload)


@app.post("/explain", response_model=ExplainedPrediction)
def explain(applicant: Applicant) -> dict:
    artifact, payload = checked(applicant)
    return score_applicant(artifact, payload, explain=True)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main(argv=None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S")
    parser = argparse.ArgumentParser(description="Credit risk PD model")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train", help="Train, evaluate and save the model")
    p = sub.add_parser("predict", help="Score one applicant")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--example", action="store_true", help="Use the built-in sample applicant")
    group.add_argument("--json", type=Path, help="Path to a JSON file with one applicant")
    s = sub.add_parser("serve", help="Run the REST API")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.command == "train":
        train()
    elif args.command == "predict":
        applicant = EXAMPLE_APPLICANT if args.example else json.loads(args.json.read_text())
        applicant = Applicant(**applicant).model_dump()
        artifact = load_artifact()
        errors = validate_applicant(artifact, applicant)
        if errors:
            sys.exit("Invalid applicant:\n  " + "\n  ".join(errors))
        print(json.dumps(score_applicant(artifact, applicant, explain=True), indent=2, default=str))
    elif args.command == "serve":
        import uvicorn

        uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
