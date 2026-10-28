Credit Risk – Probability of Default Model

An end-to-end credit risk model that predicts the probability of default (PD) for a loan applicant, turns it into a credit score and approve/decline decision, explains every decision with SHAP reason codes, and serves it all through a FastAPI service packaged with Docker.

Built to mirror how retail credit models are developed and governed in banks: a transparent benchmark next to a challenger model, business constraints on the model, a cut-off chosen on cost rather than accuracy, and explanations that could support an adverse-action notice.

Highlights
Area	What the project does
Data	German Credit dataset (UCI / OpenML credit-g): 1,000 applicants, 20 attributes, 30% defaults
Fair lending	Drops personal_status (encodes sex) and foreign_worker (national-origin proxy)
Models	Logistic regression benchmark vs XGBoost champion, both tuned with 5-fold stratified CV
Business constraints	Monotonic XGBoost: risk can only rise with loan duration, amount and instalment burden
Metrics	ROC AUC, Gini, KS, PR AUC, Brier score, calibration curve
Decision cut-off	Minimises expected cost using the dataset's 5:1 cost matrix, chosen on out-of-fold training predictions so the test set stays untouched
Credit score	Scorecard scaling: 600 points = 50:1 good/bad odds, 20 points to double the odds (PDO)
Explainability	Exact TreeSHAP values; global importance, beeswarm, and a waterfall for a declined applicant
Serving	FastAPI with /predict, /explain (top factors + reasons for decline), /model-info, /health
Engineering	Input validation (Pydantic + known categories), pytest suite, Docker image that trains at build
Project structure
credit-risk-pd-model/
├── credit_risk.py        # the whole pipeline: data, training, evaluation, SHAP, API, CLI
├── tests/test_api.py     # API, validation, SHAP additivity and monotonicity tests
├── data/                 # german_credit.csv is downloaded here on first run
├── models/               # trained model (git-ignored, rebuilt by `train`)
├── reports/
│   ├── metrics.json
│   ├── model_comparison.md
│   └── figures/
├── Dockerfile
├── environment.yml       # conda environment (local development)
└── requirements.txt      # pip requirements (Docker)
Quick start
bash
conda env create -f environment.yml
conda activate credit-risk

python credit_risk.py train             # downloads data, trains, writes reports/
python credit_risk.py predict --example # scores a sample applicant with reason codes
pytest -q                               # runs the test suite
python credit_risk.py serve             # API at http://127.0.0.1:8000/docs
Using the API

Interactive docs are at http://127.0.0.1:8000/docs. From the terminal:

bash
curl -X POST http://127.0.0.1:8000/explain \
  -H "Content-Type: application/json" \
  -d '{
    "checking_status": "no checking", "duration": 12, "credit_history": "existing paid",
    "purpose": "radio/tv", "credit_amount": 2500, "savings_status": "<100",
    "employment": "1<=X<4", "installment_commitment": 2, "other_parties": "none",
    "residence_since": 2, "property_magnitude": "car", "age": 35,
    "other_payment_plans": "none", "housing": "own", "existing_credits": 1,
    "job": "skilled", "num_dependents": 1, "own_telephone": "none"
  }'

The response contains the PD, credit score, risk band (A–E), decision, the five factors that moved the prediction most (in log-odds), and – for declines – the reasons for decline. GET /model-info lists every allowed category value and the test-set metrics.

Docker
bash
docker build -t credit-risk-api .
docker run -p 8000:8000 credit-risk-api

The model is trained during the image build, so the saved model always matches the library versions inside the container.

Results

Test set of 200 held-out applicants (30% default rate), each model at its own cost-optimal cut-off:

Model	CV AUC	Test AUC	Gini	KS	Brier	Cut-off	Approval rate	Bad rate in approved	Cost / applicant
Logistic regression	0.775	0.807	0.614	0.569	0.161	0.21	30.5%	13.1%	0.635
XGBoost (monotonic)	0.785	0.803	0.606	0.529	0.156	0.17	37.5%	9.3%	0.535

Approving every applicant would cost 1.500 per applicant.

Key findings

XGBoost cuts expected credit losses by 64% compared with approving everyone (0.535 vs 1.500 per applicant) and by 16% compared with the logistic regression (0.535 vs 0.635).
It approves more customers and takes on less risk at the same time: 37.5% approved vs 30.5%, with a default rate among approved loans of 9.3% vs 13.1%. The portfolio's overall default rate is 30%.
Ranking power is effectively tied. Test AUC differs by 0.004, well within the noise of a 200-applicant test set, and the logistic regression's slightly higher test score is reversed in cross-validation. XGBoost's advantage comes from better-calibrated PDs (lower Brier score), which matters most where the cut-off sits.
The approval rate is deliberately conservative. With a missed default costing five times a lost good customer, declining borderline applicants is the cheaper choice. A lender with a different cost ratio would move the cut-off, not retrain the model.
Explanations behave sensibly. For the example applicant (PD 4.2%, score 577, band A, approved), the biggest factor lowering risk is having no current account with the bank, followed by a short 12-month term and a low instalment burden.

Full numbers for both models are in reports/model_comparison.md and reports/metrics.json after running train.

ROC curve	Calibration
ROC	Calibration
Cost-based cut-off	Score distribution
Cost	Scores
Explainability
What drives risk overall	Why one applicant was declined
SHAP beeswarm	SHAP waterfall
Methodology notes

Why Gini and KS? They are the standard discrimination measures in retail credit risk. Gini = 2 × AUC − 1; KS is the largest gap between the cumulative distributions of good and bad borrowers.

Why a cost-based cut-off? Approving a borrower who defaults costs far more than turning away one who would repay. With a 5:1 cost ratio, a 0.5 cut-off is the wrong choice; the code searches cut-offs on out-of-fold predictions and keeps the one with the lowest expected cost.

Why monotonic constraints? A model where a longer or larger loan lowers predicted risk is hard to defend to model validators and regulators, even if it fits the training data. The constraints rule that out, and a test checks it.

Why SHAP from XGBoost directly? pred_contribs=True gives exact TreeSHAP values in log-odds. One-hot columns are summed back to their original field, so each applicant gets one contribution per input – the shape reason codes need – and base value + contributions reproduce the model's prediction exactly (also tested).

Limitations and next steps
1,000 applicants is small; results carry wide confidence intervals. The same pipeline would run on a larger book such as Home Credit or Lending Club data.
Out-of-time validation and population stability (PSI) monitoring would be required before real use.
Age is kept as a feature; in a production model its use would need legal review.
Probability calibration (isotonic or Platt) could be added if PDs feed capital or provisioning (IFRS 9) calculations.
Data source

Hofmann, H. (1994). Statlog (German Credit Data). UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77 – licensed CC BY 4.0.
