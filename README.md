# RiskGate: Probability-of-Default (PD) Modelling for Automated Underwriting

RiskGate is an undergraduate Business Analytics / Data Mining project that builds a calibrated probability-of-default (PD) underwriting engine for LendingClub-style loan data.

The system supports a 3-tier decision policy:
- Auto-approve for low-risk applicants
- Manual review for medium-risk applicants
- Reject for high-risk applicants

The current local prototype includes:
- model training and artifact saving
- calibrated scoring
- frozen underwriting thresholds
- reusable batch scoring
- a Streamlit website for threshold tuning and portfolio reporting

---

## Project structure

```text
RiskGate_project/
├── artifacts/
├── data/
│   ├── raw/
│   └── scored/
├── notebooks/
├── riskgate/
├── scripts/
├── webapp/
├── requirements.txt
├── README.md
├── start_webapp.bat
├── retrain_engine.bat
└── score_default_file.bat