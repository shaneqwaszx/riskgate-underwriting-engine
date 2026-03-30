from pathlib import Path

APP_NAME = "RiskGate"
ENGINE_VERSION = "0.2.1"
RANDOM_STATE = 0

PROJECT_ROOT = Path(".")
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TARGET_COL = "loan_status"
TARGET_BINARY_COL = "target_default"
NEGATIVE_LABEL = "Fully Paid"
POSITIVE_LABEL = "Charged Off"
LABEL_MAP = {
    NEGATIVE_LABEL: 0,
    POSITIVE_LABEL: 1,
}
VALID_TARGET_LABELS = set(LABEL_MAP.keys())

VALID_SIZE = 0.30
CALIBRATION_METHOD = "sigmoid"
CALIBRATION_CV = 5

# --------------------------------------------------
# Frozen notebook recipe
# --------------------------------------------------
USE_FROZEN_NOTEBOOK_RECIPE = True
SKIP_MODEL_SEARCH = True
SKIP_BENCHMARKING = True

FROZEN_MODEL_NAME = "xgb"

FROZEN_XGB_PARAMS = {
    "random_state": 0,
    "reg_lambda": 2,
    "reg_alpha": 5,
    "max_depth": 4,
    "learning_rate": 0.26,
    "gamma": 0.4,
    "colsample_bytree": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
}

NOTEBOOK_REFERENCE_METRICS = {
    "roc_auc": 0.9097792632943688,
    "average_precision": 0.7852013757526998,
    "brier_uncalibrated": 0.0807392954093243,
    "best_binary_threshold": 0.35,
}

# --------------------------------------------------
# Policy settings
# --------------------------------------------------
USE_FROZEN_THRESHOLDS = True

# IMPORTANT:
# Fill these with your actual policy choices from the notebook.
# From your charts, t_high=0.28 looks like the chosen reject threshold
# under capacity logic. Keep that only if it is truly your chosen one.
FROZEN_T_LOW = 0.10   # replace if your notebook already chose a different lower bound
FROZEN_T_HIGH = 0.28  # replace if your notebook final decision differs

MIN_REVIEW_BAND_WIDTH = 0.08

REVIEW_RATE_CAP = 0.20
MAX_AUTO_APPROVE_BAD_RATE = 0.08
POLICY_RISK_WEIGHT = 0.35
POLICY_REJECT_QUALITY_WEIGHT = 0.15

# --------------------------------------------------
# Feature handling
# --------------------------------------------------
NUMERIC_FEATURES = [
    "loan_amnt",
    "installment",
    "int_rate",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "term_months",
    "emp_length_years",
    "credit_history_months",
]

CATEGORICAL_FEATURES = [
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "initial_list_status",
    "state",
    "zip3",
]

RAW_COLUMNS_NEEDED = [
    "loan_amnt",
    "installment",
    "int_rate",
    "annual_inc",
    "dti",
    "open_acc",
    "pub_rec",
    "revol_bal",
    "revol_util",
    "total_acc",
    "mort_acc",
    "pub_rec_bankruptcies",
    "term",
    "emp_length",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "initial_list_status",
    "issue_d",
    "earliest_cr_line",
    "address",
    "zip_code",
    "state",
    "addr_state",
]

EAD_PROXY_COL = "loan_amnt"

BINARY_THRESHOLD_GRID = [round(x / 100, 2) for x in range(5, 100, 5)]
T_LOW_GRID = [round(x / 100, 2) for x in range(2, 26, 2)]
T_HIGH_GRID = [round(x / 100, 2) for x in range(15, 65, 3)]

LOGREG_PARAMS = {
    "solver": "saga",
    "max_iter": 2000,
    "random_state": RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 10,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

XGB_BASE_PARAMS = FROZEN_XGB_PARAMS.copy()

KNN_PARAMS = {
    "n_neighbors": 25,
    "weights": "distance",
}