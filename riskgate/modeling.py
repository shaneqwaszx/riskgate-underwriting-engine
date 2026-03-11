from typing import Dict, Tuple, Any

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from riskgate import config as cfg
from riskgate.features import LoanFeatureBuilder


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        # Backward-compatible fallback
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, cfg.NUMERIC_FEATURES),
            ("cat", categorical_pipe, cfg.CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return preprocessor


def build_pipeline(model_name: str, xgb_params: Dict[str, Any] | None = None) -> Pipeline:
    preprocessor = make_preprocessor()

    if model_name == "logreg":
        estimator = LogisticRegression(**cfg.LOGREG_PARAMS)
    elif model_name == "random_forest":
        estimator = RandomForestClassifier(**cfg.RF_PARAMS)
    elif model_name == "xgb":
        params = cfg.XGB_BASE_PARAMS.copy()
        if xgb_params:
            params.update(xgb_params)
        estimator = XGBClassifier(**params)
    elif model_name == "knn":
        estimator = KNeighborsClassifier(**cfg.KNN_PARAMS)
    else:
        raise ValueError(f"Unknown model_name={model_name}")

    pipe = Pipeline(
        steps=[
            ("features", LoanFeatureBuilder()),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )
    return pipe


def make_calibrator(base_estimator, method: str = cfg.CALIBRATION_METHOD, cv: int = cfg.CALIBRATION_CV):
    try:
        return CalibratedClassifierCV(estimator=base_estimator, method=method, cv=cv)
    except TypeError:
        # Backward-compatible fallback
        return CalibratedClassifierCV(base_estimator=base_estimator, method=method, cv=cv)


def evaluate_probabilities(y_true, p_default, model_name: str) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "roc_auc": roc_auc_score(y_true, p_default),
        "average_precision": average_precision_score(y_true, p_default),
        "brier": brier_score_loss(y_true, p_default),
    }


def benchmark_models(X_train: pd.DataFrame, y_train: pd.Series,
                     X_valid: pd.DataFrame, y_valid: pd.Series) -> Tuple[Dict[str, Any], pd.DataFrame]:
    fitted = {}
    rows = []

    candidates = {
        "logreg": build_pipeline("logreg"),
        "random_forest": build_pipeline("random_forest"),
        "xgb_base": build_pipeline("xgb"),
    }

    if cfg.ENABLE_KNN:
        candidates["knn"] = build_pipeline("knn")

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        p = pipe.predict_proba(X_valid)[:, 1]
        fitted[name] = pipe
        rows.append(evaluate_probabilities(y_valid, p, name))

    benchmark_df = pd.DataFrame(rows).sort_values(
        by=["average_precision", "roc_auc", "brier"],
        ascending=[False, False, True],
    )
    return fitted, benchmark_df


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_iter: int = 20, cv_splits: int = 3):
    pipe = build_pipeline("xgb")

    param_distributions = {
        "model__n_estimators": [250, 400, 600],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.05, 0.08, 0.10],
        "model__subsample": [0.70, 0.85, 1.00],
        "model__colsample_bytree": [0.60, 0.80, 1.00],
        "model__min_child_weight": [1, 3, 5],
        "model__gamma": [0.0, 0.5, 1.0],
        "model__reg_alpha": [0.0, 0.1, 1.0],
        "model__reg_lambda": [1.0, 2.0, 5.0],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=cfg.RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="average_precision",
        n_jobs=-1,
        cv=cv,
        random_state=cfg.RANDOM_STATE,
        verbose=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    cv_results = pd.DataFrame(search.cv_results_).sort_values(by="rank_test_score")
    return search, cv_results


def fit_calibrated_model(base_estimator, X_train: pd.DataFrame, y_train: pd.Series):
    calibrated = make_calibrator(base_estimator, method=cfg.CALIBRATION_METHOD, cv=cfg.CALIBRATION_CV)
    calibrated.fit(X_train, y_train)
    return calibrated