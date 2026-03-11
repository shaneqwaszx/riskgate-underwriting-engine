import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

from riskgate import config as cfg
from riskgate.artifacts import save_dataframe, save_joblib, save_json, utc_now_iso
from riskgate.features import filter_to_binary_outcomes
from riskgate.modeling import (
    benchmark_models,
    build_pipeline,
    evaluate_probabilities,
    fit_calibrated_model,
    tune_xgboost,
)
from riskgate.policy import (
    build_binary_threshold_table,
    build_policy_grid,
    choose_policy_thresholds,
    summarize_policy,
)


def extract_final_xgb_params(search_best_estimator) -> dict:
    model = search_best_estimator.named_steps["model"]
    keep_keys = [
        "n_estimators", "max_depth", "learning_rate", "subsample",
        "colsample_bytree", "min_child_weight", "gamma", "reg_alpha",
        "reg_lambda", "objective", "eval_metric", "tree_method",
        "n_jobs", "random_state",
    ]
    params = model.get_params()
    return {k: params[k] for k in keep_keys if k in params}


def main():
    parser = argparse.ArgumentParser(description="Train RiskGate underwriting engine.")
    parser.add_argument("--train_csv", required=True, help="Path to labelled training CSV")
    parser.add_argument("--artifacts_dir", default=str(cfg.ARTIFACTS_DIR), help="Where to save artifacts")
    parser.add_argument("--xgb_search_iter", type=int, default=20, help="RandomizedSearch iterations for XGBoost")
    args = parser.parse_args()

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.train_csv)
    labeled_df = filter_to_binary_outcomes(raw_df)

    y = labeled_df[cfg.TARGET_BINARY_COL].copy()
    X = labeled_df.drop(columns=[cfg.TARGET_BINARY_COL]).copy()

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg.VALID_SIZE,
        stratify=y,
        random_state=cfg.RANDOM_STATE,
    )

    # --------------------------------------------------
    # Mode 1: frozen notebook recipe
    # --------------------------------------------------
    if cfg.USE_FROZEN_NOTEBOOK_RECIPE:
        chosen_model_family = cfg.FROZEN_MODEL_NAME
        chosen_xgb_params = cfg.FROZEN_XGB_PARAMS.copy()

        notebook_reference_df = pd.DataFrame([{
            "model_name": "xgb_frozen_from_notebook",
            "roc_auc_reference": cfg.NOTEBOOK_REFERENCE_METRICS["roc_auc"],
            "average_precision_reference": cfg.NOTEBOOK_REFERENCE_METRICS["average_precision"],
            "brier_uncalibrated_reference": cfg.NOTEBOOK_REFERENCE_METRICS["brier_uncalibrated"],
            "best_binary_threshold_reference": cfg.NOTEBOOK_REFERENCE_METRICS["best_binary_threshold"],
        }])
        save_dataframe(notebook_reference_df, artifacts_dir / "notebook_reference_metrics.csv")

    # --------------------------------------------------
    # Mode 2: fresh search / fresh benchmark
    # --------------------------------------------------
    else:
        fitted_models, benchmark_df = benchmark_models(X_train, y_train, X_valid, y_valid)
        save_dataframe(benchmark_df, artifacts_dir / "validation_benchmark.csv")

        xgb_search, search_results_df = tune_xgboost(X_train, y_train, n_iter=args.xgb_search_iter)
        save_dataframe(search_results_df, artifacts_dir / "xgb_random_search_results.csv")

        tuned_xgb = xgb_search.best_estimator_
        tuned_xgb_p = tuned_xgb.predict_proba(X_valid)[:, 1]
        tuned_xgb_row = pd.DataFrame([evaluate_probabilities(y_valid, tuned_xgb_p, "xgb_tuned")])

        benchmark_plus_tuned = pd.concat([benchmark_df, tuned_xgb_row], ignore_index=True).sort_values(
            by=["average_precision", "roc_auc", "brier"],
            ascending=[False, False, True],
        )
        save_dataframe(benchmark_plus_tuned, artifacts_dir / "validation_benchmark.csv")

        best_name = benchmark_plus_tuned.iloc[0]["model_name"]

        if best_name == "xgb_tuned":
            chosen_model_family = "xgb"
            chosen_xgb_params = extract_final_xgb_params(tuned_xgb)
        elif best_name == "xgb_base":
            chosen_model_family = "xgb"
            chosen_xgb_params = cfg.XGB_BASE_PARAMS.copy()
        elif best_name == "random_forest":
            chosen_model_family = "random_forest"
            chosen_xgb_params = None
        elif best_name == "logreg":
            chosen_model_family = "logreg"
            chosen_xgb_params = None
        else:
            raise ValueError(f"Unexpected best model name: {best_name}")

    # --------------------------------------------------
    # Fit calibrated model on development split
    # --------------------------------------------------
    chosen_base_estimator = build_pipeline(chosen_model_family, xgb_params=chosen_xgb_params)
    calibrated_model = fit_calibrated_model(chosen_base_estimator, X_train, y_train)

    valid_pd = calibrated_model.predict_proba(X_valid)[:, 1]
    calibrated_metrics = evaluate_probabilities(y_valid, valid_pd, f"{chosen_model_family}_calibrated")
    save_json(calibrated_metrics, artifacts_dir / "calibrated_validation_metrics.json")

    threshold_table = build_binary_threshold_table(y_valid.to_numpy(), valid_pd)
    save_dataframe(threshold_table, artifacts_dir / "threshold_sweep.csv")

    ead_valid = pd.to_numeric(X_valid[cfg.EAD_PROXY_COL], errors="coerce").fillna(0.0).to_numpy()

    # --------------------------------------------------
    # Threshold selection
    # --------------------------------------------------
    if cfg.USE_FROZEN_THRESHOLDS:
        if cfg.FROZEN_T_LOW is None or cfg.FROZEN_T_HIGH is None:
            raise ValueError("Set FROZEN_T_LOW and FROZEN_T_HIGH in config.py")

        chosen_policy = {
            "status": "frozen_from_notebook",
            "t_low": float(cfg.FROZEN_T_LOW),
            "t_high": float(cfg.FROZEN_T_HIGH),
            "source": "manual_notebook_freeze",
        }
    else:
        policy_grid = build_policy_grid(
            y_true=y_valid.to_numpy(),
            p_default=valid_pd,
            ead_proxy=ead_valid,
        )
        save_dataframe(policy_grid, artifacts_dir / "policy_grid.csv")
        chosen_policy = choose_policy_thresholds(policy_grid)

    save_json(chosen_policy, artifacts_dir / "threshold_config.json")

    policy_summary = summarize_policy(
        y_true=y_valid.to_numpy(),
        p_default=valid_pd,
        ead_proxy=ead_valid,
        t_low=chosen_policy["t_low"],
        t_high=chosen_policy["t_high"],
    )
    save_json(policy_summary, artifacts_dir / "policy_summary.json")

    # --------------------------------------------------
    # Final refit on all labelled data
    # --------------------------------------------------
    final_base_estimator = build_pipeline(chosen_model_family, xgb_params=chosen_xgb_params)
    final_calibrated_model = fit_calibrated_model(final_base_estimator, X, y)

    feature_schema = {
        "numeric_features": cfg.NUMERIC_FEATURES,
        "categorical_features": cfg.CATEGORICAL_FEATURES,
        "raw_columns_needed": cfg.RAW_COLUMNS_NEEDED,
        "ead_proxy_col": cfg.EAD_PROXY_COL,
    }
    save_json(feature_schema, artifacts_dir / "feature_schema.json")

    metadata = {
        "app_name": cfg.APP_NAME,
        "engine_version": cfg.ENGINE_VERSION,
        "created_at_utc": utc_now_iso(),
        "random_state": cfg.RANDOM_STATE,
        "train_csv": str(args.train_csv),
        "training_rows_total": int(len(X)),
        "training_rows_dev": int(len(X_train)),
        "validation_rows": int(len(X_valid)),
        "chosen_model_name": chosen_model_family,
        "chosen_xgb_params": chosen_xgb_params,
        "calibration_method": cfg.CALIBRATION_METHOD,
        "calibration_cv": cfg.CALIBRATION_CV,
        "target_col": cfg.TARGET_COL,
        "target_binary_col": cfg.TARGET_BINARY_COL,
        "label_map": cfg.LABEL_MAP,
        "policy_thresholds": {
            "t_low": chosen_policy["t_low"],
            "t_high": chosen_policy["t_high"],
        },
        "notebook_reference_metrics": cfg.NOTEBOOK_REFERENCE_METRICS,
        "notes": (
            "Frozen notebook recipe used for XGBoost model family and operating logic. "
            "Final saved artifact was rebuilt cleanly for reusable scoring."
        ),
    }
    save_json(metadata, artifacts_dir / "model_metadata.json")

    bundle = {
        "model": final_calibrated_model,
        "thresholds": {
            "t_low": chosen_policy["t_low"],
            "t_high": chosen_policy["t_high"],
        },
        "feature_schema": feature_schema,
        "metadata": metadata,
    }

    save_joblib(final_calibrated_model, artifacts_dir / "final_calibrated_model.joblib")
    save_joblib(bundle, artifacts_dir / "final_model_bundle.joblib")

    print("Training complete.")
    print(f"Frozen recipe model family: {chosen_model_family}")
    print(f"Frozen t_low={chosen_policy['t_low']:.2f}, t_high={chosen_policy['t_high']:.2f}")
    print(f"Artifacts saved to: {artifacts_dir.resolve()}")


if __name__ == "__main__":
    main()