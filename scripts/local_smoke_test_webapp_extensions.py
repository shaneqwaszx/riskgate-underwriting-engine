from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from webapp.assistant import assign_decision
from riskgate.drift import build_drift_report
from riskgate.history import append_run_history, run_history_df
from riskgate.reviewer import append_override_log, load_override_log


BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "final_model_bundle.joblib"
INPUT_PATH = PROJECT_ROOT / "data" / "scored" / "new_applications_to_score.csv"


def build_summary(scored_df: pd.DataFrame) -> dict:
    total = len(scored_df)
    counts = scored_df["decision"].value_counts().to_dict()

    approve_n = counts.get("approve", 0)
    review_n = counts.get("review", 0)
    reject_n = counts.get("reject", 0)

    return {
        "total_applicants": total,
        "approve_n": approve_n,
        "review_n": review_n,
        "reject_n": reject_n,
        "approve_rate": approve_n / total if total else 0.0,
        "review_rate": review_n / total if total else 0.0,
        "reject_rate": reject_n / total if total else 0.0,
        "avg_pd_overall": float(scored_df["pd_score"].mean()),
        "avg_pd_approve": float(scored_df.loc[scored_df["decision"] == "approve", "pd_score"].mean()) if approve_n else np.nan,
        "risk_proxy_nonreject": float(
            scored_df.loc[
                scored_df["decision"].isin(["approve", "review"]),
                "risk_proxy_pd_x_loan_amnt"
            ].sum()
        ),
    }


def main():
    print("Loading bundle...")
    bundle = joblib.load(BUNDLE_PATH)
    model = bundle["model"]
    thresholds = bundle["thresholds"]

    print("Loading default input...")
    df = pd.read_csv(INPUT_PATH)

    print("Scoring...")
    pd_scores = model.predict_proba(df)[:, 1]
    decisions = assign_decision(pd_scores, thresholds["t_low"], thresholds["t_high"])

    scored_df = df.copy()
    scored_df["pd_score"] = pd_scores
    scored_df["decision"] = decisions
    scored_df["risk_proxy_pd_x_loan_amnt"] = (
        pd_scores * pd.to_numeric(scored_df["loan_amnt"], errors="coerce").fillna(0.0)
    )

    summary = build_summary(scored_df)
    assert summary["total_applicants"] == len(df)
    assert {"approve", "review", "reject"}.intersection(set(scored_df["decision"].unique()))

    print("Testing drift report...")
    sample_df = df.sample(min(len(df), 3000), random_state=42).copy()
    drift_report = build_drift_report(df, sample_df)

    assert "missingness_table" in drift_report
    assert "numeric_psi_table" in drift_report
    assert "unseen_category_table" in drift_report

    print("Testing run history...")
    fake_session = {}
    run_info = {
        "input_name": str(INPUT_PATH.name),
        "rows_scored": len(df),
        "t_low_used": thresholds["t_low"],
        "t_high_used": thresholds["t_high"],
        "use_frozen_thresholds": True,
    }
    append_run_history(fake_session, run_info, summary)
    history_df = run_history_df(fake_session)
    assert len(history_df) == 1

    print("Testing reviewer override log...")
    with tempfile.TemporaryDirectory() as td:
        override_path = Path(td) / "reviewer_overrides.csv"
        append_override_log(
            override_path,
            {
                "reviewer_name": "local_test",
                "input_name": INPUT_PATH.name,
                "record_id": "demo_001",
                "pd_score": 0.213,
                "original_decision": "review",
                "override_decision": "approve",
                "reviewer_note": "smoke test",
                "t_low_used": thresholds["t_low"],
                "t_high_used": thresholds["t_high"],
            },
        )
        override_df = load_override_log(override_path)
        assert len(override_df) == 1

    print("All extension smoke tests passed.")


if __name__ == "__main__":
    main()