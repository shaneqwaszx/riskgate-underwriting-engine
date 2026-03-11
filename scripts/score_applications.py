import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from riskgate.artifacts import load_joblib, utc_now_iso
from riskgate.policy import assign_underwriting_bucket


def main():
    parser = argparse.ArgumentParser(description="Score new applications with RiskGate bundle.")
    parser.add_argument("--input_csv", required=True, help="Raw new applications CSV")
    parser.add_argument("--bundle_path", required=True, help="Path to final_model_bundle.joblib")
    parser.add_argument("--output_csv", required=True, help="Where to write scored decision CSV")
    parser.add_argument("--id_col", default=None, help="Optional application identifier column")
    args = parser.parse_args()

    bundle = load_joblib(Path(args.bundle_path))
    model = bundle["model"]
    t_low = bundle["thresholds"]["t_low"]
    t_high = bundle["thresholds"]["t_high"]
    metadata = bundle["metadata"]

    raw_df = pd.read_csv(args.input_csv)
    pd_score = model.predict_proba(raw_df)[:, 1]
    decision = assign_underwriting_bucket(pd_score, t_low=t_low, t_high=t_high)

    if args.id_col and args.id_col in raw_df.columns:
        record_id = raw_df[args.id_col]
    else:
        record_id = pd.Series(raw_df.index, name="record_id")

    ead_proxy = pd.to_numeric(raw_df.get("loan_amnt", pd.Series(np.nan, index=raw_df.index)), errors="coerce")
    risk_proxy = pd_score * ead_proxy.fillna(0.0)

    reason_code = np.where(
        pd_score < t_low,
        "pd_below_t_low",
        np.where(pd_score < t_high, "pd_in_review_band", "pd_above_t_high")
    )

    out = pd.DataFrame({
        "record_id": record_id,
        "pd_score": pd_score,
        "decision": decision,
        "decision_reason_code": reason_code,
        "t_low": t_low,
        "t_high": t_high,
        "ead_proxy_loan_amnt": ead_proxy,
        "risk_proxy_pd_x_loan_amnt": risk_proxy,
        "model_name": metadata["chosen_model_name"],
        "engine_version": metadata["engine_version"],
        "scored_at_utc": utc_now_iso(),
    })

    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Scored {len(out)} rows.")
    print(f"Output written to: {Path(args.output_csv).resolve()}")


if __name__ == "__main__":
    main()