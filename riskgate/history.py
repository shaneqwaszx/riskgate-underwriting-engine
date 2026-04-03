from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd


RUN_HISTORY_COLUMNS = [
    "run_at_utc",
    "input_name",
    "rows_scored",
    "t_low_used",
    "t_high_used",
    "use_frozen_thresholds",
    "approve_n",
    "review_n",
    "reject_n",
    "approve_rate",
    "review_rate",
    "reject_rate",
    "avg_pd_overall",
    "avg_pd_approve",
    "risk_proxy_nonreject",
]


def make_run_record(run_info: dict, summary: dict) -> dict:
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": run_info["input_name"],
        "rows_scored": run_info["rows_scored"],
        "t_low_used": run_info["t_low_used"],
        "t_high_used": run_info["t_high_used"],
        "use_frozen_thresholds": run_info["use_frozen_thresholds"],
        "approve_n": summary["approve_n"],
        "review_n": summary["review_n"],
        "reject_n": summary["reject_n"],
        "approve_rate": summary["approve_rate"],
        "review_rate": summary["review_rate"],
        "reject_rate": summary["reject_rate"],
        "avg_pd_overall": summary["avg_pd_overall"],
        "avg_pd_approve": summary["avg_pd_approve"],
        "risk_proxy_nonreject": summary["risk_proxy_nonreject"],
    }


def append_run_history(session_state, run_info: dict, summary: dict, max_runs: int = 20) -> None:
    history = list(session_state.get("run_history", []))
    history.append(make_run_record(run_info, summary))
    session_state["run_history"] = history[-max_runs:]


def run_history_df(session_state) -> pd.DataFrame:
    rows = list(session_state.get("run_history", []))
    if not rows:
        return pd.DataFrame(columns=RUN_HISTORY_COLUMNS)

    df = pd.DataFrame(rows)
    cols = [c for c in RUN_HISTORY_COLUMNS if c in df.columns]
    return df[cols].copy()