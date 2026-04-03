from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


OVERRIDE_COLUMNS = [
    "logged_at_utc",
    "reviewer_name",
    "input_name",
    "record_id",
    "pd_score",
    "original_decision",
    "override_decision",
    "reviewer_note",
    "t_low_used",
    "t_high_used",
]


def append_override_log(output_path: Path, row: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {col: row.get(col, None) for col in OVERRIDE_COLUMNS}
    payload["logged_at_utc"] = datetime.now(timezone.utc).isoformat()

    out_df = pd.DataFrame([payload], columns=OVERRIDE_COLUMNS)
    write_header = not output_path.exists()

    out_df.to_csv(
        output_path,
        mode="a",
        index=False,
        header=write_header,
    )


def load_override_log(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame(columns=OVERRIDE_COLUMNS)

    return pd.read_csv(output_path)