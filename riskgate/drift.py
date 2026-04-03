from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_NUMERIC_DRIFT_COLS = [
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
]

DEFAULT_CATEGORICAL_DRIFT_COLS = [
    "term",
    "emp_length",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "initial_list_status",
    "state",
    "addr_state",
    "zip_code",
]


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _severity_from_psi(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if value < 0.10:
        return "Low"
    if value < 0.25:
        return "Moderate"
    return "High"


def _compute_psi(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-6) -> float:
    ref = _safe_numeric(reference).dropna()
    cur = _safe_numeric(current).dropna()

    if len(ref) < 20 or len(cur) < 20:
        return np.nan

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))

    if len(edges) < 3:
        return 0.0

    edges = edges.astype(float)
    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_bins = pd.cut(ref, bins=edges, include_lowest=True)
    cur_bins = pd.cut(cur, bins=edges, include_lowest=True)

    ref_dist = ref_bins.value_counts(normalize=True, sort=False)
    cur_dist = cur_bins.value_counts(normalize=True, sort=False)

    ref_dist = ref_dist.reindex(ref_dist.index.union(cur_dist.index), fill_value=0.0)
    cur_dist = cur_dist.reindex(ref_dist.index, fill_value=0.0)

    ref_arr = np.clip(ref_dist.to_numpy(dtype=float), eps, None)
    cur_arr = np.clip(cur_dist.to_numpy(dtype=float), eps, None)

    return float(np.sum((cur_arr - ref_arr) * np.log(cur_arr / ref_arr)))


def build_missingness_table(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    shared_cols = sorted(set(reference_df.columns).intersection(current_df.columns))
    rows = []

    for col in shared_cols:
        ref_missing = float(reference_df[col].isna().mean())
        cur_missing = float(current_df[col].isna().mean())

        rows.append({
            "column": col,
            "reference_missing_rate": ref_missing,
            "current_missing_rate": cur_missing,
            "delta_missing_rate": cur_missing - ref_missing,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("delta_missing_rate", ascending=False).reset_index(drop=True)


def build_numeric_psi_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    numeric_cols: List[str] | None = None,
) -> pd.DataFrame:
    candidate_cols = numeric_cols or DEFAULT_NUMERIC_DRIFT_COLS
    cols = [c for c in candidate_cols if c in reference_df.columns and c in current_df.columns]

    rows = []
    for col in cols:
        psi_value = _compute_psi(reference_df[col], current_df[col])

        rows.append({
            "column": col,
            "psi": psi_value,
            "severity": _severity_from_psi(psi_value),
            "reference_mean": float(_safe_numeric(reference_df[col]).mean()),
            "current_mean": float(_safe_numeric(current_df[col]).mean()),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("psi", ascending=False).reset_index(drop=True)


def build_unseen_category_table(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    categorical_cols: List[str] | None = None,
    max_examples: int = 8,
) -> pd.DataFrame:
    candidate_cols = categorical_cols or DEFAULT_CATEGORICAL_DRIFT_COLS
    cols = [c for c in candidate_cols if c in reference_df.columns and c in current_df.columns]

    rows = []
    for col in cols:
        ref_vals = (
            reference_df[col]
            .astype("string")
            .fillna("MISSING")
            .str.strip()
        )
        cur_vals = (
            current_df[col]
            .astype("string")
            .fillna("MISSING")
            .str.strip()
        )

        ref_set = set(ref_vals.unique().tolist())
        cur_unique = cur_vals.unique().tolist()
        unseen = sorted([x for x in cur_unique if x not in ref_set])

        unseen_mask = ~cur_vals.isin(ref_set)
        unseen_rate = float(unseen_mask.mean()) if len(cur_vals) else 0.0

        rows.append({
            "column": col,
            "unseen_rate": unseen_rate,
            "n_unseen_categories": len(unseen),
            "example_unseen_values": ", ".join(unseen[:max_examples]) if unseen else "",
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values("unseen_rate", ascending=False).reset_index(drop=True)


def build_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> Dict[str, object]:
    ref_cols = set(reference_df.columns)
    cur_cols = set(current_df.columns)

    reference_only_cols = sorted(ref_cols - cur_cols)
    current_only_cols = sorted(cur_cols - ref_cols)
    shared_cols = sorted(ref_cols.intersection(cur_cols))

    schema_summary = pd.DataFrame([
        {"item": "shared_columns", "value": len(shared_cols)},
        {"item": "reference_only_columns", "value": len(reference_only_cols)},
        {"item": "current_only_columns", "value": len(current_only_cols)},
        {"item": "reference_rows", "value": len(reference_df)},
        {"item": "current_rows", "value": len(current_df)},
    ])

    missingness_table = build_missingness_table(reference_df, current_df)
    numeric_psi_table = build_numeric_psi_table(reference_df, current_df)
    unseen_category_table = build_unseen_category_table(reference_df, current_df)

    return {
        "schema_summary": schema_summary,
        "reference_only_cols": reference_only_cols,
        "current_only_cols": current_only_cols,
        "missingness_table": missingness_table,
        "numeric_psi_table": numeric_psi_table,
        "unseen_category_table": unseen_category_table,
    }