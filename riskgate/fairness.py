from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


DEFAULT_SEGMENT_COLS = [
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "state",
    "addr_state",
    "zip3",
    "grade",
    "sub_grade",
]

PROTECTED_HINT_COLS = [
    "gender",
    "sex",
    "race",
    "ethnicity",
    "age_band",
    "age_group",
]


def with_proxy_columns(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df.copy()

    if "zip3" not in out.columns and "zip_code" in out.columns:
        out["zip3"] = (
            out["zip_code"]
            .astype("string")
            .str.extract(r"(\d{3})", expand=False)
        )

    return out


def get_candidate_group_cols(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    df = with_proxy_columns(raw_df)
    cols = [c for c in DEFAULT_SEGMENT_COLS if c in df.columns]
    return df, cols


def build_group_outcome_table(
    raw_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    group_col: str,
    min_group_size: int = 200,
) -> pd.DataFrame:
    df = with_proxy_columns(raw_df).copy()
    df[group_col] = df[group_col].astype("string").fillna("MISSING")
    df["_decision"] = scored_df["decision"].to_numpy()
    df["_pd_score"] = scored_df["pd_score"].to_numpy()

    out = (
        df.groupby(group_col, dropna=False)
        .agg(
            n=(group_col, "size"),
            approve_rate=("_decision", lambda s: (s == "approve").mean()),
            review_rate=("_decision", lambda s: (s == "review").mean()),
            reject_rate=("_decision", lambda s: (s == "reject").mean()),
            avg_pd=("_pd_score", "mean"),
        )
        .reset_index()
    )

    out = out[out["n"] >= min_group_size].copy()

    if out.empty:
        return out

    overall_approve = (df["_decision"] == "approve").mean()
    overall_reject = (df["_decision"] == "reject").mean()
    best_approve = out["approve_rate"].max()

    out["approve_gap_vs_overall"] = out["approve_rate"] - overall_approve
    out["reject_gap_vs_overall"] = out["reject_rate"] - overall_reject
    out["approve_ratio_vs_best"] = np.where(
        best_approve > 0,
        out["approve_rate"] / best_approve,
        np.nan,
    )

    return out.sort_values("approve_rate", ascending=True).reset_index(drop=True)


def build_feat_assessment_table(
    raw_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    metadata: Dict,
    has_reason_codes: bool = True,
    has_explanations: bool = False,
    has_override_logging: bool = False,
) -> pd.DataFrame:
    df = with_proxy_columns(raw_df)

    has_protected = any(c in df.columns for c in PROTECTED_HINT_COLS)
    has_proxy_groups = any(c in df.columns for c in DEFAULT_SEGMENT_COLS)

    fairness_status = (
        "Measured on supplied groups"
        if has_protected
        else "Proxy diagnostics only"
        if has_proxy_groups
        else "Not assessed"
    )

    fairness_note = (
        "Protected or sensitive group fields were supplied, so direct disparity review is possible."
        if has_protected
        else "Only proxy / business-segment diagnostics are available in the current input."
        if has_proxy_groups
        else "No suitable grouping variables were supplied for fairness diagnostics."
    )

    accountability_status = "Present"
    accountability_note = (
        "Thresholds, model metadata, and runtime diagnostics are logged in the app."
    )

    transparency_status = (
        "Strong" if has_reason_codes and has_explanations else
        "Partial" if has_reason_codes else
        "Weak"
    )
    transparency_note = (
        "Reason codes and feature-level explanations are available."
        if has_reason_codes and has_explanations else
        "Reason codes are available, but feature-level explanations are not yet enabled."
        if has_reason_codes else
        "No explanation artifacts are currently exposed."
    )

    ethics_status = "Partial" if has_override_logging else "Limited"
    ethics_note = (
        "Human review / override logging is available."
        if has_override_logging else
        "Manual review exists conceptually, but structured reviewer override logging is not yet implemented."
    )

    rows = [
        {
            "FEAT dimension": "Fairness",
            "Status": fairness_status,
            "Current evidence": fairness_note,
            "Next action": "Review group-level outcome gaps and extend to protected attributes where appropriate.",
        },
        {
            "FEAT dimension": "Ethics",
            "Status": ethics_status,
            "Current evidence": ethics_note,
            "Next action": "Add reviewer override capture and escalation notes.",
        },
        {
            "FEAT dimension": "Accountability",
            "Status": accountability_status,
            "Current evidence": accountability_note,
            "Next action": "Add persistent run registry and approval of policy changes.",
        },
        {
            "FEAT dimension": "Transparency",
            "Status": transparency_status,
            "Current evidence": transparency_note,
            "Next action": "Expose applicant-level drivers and explanation downloads.",
        },
    ]

    return pd.DataFrame(rows)

def discover_candidate_group_cols(
    raw_df: pd.DataFrame,
    max_unique: int = 20,
    min_unique: int = 2,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return only sensible low/medium-cardinality segment columns for disparity review.
    """
    df = with_proxy_columns(raw_df)

    candidate_pool = [
        "home_ownership",
        "verification_status",
        "purpose",
        "application_type",
        "grade",
        "sub_grade",
        "state",
        "addr_state",
        "zip3",
        "term",
        "emp_length",
    ]

    usable = []
    for col in candidate_pool:
        if col not in df.columns:
            continue

        nunique = (
            df[col]
            .astype("string")
            .fillna("MISSING")
            .str.strip()
            .nunique(dropna=False)
        )

        if min_unique <= nunique <= max_unique:
            usable.append(col)

    return df, usable


def build_all_group_outcome_tables(
    raw_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    min_group_size: int = 200,
    max_unique: int = 20,
) -> Tuple[pd.DataFrame, List[str], Dict[str, pd.DataFrame]]:
    """
    Precompute all group disparity tables for the current run.
    """
    df, candidate_cols = discover_candidate_group_cols(
        raw_df=raw_df,
        max_unique=max_unique,
    )

    table_map: Dict[str, pd.DataFrame] = {}
    for col in candidate_cols:
        table_map[col] = build_group_outcome_table(
            raw_df=df,
            scored_df=scored_df,
            group_col=col,
            min_group_size=min_group_size,
        )

    return df, candidate_cols, table_map