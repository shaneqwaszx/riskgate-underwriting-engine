import numpy as np
import pandas as pd
import requests


def assign_decision(pd_scores: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    return np.where(
        pd_scores < t_low,
        "approve",
        np.where(pd_scores < t_high, "review", "reject")
    )


def build_scenario_grid_from_scores(
    base_scored_df: pd.DataFrame,
    t_low_values=None,
    t_high_values=None,
    min_band_width: float = 0.05
) -> pd.DataFrame:
    """
    Build a scenario table using already-generated PD scores.
    Requires columns:
    - pd_score
    - loan_amnt
    """
    if t_low_values is None:
        t_low_values = np.round(np.arange(0.05, 0.21, 0.02), 2)

    if t_high_values is None:
        t_high_values = np.round(np.arange(0.15, 0.41, 0.02), 2)

    pd_scores = base_scored_df["pd_score"].to_numpy()
    loan_amnt = pd.to_numeric(
        base_scored_df.get("loan_amnt", pd.Series(0, index=base_scored_df.index)),
        errors="coerce"
    ).fillna(0.0).to_numpy()

    rows = []

    for t_low in t_low_values:
        for t_high in t_high_values:
            if t_high <= t_low:
                continue
            if (t_high - t_low) < min_band_width:
                continue

            decisions = assign_decision(pd_scores, t_low, t_high)

            approve_mask = decisions == "approve"
            review_mask = decisions == "review"
            reject_mask = decisions == "reject"
            nonreject_mask = decisions != "reject"

            rows.append({
                "t_low": float(t_low),
                "t_high": float(t_high),
                "approve_rate": float(np.mean(approve_mask)),
                "review_rate": float(np.mean(review_mask)),
                "reject_rate": float(np.mean(reject_mask)),
                "avg_pd_approve": float(np.mean(pd_scores[approve_mask])) if approve_mask.sum() else np.nan,
                "avg_pd_review": float(np.mean(pd_scores[review_mask])) if review_mask.sum() else np.nan,
                "avg_pd_reject": float(np.mean(pd_scores[reject_mask])) if reject_mask.sum() else np.nan,
                "risk_proxy_nonreject": float(np.sum(pd_scores[nonreject_mask] * loan_amnt[nonreject_mask])),
                "risk_proxy_total": float(np.sum(pd_scores * loan_amnt)),
            })

    grid = pd.DataFrame(rows)

    if grid.empty:
        return grid

    risk_min = grid["risk_proxy_nonreject"].min()
    risk_max = grid["risk_proxy_nonreject"].max()
    denom = (risk_max - risk_min) if risk_max > risk_min else 1.0

    grid["risk_proxy_norm"] = (grid["risk_proxy_nonreject"] - risk_min) / denom
    return grid.sort_values(["t_low", "t_high"]).reset_index(drop=True)


def recommend_thresholds(
    grid: pd.DataFrame,
    goal: str,
    max_review_rate: float,
    min_approve_rate: float
) -> pd.DataFrame:
    """
    Deterministic recommendation engine.
    Targets act as both hard feasibility filters (when possible)
    and soft preferences through proximity scoring.
    """
    if grid.empty:
        return grid

    feasible = grid[
        (grid["review_rate"] <= max_review_rate) &
        (grid["approve_rate"] >= min_approve_rate)
    ].copy()

    used_fallback = False
    if feasible.empty:
        feasible = grid.copy()
        used_fallback = True

    # Constraint-gap diagnostics
    feasible["review_excess"] = np.maximum(0, feasible["review_rate"] - max_review_rate)
    feasible["approve_shortfall"] = np.maximum(0, min_approve_rate - feasible["approve_rate"])
    feasible["constraint_penalty"] = feasible["review_excess"] + feasible["approve_shortfall"]

    # Soft preference: among feasible candidates, prefer ones near the stated targets
    feasible["review_proximity"] = -np.abs(feasible["review_rate"] - max_review_rate)
    feasible["approve_proximity"] = -np.abs(feasible["approve_rate"] - min_approve_rate)

    if goal == "Balanced":
        feasible["score"] = (
            feasible["approve_rate"]
            - 0.50 * feasible["risk_proxy_norm"]
            + 0.20 * feasible["review_proximity"]
            + 0.15 * feasible["approve_proximity"]
            - 2.00 * feasible["constraint_penalty"]
        )

    elif goal == "Growth":
        feasible["score"] = (
            1.20 * feasible["approve_rate"]
            - 0.25 * feasible["risk_proxy_norm"]
            + 0.10 * feasible["review_proximity"]
            + 0.20 * feasible["approve_proximity"]
            - 2.00 * feasible["constraint_penalty"]
        )

    elif goal == "Conservative":
        feasible["score"] = (
            -1.10 * feasible["risk_proxy_norm"]
            + 0.10 * feasible["approve_rate"]
            + 0.10 * feasible["review_proximity"]
            + 0.10 * feasible["approve_proximity"]
            - 2.00 * feasible["constraint_penalty"]
        )

    elif goal == "Operations-first":
        feasible["score"] = (
            -1.20 * feasible["review_rate"]
            + 0.20 * feasible["approve_rate"]
            - 0.20 * feasible["risk_proxy_norm"]
            + 0.20 * feasible["review_proximity"]
            + 0.10 * feasible["approve_proximity"]
            - 2.00 * feasible["constraint_penalty"]
        )

    else:
        feasible["score"] = (
            feasible["approve_rate"]
            - 0.50 * feasible["risk_proxy_norm"]
            + 0.20 * feasible["review_proximity"]
            + 0.15 * feasible["approve_proximity"]
            - 2.00 * feasible["constraint_penalty"]
        )

    feasible["used_fallback"] = used_fallback

    return feasible.sort_values(
        ["constraint_penalty", "score"],
        ascending=[True, False]
    ).head(5)

def diagnose_assistant_constraints(
    grid: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    max_review_rate: float,
    min_approve_rate: float
) -> dict:
    if grid.empty:
        return {
            "feasible_count": 0,
            "review_target_binding": None,
            "approve_target_binding": None,
            "message": "No scenario grid available."
        }

    feasible = grid[
        (grid["review_rate"] <= max_review_rate) &
        (grid["approve_rate"] >= min_approve_rate)
    ].copy()

    if recommendation_df.empty:
        return {
            "feasible_count": int(len(feasible)),
            "review_target_binding": None,
            "approve_target_binding": None,
            "message": "No recommendation could be generated."
        }

    top = recommendation_df.iloc[0]

    review_gap = float(max_review_rate - top["review_rate"])
    approve_gap = float(top["approve_rate"] - min_approve_rate)

    review_binding = review_gap < 0.03
    approve_binding = approve_gap < 0.03

    if len(feasible) == 0:
        msg = (
            "No candidate threshold pair satisfies both targets exactly. "
            "The assistant is showing the closest available option."
        )
    else:
        parts = [f"{len(feasible)} scenario(s) satisfy your current targets."]
        if review_binding:
            parts.append("The review-rate target is binding or close to binding.")
        else:
            parts.append("The review-rate target is not currently binding.")
        if approve_binding:
            parts.append("The minimum approve-rate target is binding or close to binding.")
        else:
            parts.append("The minimum approve-rate target is not currently binding.")
        msg = " ".join(parts)

    return {
        "feasible_count": int(len(feasible)),
        "review_target_binding": review_binding,
        "approve_target_binding": approve_binding,
        "review_gap": review_gap,
        "approve_gap": approve_gap,
        "message": msg,
    }

def _get_secret(name: str, default: str = "") -> str:
    try:
        import streamlit as st
        return st.secrets.get(name, default)
    except Exception:
        return default


def ollama_explanation(prompt: str, model_name: str | None = None, host: str | None = None, api_key: str | None = None) -> str:
    """
    Hosted Ollama Cloud explanation using ollama.com API.
    This is advisory only.
    """
    try:
        resolved_host = host or _get_secret("OLLAMA_BASE_URL", "https://ollama.com/api")
        resolved_api_key = api_key or _get_secret("OLLAMA_API_KEY", "")
        resolved_model = model_name or _get_secret("OLLAMA_MODEL", "gpt-oss:120b")

        if not resolved_api_key:
            return "Ollama API key is not configured in Streamlit secrets."

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {resolved_api_key}",
        }

        payload = {
            "model": resolved_model,
            "prompt": prompt,
            "stream": False,
        }

        response = requests.post(
            f"{resolved_host.rstrip('/')}/generate",
            json=payload,
            headers=headers,
            timeout=90,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("response", "").strip()

    except Exception as e:
        return f"Ollama explanation unavailable: {e}"