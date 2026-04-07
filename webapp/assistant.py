import numpy as np
import pandas as pd
import requests

CUSTOM_PROFILE_NAME = "Custom target-driven"

PROFILE_SETTINGS = {
    "Balanced": {
        "scoring_goal": "Balanced",
        "review_band": (0.08, 0.20),
        "approve_band": (0.45, 0.60),
    },
    "Growth": {
        "scoring_goal": "Growth",
        "review_band": (0.08, 0.25),
        "approve_band": (0.60, 0.80),
    },
    "Conservative": {
        "scoring_goal": "Conservative",
        "review_band": (0.05, 0.15),
        "approve_band": (0.35, 0.55),
    },
    "Operations-first": {
        "scoring_goal": "Operations-first",
        "review_band": (0.03, 0.12),
        "approve_band": (0.40, 0.60),
    },
}


def get_assistant_profile_settings(
    profile_name: str,
    user_review_band: tuple[float, float],
    user_approve_band: tuple[float, float],
) -> dict:
    """
    Fixed profiles ignore the user's target bands.
    Only Custom target-driven uses the user-selected ranges.
    """
    if profile_name == CUSTOM_PROFILE_NAME:
        review_min, review_max = sorted([float(user_review_band[0]), float(user_review_band[1])])
        approve_min, approve_max = sorted([float(user_approve_band[0]), float(user_approve_band[1])])

        return {
            "profile_name": CUSTOM_PROFILE_NAME,
            "scoring_goal": "Balanced",
            "min_review_rate": review_min,
            "max_review_rate": review_max,
            "min_approve_rate": approve_min,
            "max_approve_rate": approve_max,
            "uses_user_targets": True,
            "mode_label": "User-defined target bands",
        }

    if profile_name not in PROFILE_SETTINGS:
        raise ValueError(f"Unknown profile: {profile_name}")

    spec = PROFILE_SETTINGS[profile_name]
    review_min, review_max = spec["review_band"]
    approve_min, approve_max = spec["approve_band"]

    return {
        "profile_name": profile_name,
        "scoring_goal": spec["scoring_goal"],
        "min_review_rate": float(review_min),
        "max_review_rate": float(review_max),
        "min_approve_rate": float(approve_min),
        "max_approve_rate": float(approve_max),
        "uses_user_targets": False,
        "mode_label": "Fixed profile target bands",
    }

def _band_penalty(x: pd.Series, lower: float, upper: float) -> pd.Series:
    return np.maximum(lower - x, 0.0) + np.maximum(x - upper, 0.0)


def _band_proximity(x: pd.Series, lower: float, upper: float) -> pd.Series:
    mid = (lower + upper) / 2.0
    half_width = max((upper - lower) / 2.0, 1e-6)
    return np.clip(1.0 - (np.abs(x - mid) / half_width), 0.0, 1.0)

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

    # ensure sorted unique values
    t_low_values = np.sort(np.unique(np.round(np.array(t_low_values, dtype=float), 2)))
    t_high_values = np.sort(np.unique(np.round(np.array(t_high_values, dtype=float), 2)))

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
    min_review_rate: float = 0.05,
    max_review_rate: float = 0.20,
    min_approve_rate: float = 0.50,
    max_approve_rate: float = 0.70,
) -> pd.DataFrame:
    g = grid.copy()

    g["review_proximity"] = _band_proximity(g["review_rate"], min_review_rate, max_review_rate)
    g["approve_proximity"] = _band_proximity(g["approve_rate"], min_approve_rate, max_approve_rate)

    g["constraint_penalty"] = (
        _band_penalty(g["review_rate"], min_review_rate, max_review_rate) +
        _band_penalty(g["approve_rate"], min_approve_rate, max_approve_rate)
    )

    feasible = g[
        (g["review_rate"] >= min_review_rate) &
        (g["review_rate"] <= max_review_rate) &
        (g["approve_rate"] >= min_approve_rate) &
        (g["approve_rate"] <= max_approve_rate)
    ].copy()

    candidate_pool = feasible if not feasible.empty else g.copy()

    if goal == "Balanced":
        candidate_pool["score"] = (
            candidate_pool["approve_rate"]
            - 0.50 * candidate_pool["risk_proxy_norm"]
            + 0.20 * candidate_pool["review_proximity"]
            + 0.15 * candidate_pool["approve_proximity"]
            - 2.00 * candidate_pool["constraint_penalty"]
        )

    elif goal == "Growth":
        candidate_pool["score"] = (
            1.20 * candidate_pool["approve_rate"]
            - 0.25 * candidate_pool["risk_proxy_norm"]
            + 0.10 * candidate_pool["review_proximity"]
            + 0.20 * candidate_pool["approve_proximity"]
            - 2.00 * candidate_pool["constraint_penalty"]
        )

    elif goal == "Conservative":
        candidate_pool["score"] = (
            -1.10 * candidate_pool["risk_proxy_norm"]
            + 0.10 * candidate_pool["approve_rate"]
            + 0.10 * candidate_pool["review_proximity"]
            + 0.10 * candidate_pool["approve_proximity"]
            - 2.00 * candidate_pool["constraint_penalty"]
        )

    elif goal == "Operations-first":
        candidate_pool["score"] = (
            -1.20 * candidate_pool["review_rate"]
            + 0.20 * candidate_pool["approve_rate"]
            - 0.20 * candidate_pool["risk_proxy_norm"]
            + 0.20 * candidate_pool["review_proximity"]
            + 0.10 * candidate_pool["approve_proximity"]
            - 2.00 * candidate_pool["constraint_penalty"]
        )

    else:
        raise ValueError(f"Unknown goal: {goal}")

    candidate_pool["goal"] = goal
    return candidate_pool.sort_values("score", ascending=False).head(1)

def diagnose_assistant_constraints(
    grid: pd.DataFrame,
    recommendation_df: pd.DataFrame,
    min_review_rate: float,
    max_review_rate: float,
    min_approve_rate: float,
    max_approve_rate: float,
) -> dict:
    if grid.empty:
        return {
            "feasible_count": 0,
            "message": "No scenario grid available."
        }

    feasible = grid[
        (grid["review_rate"] >= min_review_rate) &
        (grid["review_rate"] <= max_review_rate) &
        (grid["approve_rate"] >= min_approve_rate) &
        (grid["approve_rate"] <= max_approve_rate)
    ].copy()

    if recommendation_df.empty:
        return {
            "feasible_count": int(len(feasible)),
            "message": "No recommendation could be generated."
        }

    top = recommendation_df.iloc[0]

    top_review_rate = float(top["review_rate"])
    top_approve_rate = float(top["approve_rate"])

    review_within = min_review_rate <= top_review_rate <= max_review_rate
    approve_within = min_approve_rate <= top_approve_rate <= max_approve_rate

    if len(feasible) == 0:
        msg = (
            "No threshold pair satisfies the current effective target bands exactly. "
            "The assistant is showing the closest available option."
        )
    else:
        msg = f"{len(feasible)} scenario(s) satisfy the current effective target bands."

    return {
        "feasible_count": int(len(feasible)),
        "message": msg,
        "top_review_rate": top_review_rate,
        "top_approve_rate": top_approve_rate,
        "review_within": review_within,
        "approve_within": approve_within,
        "review_band_text": f"{top_review_rate:.1%} vs target band {min_review_rate:.1%}–{max_review_rate:.1%}",
        "approve_band_text": f"{top_approve_rate:.1%} vs target band {min_approve_rate:.1%}–{max_approve_rate:.1%}",
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