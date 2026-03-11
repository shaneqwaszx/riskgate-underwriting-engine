from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from riskgate import config as cfg


def assign_underwriting_bucket(p_default: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    return np.where(
        p_default < t_low,
        "approve",
        np.where(p_default < t_high, "review", "reject")
    )


def build_binary_threshold_table(y_true: np.ndarray, p_default: np.ndarray,
                                 threshold_grid: list[float] | None = None) -> pd.DataFrame:
    if threshold_grid is None:
        threshold_grid = cfg.BINARY_THRESHOLD_GRID

    rows = []
    for t in threshold_grid:
        y_pred = (p_default >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        rows.append({
            "threshold": t,
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
            "approval_rate_pred0": float((y_pred == 0).mean()),
            "flag_rate_pred1": float((y_pred == 1).mean()),
        })

    return pd.DataFrame(rows)


def summarize_policy(y_true: np.ndarray,
                     p_default: np.ndarray,
                     ead_proxy: np.ndarray,
                     t_low: float,
                     t_high: float) -> Dict[str, Any]:
    bucket = assign_underwriting_bucket(p_default, t_low, t_high)

    approve_mask = bucket == "approve"
    review_mask = bucket == "review"
    reject_mask = bucket == "reject"
    non_reject_mask = ~reject_mask

    def safe_mean(arr, mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(arr[mask]))

    return {
        "t_low": float(t_low),
        "t_high": float(t_high),
        "approve_rate": float(np.mean(approve_mask)),
        "review_rate": float(np.mean(review_mask)),
        "reject_rate": float(np.mean(reject_mask)),
        "approve_observed_default_rate": safe_mean(y_true, approve_mask),
        "review_observed_default_rate": safe_mean(y_true, review_mask),
        "reject_observed_default_rate": safe_mean(y_true, reject_mask),
        "nonrejected_observed_default_rate": safe_mean(y_true, non_reject_mask),
        "approve_avg_pd": safe_mean(p_default, approve_mask),
        "review_avg_pd": safe_mean(p_default, review_mask),
        "reject_avg_pd": safe_mean(p_default, reject_mask),
        "risk_proxy_nonrejected": float(np.sum(p_default[non_reject_mask] * ead_proxy[non_reject_mask])),
        "risk_proxy_approved_only": float(np.sum(p_default[approve_mask] * ead_proxy[approve_mask])),
    }


def build_policy_grid(y_true: np.ndarray,
                      p_default: np.ndarray,
                      ead_proxy: np.ndarray,
                      t_low_grid: list[float] | None = None,
                      t_high_grid: list[float] | None = None) -> pd.DataFrame:
    if t_low_grid is None:
        t_low_grid = cfg.T_LOW_GRID
    if t_high_grid is None:
        t_high_grid = cfg.T_HIGH_GRID

    rows = []
    for t_low in t_low_grid:
        for t_high in t_high_grid:
            if t_high <= t_low:
                continue
            if (t_high - t_low) < cfg.MIN_REVIEW_BAND_WIDTH:
                continue

            rows.append(
                summarize_policy(
                    y_true=y_true,
                    p_default=p_default,
                    ead_proxy=ead_proxy,
                    t_low=t_low,
                    t_high=t_high,
                )
            )

    grid = pd.DataFrame(rows)
    return grid.sort_values(["t_low", "t_high"]).reset_index(drop=True)


def choose_policy_thresholds(policy_grid: pd.DataFrame,
                             review_rate_cap: float = cfg.REVIEW_RATE_CAP,
                             max_auto_bad_rate: float = cfg.MAX_AUTO_APPROVE_BAD_RATE,
                             risk_weight: float = cfg.POLICY_RISK_WEIGHT,
                             reject_quality_weight: float = cfg.POLICY_REJECT_QUALITY_WEIGHT) -> Dict[str, Any]:
    grid = policy_grid.copy()

    grid["approve_observed_default_rate"] = grid["approve_observed_default_rate"].fillna(1.0)
    grid["reject_observed_default_rate"] = grid["reject_observed_default_rate"].fillna(0.0)

    grid["is_feasible"] = (
        (grid["review_rate"] <= review_rate_cap) &
        (grid["approve_observed_default_rate"] <= max_auto_bad_rate)
    )

    # Normalize the risk proxy so the score is scale-stable
    risk_min = grid["risk_proxy_nonrejected"].min()
    risk_max = grid["risk_proxy_nonrejected"].max()
    denom = (risk_max - risk_min) if risk_max > risk_min else 1.0
    grid["risk_proxy_norm"] = (grid["risk_proxy_nonrejected"] - risk_min) / denom

    # Higher is better
    grid["selection_score"] = (
        grid["approve_rate"]
        - risk_weight * grid["risk_proxy_norm"]
        + reject_quality_weight * grid["reject_observed_default_rate"]
    )

    feasible = grid.loc[grid["is_feasible"]].copy()

    if len(feasible) == 0:
        # Fallback: choose the least-bad pair if nothing satisfies both constraints
        grid["constraint_penalty"] = (
            np.maximum(0.0, grid["review_rate"] - review_rate_cap) +
            np.maximum(0.0, grid["approve_observed_default_rate"] - max_auto_bad_rate)
        )
        best_row = grid.sort_values(
            by=["constraint_penalty", "selection_score"],
            ascending=[True, False]
        ).iloc[0]
        status = "fallback_no_fully_feasible_pair"
    else:
        best_row = feasible.sort_values(by="selection_score", ascending=False).iloc[0]
        status = "feasible_pair_selected"

    return {
        "status": status,
        "t_low": float(best_row["t_low"]),
        "t_high": float(best_row["t_high"]),
        "review_rate_cap": review_rate_cap,
        "max_auto_bad_rate": max_auto_bad_rate,
        "selection_score": float(best_row["selection_score"]),
    }