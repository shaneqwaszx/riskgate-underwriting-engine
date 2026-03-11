import re
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from riskgate import config as cfg


def filter_to_binary_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only Fully Paid and Charged Off rows and add binary target column.
    """
    out = df.copy()
    mask = out[cfg.TARGET_COL].isin(cfg.VALID_TARGET_LABELS)
    out = out.loc[mask].copy()
    out[cfg.TARGET_BINARY_COL] = out[cfg.TARGET_COL].map(cfg.LABEL_MAP).astype(int)
    return out


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    # Handles strings like "Jan-2015"
    return pd.to_datetime(series, format="%b-%Y", errors="coerce")


def _parse_term_months(value) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def _parse_emp_length_years(value) -> float:
    if pd.isna(value):
        return np.nan

    text = str(value).strip().lower()

    if text in {"n/a", "nan", "none", ""}:
        return np.nan
    if "< 1 year" in text or "<1 year" in text:
        return 0.5
    if "10+ years" in text or "10 + years" in text:
        return 10.0

    match = re.search(r"(\d+)", text)
    return float(match.group(1)) if match else np.nan


def _extract_zip_code(address_series: pd.Series, zip_series: Optional[pd.Series] = None) -> pd.Series:
    if zip_series is not None:
        z = zip_series.astype("string")
        z = z.str.extract(r"(\d{5})", expand=False)
    else:
        z = pd.Series(index=address_series.index, dtype="string")

    fallback = address_series.astype("string").str.extract(r"(\d{5})\s*$", expand=False)
    out = z.fillna(fallback)
    return out.astype("string")


def _extract_state(address_series: pd.Series, state_series: Optional[pd.Series] = None,
                   addr_state_series: Optional[pd.Series] = None) -> pd.Series:
    candidates = []

    if state_series is not None:
        candidates.append(state_series.astype("string").str.upper().str.strip())

    if addr_state_series is not None:
        candidates.append(addr_state_series.astype("string").str.upper().str.strip())

    address_state = address_series.astype("string").str.extract(r"([A-Z]{2})\s+\d{5}\s*$", expand=False)
    candidates.append(address_state.astype("string").str.upper().str.strip())

    out = candidates[0]
    for s in candidates[1:]:
        out = out.fillna(s)

    return out.astype("string")


def _month_diff(later_dates: pd.Series, earlier_dates: pd.Series) -> pd.Series:
    years = later_dates.dt.year - earlier_dates.dt.year
    months = later_dates.dt.month - earlier_dates.dt.month
    result = years * 12 + months
    return result.where(result >= 0, np.nan)


class LoanFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Stateless feature builder that converts raw LendingClub-like inputs into
    stable engineered features used by the underwriting engine.
    """

    def __init__(self, reference_date: Optional[str] = None):
        self.reference_date = reference_date

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Ensure all expected raw columns exist
        for col in cfg.RAW_COLUMNS_NEEDED:
            if col not in df.columns:
                df[col] = pd.NA

        out = pd.DataFrame(index=df.index)

        # Numeric passthrough
        for col in [
            "loan_amnt", "installment", "int_rate", "annual_inc", "dti", "open_acc",
            "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc",
            "pub_rec_bankruptcies"
        ]:
            out[col] = _safe_to_numeric(df[col])

        # Derived numerics
        out["term_months"] = df["term"].apply(_parse_term_months)
        out["emp_length_years"] = df["emp_length"].apply(_parse_emp_length_years)

        issue_dt = _safe_to_datetime(df["issue_d"])
        earliest_dt = _safe_to_datetime(df["earliest_cr_line"])

        if self.reference_date is not None:
            fallback_issue = pd.Timestamp(self.reference_date)
        else:
            fallback_issue = pd.Timestamp.utcnow().tz_localize(None).normalize()

        issue_dt = issue_dt.fillna(fallback_issue)
        out["credit_history_months"] = _month_diff(issue_dt, earliest_dt)

        # Geographic derivation
        address_series = df["address"].astype("string")
        zip_series = df["zip_code"] if "zip_code" in df.columns else None
        state_series = df["state"] if "state" in df.columns else None
        addr_state_series = df["addr_state"] if "addr_state" in df.columns else None

        full_zip = _extract_zip_code(address_series, zip_series)
        out["zip3"] = full_zip.str.slice(0, 3)
        out["state"] = _extract_state(address_series, state_series, addr_state_series)

        # Stable categoricals
        for col in [
            "grade",
            "sub_grade",
            "home_ownership",
            "verification_status",
            "purpose",
            "application_type",
            "initial_list_status",
        ]:
            out[col] = df[col].astype("string").str.strip()

        # Final ordered schema
        final_cols = cfg.NUMERIC_FEATURES + cfg.CATEGORICAL_FEATURES
        for col in final_cols:
            if col not in out.columns:
                out[col] = pd.NA

        return out[final_cols]