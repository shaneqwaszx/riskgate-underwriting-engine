import sys
from pathlib import Path

# --------------------------------------------------
# Make project root importable before joblib unpickles
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from webapp.assistant import (
    assign_decision,
    build_scenario_grid_from_scores,
    recommend_thresholds,
    rule_based_advice,
    ollama_explanation,
)

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="RiskGate Underwriting Engine",
    page_icon="📊",
    layout="wide"
)

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "final_model_bundle.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "scored" / "new_applications_to_score.csv"


# --------------------------------------------------
# Helpers
# --------------------------------------------------
@st.cache_resource
def load_bundle(bundle_path: str):
    return joblib.load(bundle_path)


@st.cache_data
def load_csv_from_path(path: str):
    return pd.read_csv(path)


def score_dataframe(model, raw_df: pd.DataFrame, t_low: float, t_high: float) -> pd.DataFrame:
    pd_scores = model.predict_proba(raw_df)[:, 1]
    decisions = assign_decision(pd_scores, t_low, t_high)

    loan_amnt = pd.to_numeric(
        raw_df.get("loan_amnt", pd.Series(0, index=raw_df.index)),
        errors="coerce"
    ).fillna(0.0)
    risk_proxy = pd_scores * loan_amnt

    if "application_id" in raw_df.columns:
        record_id = raw_df["application_id"]
    elif "id" in raw_df.columns:
        record_id = raw_df["id"]
    else:
        record_id = pd.Series(raw_df.index, name="record_id")

    out = raw_df.copy()
    out["record_id"] = record_id
    out["pd_score"] = pd_scores
    out["decision"] = decisions
    out["t_low"] = t_low
    out["t_high"] = t_high
    out["risk_proxy_pd_x_loan_amnt"] = risk_proxy

    out["decision_reason_code"] = np.where(
        pd_scores < t_low,
        "pd_below_t_low",
        np.where(pd_scores < t_high, "pd_in_review_band", "pd_above_t_high")
    )

    return out


def build_summary(scored_df: pd.DataFrame) -> dict:
    total = len(scored_df)
    counts = scored_df["decision"].value_counts().to_dict()

    approve_n = counts.get("approve", 0)
    review_n = counts.get("review", 0)
    reject_n = counts.get("reject", 0)

    summary = {
        "total_applicants": total,
        "approve_n": approve_n,
        "review_n": review_n,
        "reject_n": reject_n,
        "approve_rate": approve_n / total if total else 0,
        "review_rate": review_n / total if total else 0,
        "reject_rate": reject_n / total if total else 0,
        "avg_pd_overall": float(scored_df["pd_score"].mean()),
        "avg_pd_approve": float(scored_df.loc[scored_df["decision"] == "approve", "pd_score"].mean()) if approve_n else np.nan,
        "avg_pd_review": float(scored_df.loc[scored_df["decision"] == "review", "pd_score"].mean()) if review_n else np.nan,
        "avg_pd_reject": float(scored_df.loc[scored_df["decision"] == "reject", "pd_score"].mean()) if reject_n else np.nan,
        "risk_proxy_total": float(scored_df["risk_proxy_pd_x_loan_amnt"].sum()),
        "risk_proxy_nonreject": float(
            scored_df.loc[
                scored_df["decision"].isin(["approve", "review"]),
                "risk_proxy_pd_x_loan_amnt"
            ].sum()
        ),
    }
    return summary


# --------------------------------------------------
# Load engine bundle
# --------------------------------------------------
if not DEFAULT_BUNDLE_PATH.exists():
    st.error("Model bundle not found at artifacts/final_model_bundle.joblib")
    st.stop()

bundle = load_bundle(str(DEFAULT_BUNDLE_PATH))
model = bundle["model"]
frozen_thresholds = bundle["thresholds"]
metadata = bundle["metadata"]

st.title("RiskGate: Automated Underwriting Policy Engine")
st.caption("Local scoring prototype for calibrated PD-based approve / review / reject decisioning")

# --------------------------------------------------
# Controls
# --------------------------------------------------
left_col, right_col = st.columns([1.1, 1.0])

with left_col:
    st.subheader("Threshold controls")

    use_frozen_thresholds = st.checkbox("Use frozen engine thresholds", value=True)

    default_t_low = float(frozen_thresholds["t_low"])
    default_t_high = float(frozen_thresholds["t_high"])

    t_low = st.slider("Auto-approve threshold (t_low)", 0.01, 0.40, default_t_low, 0.01)
    t_high = st.slider("Auto-reject threshold (t_high)", 0.05, 0.80, default_t_high, 0.01)

    if use_frozen_thresholds:
        t_low = default_t_low
        t_high = default_t_high

    if t_high <= t_low:
        st.error("t_high must be greater than t_low.")
        st.stop()

    st.markdown(
        f"""
        **Current policy**
        - Auto-approve if PD < **{t_low:.2f}**
        - Review if **{t_low:.2f} ≤ PD < {t_high:.2f}**
        - Reject if PD ≥ **{t_high:.2f}**
        """
    )

with right_col:
    st.subheader("Policy assistant")

    goal = st.selectbox(
        "Preferred business goal",
        ["Balanced", "Growth", "Conservative", "Operations-first"]
    )

    max_review_rate = st.slider("Maximum review rate target", 0.05, 0.50, 0.20, 0.01)
    min_approve_rate = st.slider("Minimum approve rate target", 0.20, 0.90, 0.50, 0.01)

    use_ollama = st.checkbox("Use local Ollama explanation", value=False)
    ollama_model_name = st.text_input("Ollama model name", value="llama3.1:8b")

# --------------------------------------------------
# Input dataset
# --------------------------------------------------
st.subheader("Input dataset")

uploaded_file = st.file_uploader("Upload a raw applications CSV", type=["csv"])
use_default_dataset = st.checkbox("Use default scoring dataset", value=True)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_name = uploaded_file.name
elif use_default_dataset:
    if not DEFAULT_INPUT_PATH.exists():
        st.error("Default scoring dataset not found.")
        st.stop()
    input_df = load_csv_from_path(str(DEFAULT_INPUT_PATH))
    input_name = str(DEFAULT_INPUT_PATH)
else:
    st.info("Upload a file or enable the default dataset.")
    st.stop()

# --------------------------------------------------
# Assistant scenario recommendation
# --------------------------------------------------
base_scored = score_dataframe(model, input_df, t_low=default_t_low, t_high=default_t_high)
scenario_grid = build_scenario_grid_from_scores(base_scored)
recommendation_df = recommend_thresholds(
    grid=scenario_grid,
    goal=goal,
    max_review_rate=max_review_rate,
    min_approve_rate=min_approve_rate
)

if use_ollama:
    prompt = f"""
You are helping an underwriting analyst choose thresholds for a triage policy.
Business goal: {goal}
Max review rate target: {max_review_rate:.2f}
Min approve rate target: {min_approve_rate:.2f}

Top candidate threshold pairs:
{recommendation_df.to_string(index=False)}

Provide a short practical recommendation in plain English.
"""
    st.info(ollama_explanation(prompt, model_name=ollama_model_name))
else:
    st.info(rule_based_advice(goal, recommendation_df))

# --------------------------------------------------
# Execute engine
# --------------------------------------------------
run_button = st.button("Execute underwriting engine", type="primary")

if run_button:
    scored_df = score_dataframe(model, input_df, t_low=t_low, t_high=t_high)
    summary = build_summary(scored_df)

    st.success("Scoring completed successfully.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Applicants", f"{summary['total_applicants']:,}")
    k2.metric("Approve", f"{summary['approve_n']:,}", f"{summary['approve_rate']:.1%}")
    k3.metric("Review", f"{summary['review_n']:,}", f"{summary['review_rate']:.1%}")
    k4.metric("Reject", f"{summary['reject_n']:,}", f"{summary['reject_rate']:.1%}")

    k5, k6, k7 = st.columns(3)
    k5.metric("Avg PD (overall)", f"{summary['avg_pd_overall']:.3f}")
    k6.metric("Avg PD (approve)", f"{summary['avg_pd_approve']:.3f}" if pd.notna(summary["avg_pd_approve"]) else "NA")
    k7.metric("Risk proxy (non-rejected)", f"{summary['risk_proxy_nonreject']:,.0f}")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        decision_counts = scored_df["decision"].value_counts().reset_index()
        decision_counts.columns = ["decision", "count"]
        fig_bar = px.bar(decision_counts, x="decision", y="count", title="Decision distribution")
        st.plotly_chart(fig_bar, use_container_width=True)

    with chart_col2:
        fig_hist = px.histogram(
            scored_df,
            x="pd_score",
            color="decision",
            nbins=50,
            title="PD score distribution by decision"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Applicant report",
        "Scenario comparison",
        "Audit / metadata",
        "Downloads"
    ])

    with tab1:
        st.subheader("Applicant decision report")

        filter_decision = st.multiselect(
            "Filter decisions",
            options=["approve", "review", "reject"],
            default=["approve", "review", "reject"]
        )

        display_df = scored_df[scored_df["decision"].isin(filter_decision)].copy()

        preferred_cols = [
            "record_id", "loan_amnt", "int_rate", "annual_inc", "dti",
            "grade", "sub_grade", "purpose", "pd_score", "decision",
            "decision_reason_code", "risk_proxy_pd_x_loan_amnt"
        ]
        available_cols = [c for c in preferred_cols if c in display_df.columns]

        st.dataframe(display_df[available_cols], use_container_width=True, height=500)

    with tab2:
        st.subheader("Top recommended scenarios")
        st.dataframe(recommendation_df, use_container_width=True, height=250)

        st.subheader("Full scenario grid")
        st.dataframe(scenario_grid, use_container_width=True, height=350)

    with tab3:
        st.subheader("Engine metadata")
        st.json(metadata)

        st.subheader("Thresholds used for this run")
        st.json({
            "input_dataset": input_name,
            "t_low_used": t_low,
            "t_high_used": t_high
        })

    with tab4:
        st.subheader("Download outputs")

        csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download scored decisions CSV",
            data=csv_bytes,
            file_name="scored_decisions_from_webapp.csv",
            mime="text/csv"
        )

        summary_json = json.dumps(summary, indent=2)
        st.download_button(
            "Download summary JSON",
            data=summary_json,
            file_name="scoring_summary.json",
            mime="application/json"
        )