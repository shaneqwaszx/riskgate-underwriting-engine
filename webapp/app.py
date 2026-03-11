from textwrap import dedent

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

REQUIRED_UPLOAD_COLUMNS = [
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
    "term",
    "emp_length",
    "grade",
    "sub_grade",
    "home_ownership",
    "verification_status",
    "purpose",
    "application_type",
    "initial_list_status",
    "issue_d",
    "earliest_cr_line",
    "zip_code",
]

OPTIONAL_LOCATION_COLUMNS = ["address", "state", "addr_state"]


def validate_upload_columns(df: pd.DataFrame):
    missing_required = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    has_location = any(c in df.columns for c in OPTIONAL_LOCATION_COLUMNS)
    return missing_required, has_location

# --------------------------------------------------
# App config
# --------------------------------------------------
st.set_page_config(
    page_title="RiskGate Underwriting Engine",
    page_icon="📊",
    layout="wide"
)

st.markdown("""
<style>
.reco-card {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border: 1px solid rgba(148, 163, 184, 0.25);
    border-radius: 16px;
    padding: 18px 20px;
    margin-top: 8px;
    margin-bottom: 18px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}
.reco-title {
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #93c5fd;
    margin-bottom: 10px;
}
.reco-main {
    font-size: 1.05rem;
    font-weight: 600;
    color: #f8fafc;
    margin-bottom: 12px;
}
.reco-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(140px, 1fr));
    gap: 10px;
    margin-top: 10px;
}
.reco-metric {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 10px 12px;
}
.reco-label {
    font-size: 0.78rem;
    color: #cbd5e1;
    margin-bottom: 4px;
}
.reco-value {
    font-size: 1rem;
    font-weight: 700;
    color: #ffffff;
}
.helper-box {
    background: rgba(255,255,255,0.03);
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 10px;
    margin-bottom: 12px;
}


</style>
""", unsafe_allow_html=True)

if "show_assistant_drawer" not in st.session_state:
    st.session_state.show_assistant_drawer = False

if "use_frozen_thresholds" not in st.session_state:
    st.session_state.use_frozen_thresholds = True

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

@st.cache_data
def load_upload_template(path: str):
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

from textwrap import dedent

def render_recommendation_card(goal: str, recommendation_df: pd.DataFrame):
    if recommendation_df.empty:
        st.warning(
            "No feasible threshold pair met the requested constraints. "
            "Try loosening the review-rate cap or reducing the minimum approve-rate target."
        )
        return

    top = recommendation_df.iloc[0]

    html = dedent(f"""
    <div class="reco-card">
        <div class="reco-title">Policy assistant recommendation</div>
        <div class="reco-main">
            Recommended setting for <b>{goal}</b>:
            <span style="color:#93c5fd;">t_low = {top['t_low']:.2f}</span>
            and
            <span style="color:#93c5fd;">t_high = {top['t_high']:.2f}</span>
        </div>

        <div class="reco-grid">
            <div class="reco-metric">
                <div class="reco-label">Approve rate</div>
                <div class="reco-value">{top['approve_rate']:.1%}</div>
            </div>
            <div class="reco-metric">
                <div class="reco-label">Review rate</div>
                <div class="reco-value">{top['review_rate']:.1%}</div>
            </div>
            <div class="reco-metric">
                <div class="reco-label">Reject rate</div>
                <div class="reco-value">{top['reject_rate']:.1%}</div>
            </div>
            <div class="reco-metric">
                <div class="reco-label">Non-rejected risk proxy</div>
                <div class="reco-value">{top['risk_proxy_nonreject']:,.0f}</div>
            </div>
        </div>
    </div>
    """).strip()

    st.markdown(html, unsafe_allow_html=True)


def open_assistant():
    st.session_state.show_assistant_drawer = True


def close_assistant():
    st.session_state.show_assistant_drawer = False


def apply_top_recommendation(recommendation_df: pd.DataFrame):
    if recommendation_df.empty:
        return

    top = recommendation_df.iloc[0]
    st.session_state.use_frozen_thresholds = False
    st.session_state.t_low_slider = float(top["t_low"])
    st.session_state.t_high_slider = float(top["t_high"])
    st.session_state.show_assistant_drawer = False


def render_policy_assistant_drawer(goal: str, recommendation_df: pd.DataFrame, scenario_grid: pd.DataFrame, ai_text: str | None = None):
    if not st.session_state.show_assistant_drawer:
        return

    with st.container(key="assistant_drawer"):
        top = recommendation_df.iloc[0] if not recommendation_df.empty else None

        head_col1, head_col2 = st.columns([0.84, 0.16])
        with head_col1:
            st.markdown("### 💡 Policy Assistant")
        with head_col2:
            st.button("✕", key="assistant_close_btn", on_click=close_assistant)

        if top is not None:
            st.markdown(
                f"""
                <div class="assistant-section">
                    <h4>Top recommendation</h4>
                    <div class="assistant-muted">
                        For <b>{goal}</b>, the strongest current setting is
                        <b>t_low = {top['t_low']:.2f}</b> and
                        <b>t_high = {top['t_high']:.2f}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            render_recommendation_card(goal, recommendation_df)

            if st.button("Apply this recommendation to threshold sliders", key="apply_top_reco_btn", use_container_width=True):
                apply_top_recommendation(recommendation_df)
                st.rerun()

        with st.expander("Explain the business goals", expanded=True):
            st.markdown("""
            **Balanced**  
            Tries to maintain a healthy approval rate while penalizing excessive retained risk and avoiding review overload.

            **Growth**  
            Favours higher approvals and conversion, while still applying a lighter penalty to retained risk.

            **Conservative**  
            Prioritizes lower retained risk in the non-rejected pool, even if approval volume falls.

            **Operations-first**  
            Prioritizes a smaller review queue so the underwriting team can manage workload more easily.
            """)

        with st.expander("How the assistant derives recommendations", expanded=False):
            st.markdown("""
            The assistant evaluates many threshold pairs (`t_low`, `t_high`) and estimates their impact on:
            - approval rate
            - review rate
            - reject rate
            - average PD in each decision bucket
            - retained risk proxy in the non-rejected population

            It then ranks threshold pairs according to the selected business goal and the operational constraints set on the page.
            """)

        with st.expander("Top threshold scenarios", expanded=False):
            st.dataframe(recommendation_df, use_container_width=True, height=220)

        if ai_text:
            st.markdown(
                f'<div class="helper-box"><b>AI explanation:</b><br>{ai_text}</div>',
                unsafe_allow_html=True
            )

def render_policy_assistant_sidebar(goal: str, recommendation_df: pd.DataFrame, scenario_grid: pd.DataFrame, ai_text: str | None = None):
    with st.sidebar:
        st.markdown("## 💡 Policy Assistant")

        if not recommendation_df.empty:
            render_recommendation_card(goal, recommendation_df)

            if st.button("Apply recommended thresholds", use_container_width=True):
                apply_top_recommendation(recommendation_df)
                st.rerun()
        else:
            st.warning("No feasible threshold recommendation is available for the current constraints.")

        with st.expander("Explain the business goals", expanded=True):
            st.markdown("""
            **Balanced**  
            Tries to maintain a healthy approval rate while penalizing excessive retained risk and avoiding review overload.

            **Growth**  
            Favours higher approvals and conversion, while still applying a lighter penalty to retained risk.

            **Conservative**  
            Prioritizes lower retained risk in the non-rejected pool, even if approval volume falls.

            **Operations-first**  
            Prioritizes a smaller review queue so the underwriting team can manage workload more easily.
            """)

        with st.expander("How the assistant derives recommendations", expanded=False):
            st.markdown("""
            The assistant evaluates many threshold pairs (`t_low`, `t_high`) and estimates their impact on:
            - approval rate
            - review rate
            - reject rate
            - average PD in each decision bucket
            - retained risk proxy in the non-rejected population

            It then ranks threshold pairs according to the selected business goal and the operational constraints set on the page.
            """)

        with st.expander("Top threshold scenarios", expanded=False):
            st.dataframe(recommendation_df, use_container_width=True, height=220)

        if ai_text:
            st.markdown(
                f'<div class="helper-box"><b>AI explanation:</b><br>{ai_text}</div>',
                unsafe_allow_html=True
            )
            
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

    use_frozen_thresholds = st.checkbox(
        "Use frozen engine thresholds",
        value=st.session_state.use_frozen_thresholds,
        key="use_frozen_thresholds"
    )

    default_t_low = float(frozen_thresholds["t_low"])
    default_t_high = float(frozen_thresholds["t_high"])

    if "t_low_slider" not in st.session_state:
        st.session_state.t_low_slider = default_t_low
    if "t_high_slider" not in st.session_state:
        st.session_state.t_high_slider = default_t_high

    t_low = st.slider(
        "Auto-approve threshold (t_low)",
        0.01, 0.40,
        st.session_state.t_low_slider,
        0.01,
        key="t_low_slider",
        disabled=use_frozen_thresholds
    )

    t_high = st.slider(
        "Auto-reject threshold (t_high)",
        0.05, 0.80,
        st.session_state.t_high_slider,
        0.01,
        key="t_high_slider",
        disabled=use_frozen_thresholds
    )

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
        ["Balanced", "Growth", "Conservative", "Operations-first"],
        help=(
            "Choose the business objective used to recommend threshold pairs. "
            "Different goals prioritize approval volume, retained risk, or operational workload differently."
        )
    )

    max_review_rate = st.slider(
        "Maximum review rate target",
        0.05, 0.50, 0.20, 0.01,
        help=(
            "Sets the highest acceptable proportion of applications routed to manual review. "
            "Lower values reduce analyst workload but may force more auto-approvals or rejections."
        )
    )

    min_approve_rate = st.slider(
        "Minimum approve rate target",
        0.20, 0.90, 0.50, 0.01,
        help=(
            "Sets the minimum desired proportion of applications that should be auto-approved. "
            "Higher values favor growth and faster throughput, but can increase retained risk."
        )
    )

    use_ollama = st.checkbox(
        "Use Ollama Cloud explanation",
        value=False,
        help="Optional advisory AI explanation using Ollama's hosted API."
    )

    ollama_model_name = st.text_input(
        "Ollama model name",
        value=st.secrets.get("OLLAMA_MODEL", "gpt-oss:120b"),
        help="Hosted Ollama model name used for threshold recommendation explanations."
    )

    with st.expander("How the policy assistant derives recommendations", expanded=False):
        st.markdown("""
        **Balanced**  
        Recommends threshold pairs that aim to maintain a healthy approval rate while penalizing excessive retained risk and keeping review volume close to the stated operational target.

        **Growth**  
        Prioritizes higher approval rates, while still applying a lighter penalty to retained risk. Suitable when portfolio expansion and customer conversion are more important.

        **Conservative**  
        Prioritizes lower retained risk in the non-rejected population. Suitable when credit quality and downside protection matter more than approval volume.

        **Operations-first**  
        Prioritizes lower review workload so the queue remains manageable for underwriters, while still considering approval rate and retained risk.

        **How the recommendation works**  
        The assistant generates many candidate threshold pairs \u2014 combinations of `t_low` and `t_high` \u2014 and estimates the resulting:
        - approve rate
        - review rate
        - reject rate
        - average PD by bucket
        - retained risk proxy in the non-rejected population

        It then ranks the candidate pairs according to the selected business goal and the operational constraints you set above.
        """)

# --------------------------------------------------
# Input dataset
# --------------------------------------------------
st.subheader("Input dataset")

template_path = PROJECT_ROOT / "data" / "scored" / "upload_template.csv"

with st.expander("Show required CSV format before upload", expanded=False):
    st.markdown("""
    **Recommended upload columns**
    - application_id
    - loan_amnt
    - installment
    - int_rate
    - annual_inc
    - dti
    - open_acc
    - pub_rec
    - revol_bal
    - revol_util
    - total_acc
    - mort_acc
    - pub_rec_bankruptcies
    - term
    - emp_length
    - grade
    - sub_grade
    - home_ownership
    - verification_status
    - purpose
    - application_type
    - initial_list_status
    - issue_d
    - earliest_cr_line
    - address
    - zip_code
    - state
    - addr_state

    **Date format used in this project**
    - `Jan-2019`
    - `Jun-2010`

    The uploaded file should contain one row per applicant.
    """)

    if template_path.exists():
        template_df = load_upload_template(str(template_path))
        st.dataframe(template_df, use_container_width=True)

        st.download_button(
            "Download upload template CSV",
            data=template_df.to_csv(index=False).encode("utf-8"),
            file_name="upload_template.csv",
            mime="text/csv"
        )
    else:
        st.warning("upload_template.csv was not found in data/scored/.")

uploaded_file = st.file_uploader("Upload a raw applications CSV", type=["csv"])
use_default_dataset = st.checkbox("Use default scoring dataset", value=True)

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_name = uploaded_file.name

    missing_required, has_location = validate_upload_columns(input_df)

    if missing_required:
        st.error(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_required)
        )
        st.stop()

    if not has_location:
        st.warning(
            "No address/zip/state fields were detected. "
            "Scoring may still run, but geographic-derived features will be weaker."
        )

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


ai_text = None

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
    ai_text = ollama_explanation(prompt, model_name=ollama_model_name)

render_policy_assistant_sidebar(goal, recommendation_df, scenario_grid, ai_text=ai_text)

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

        st.subheader("Deployment identity")
        st.json({
            "bundle_created_at": metadata.get("created_at_utc"),
            "engine_version": metadata.get("engine_version"),
            "input_rows": len(input_df),
            "thresholds_in_bundle": frozen_thresholds,
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