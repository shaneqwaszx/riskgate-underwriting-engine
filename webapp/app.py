import sys
from pathlib import Path

# --------------------------------------------------
# Make project root importable before unpickling
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import hashlib
import platform

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import sklearn
import streamlit as st
import xgboost

from webapp.assistant import (
    assign_decision,
    build_scenario_grid_from_scores,
    recommend_thresholds,
    diagnose_assistant_constraints,
    ollama_explanation,
)

APP_BUILD = "2026-03-11-clean-reset-v1"

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "final_model_bundle.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "scored" / "new_applications_to_score.csv"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "scored" / "upload_template.csv"

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


# --------------------------------------------------
# Page setup
# --------------------------------------------------
st.set_page_config(
    page_title="RiskGate Underwriting Engine",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
.small-muted {
    color: #94a3b8;
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# Session state
# --------------------------------------------------
if "threshold_mode_frozen" not in st.session_state:
    st.session_state.threshold_mode_frozen = True

if "active_t_low" not in st.session_state:
    st.session_state.active_t_low = None

if "active_t_high" not in st.session_state:
    st.session_state.active_t_high = None


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def file_signature(path: Path):
    stat = path.stat()
    return (str(path), stat.st_size, stat.st_mtime_ns)


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


@st.cache_resource
def load_bundle_cached(bundle_path: str, bundle_sig: tuple):
    return joblib.load(bundle_path)


@st.cache_data
def load_csv_cached(path: str, file_sig: tuple) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def compute_default_pd_pack(bundle_path: str, bundle_sig: tuple, input_path: str, input_sig: tuple):
    bundle = load_bundle_cached(bundle_path, bundle_sig)
    model = bundle["model"]
    df = load_csv_cached(input_path, input_sig)
    pd_scores = model.predict_proba(df)[:, 1]
    return {
        "pd_scores": pd_scores,
        "rows": len(df),
    }


def get_scenario_grid_once(default_df: pd.DataFrame, default_pd_scores: np.ndarray, bundle_sig: tuple, input_sig: tuple):
    cache_key = ("scenario_grid", bundle_sig, input_sig)

    if st.session_state.get("scenario_grid_key") != cache_key:
        default_loan_amnt = pd.to_numeric(
            default_df.get("loan_amnt", 0),
            errors="coerce"
        ).fillna(0.0).to_numpy()

        base_scored_df = pd.DataFrame({
            "pd_score": default_pd_scores,
            "loan_amnt": default_loan_amnt,
        })

        st.session_state.scenario_grid = build_scenario_grid_from_scores(base_scored_df)
        st.session_state.scenario_grid_key = cache_key

    return st.session_state.scenario_grid


def validate_upload_columns(df: pd.DataFrame):
    missing_required = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    has_location = any(c in df.columns for c in OPTIONAL_LOCATION_COLUMNS)
    return missing_required, has_location


def apply_decisions_from_pd(raw_df: pd.DataFrame, pd_scores: np.ndarray, t_low: float, t_high: float) -> pd.DataFrame:
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

    return {
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

def build_execution_explainer_prompt(summary: dict, run_info: dict) -> str:
    return f"""
    You are an underwriting analytics assistant.

    Interpret the following execution outcome for a probability-of-default underwriting engine.

    Execution thresholds:
    - t_low = {run_info['t_low_used']:.2f}
    - t_high = {run_info['t_high_used']:.2f}
    - input dataset = {run_info['input_name']}
    - rows scored = {run_info['rows_scored']}

    Execution result:
    - approve count = {summary['approve_n']}
    - review count = {summary['review_n']}
    - reject count = {summary['reject_n']}
    - approve rate = {summary['approve_rate']:.3f}
    - review rate = {summary['review_rate']:.3f}
    - reject rate = {summary['reject_rate']:.3f}
    - average PD overall = {summary['avg_pd_overall']:.4f}
    - average PD approve = {summary['avg_pd_approve']:.4f if pd.notna(summary['avg_pd_approve']) else 'NA'}
    - average PD review = {summary['avg_pd_review']:.4f if pd.notna(summary['avg_pd_review']) else 'NA'}
    - average PD reject = {summary['avg_pd_reject']:.4f if pd.notna(summary['avg_pd_reject']) else 'NA'}
    - non-rejected risk proxy = {summary['risk_proxy_nonreject']:.2f}

    Write a concise but useful explanation in plain English covering:
    1. what this execution means,
    2. the likely business benefits,
    3. the likely operational or risk trade-offs,
    4. one or two possible next adjustments to consider.

    Do not make legal or regulatory claims. Keep it advisory and practical.
    """.strip()

def render_recommendation_panel(goal: str, recommendation_df: pd.DataFrame):
    if recommendation_df.empty:
        st.warning(
            "No feasible threshold pair met the requested constraints. "
            "Try loosening the review-rate cap or reducing the minimum approve-rate target."
        )
        return

    top = recommendation_df.iloc[0]

    st.caption("POLICY ASSISTANT RECOMMENDATION")
    st.markdown(
        f"**Recommended setting for {goal}:**  \n"
        f"`t_low = {top['t_low']:.2f}` and `t_high = {top['t_high']:.2f}`"
    )

    col1, col2 = st.columns(2)
    col1.metric("Approve rate", f"{top['approve_rate']:.1%}")
    col2.metric("Review rate", f"{top['review_rate']:.1%}")

    col3, col4 = st.columns(2)
    col3.metric("Reject rate", f"{top['reject_rate']:.1%}")
    col4.metric("Risk proxy", f"{top['risk_proxy_nonreject']:,.0f}")


def apply_top_recommendation(recommendation_df: pd.DataFrame):
    if recommendation_df.empty:
        return

    top = recommendation_df.iloc[0]
    st.session_state.threshold_mode_frozen = False
    st.session_state.active_t_low = float(top["t_low"])
    st.session_state.active_t_high = float(top["t_high"])


def build_runtime_identity(bundle_path: Path, default_input_path: Path, metadata: dict) -> dict:
    return {
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "bundle_path": str(bundle_path),
        "bundle_sha256": file_sha256(bundle_path) if bundle_path.exists() else "missing",
        "bundle_created_at": metadata.get("created_at_utc"),
        "engine_version": metadata.get("engine_version"),
        "thresholds_in_bundle": metadata.get("policy_thresholds"),
        "default_input_path": str(default_input_path),
        "default_input_sha256": file_sha256(default_input_path) if default_input_path.exists() else "missing",
    }


def compute_default_parity_summary(default_pd_scores: np.ndarray, thresholds: dict) -> dict:
    decisions = assign_decision(default_pd_scores, thresholds["t_low"], thresholds["t_high"])
    counts = pd.Series(decisions).value_counts()

    return {
        "rows": int(len(default_pd_scores)),
        "pd_mean": float(default_pd_scores.mean()),
        "pd_std": float(default_pd_scores.std()),
        "approve": int(counts.get("approve", 0)),
        "review": int(counts.get("review", 0)),
        "reject": int(counts.get("reject", 0)),
        "first_20_pd_scores": [float(x) for x in default_pd_scores[:20]],
    }


# --------------------------------------------------
# Load bundle and default data
# --------------------------------------------------
if not DEFAULT_BUNDLE_PATH.exists():
    st.error("Model bundle not found at artifacts/final_model_bundle.joblib")
    st.stop()

if not DEFAULT_INPUT_PATH.exists():
    st.error("Default scoring dataset not found at data/scored/new_applications_to_score.csv")
    st.stop()

bundle_sig = file_signature(DEFAULT_BUNDLE_PATH)
input_sig = file_signature(DEFAULT_INPUT_PATH)

try:
    bundle = load_bundle_cached(str(DEFAULT_BUNDLE_PATH), bundle_sig)
except Exception as e:
    st.error(f"Failed to load model bundle: {e}")
    st.stop()

model = bundle["model"]
frozen_thresholds = bundle["thresholds"]
metadata = bundle["metadata"]

default_df = load_csv_cached(str(DEFAULT_INPUT_PATH), input_sig)

default_pd_pack = compute_default_pd_pack(
    str(DEFAULT_BUNDLE_PATH),
    bundle_sig,
    str(DEFAULT_INPUT_PATH),
    input_sig,
)

default_pd_scores = np.array(default_pd_pack["pd_scores"])

scenario_grid = get_scenario_grid_once(
    default_df=default_df,
    default_pd_scores=default_pd_scores,
    bundle_sig=bundle_sig,
    input_sig=input_sig,
)

runtime_identity = build_runtime_identity(DEFAULT_BUNDLE_PATH, DEFAULT_INPUT_PATH, metadata)
default_parity_summary = compute_default_parity_summary(default_pd_scores, frozen_thresholds)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("RiskGate: Automated Underwriting Policy Engine")
st.caption("Local scoring prototype for calibrated PD-based approve / review / reject decisioning")
st.caption(f"Build: {APP_BUILD}")

# --------------------------------------------------
# Sidebar assistant
# --------------------------------------------------
if "assistant_goal" not in st.session_state:
    st.session_state.assistant_goal = "Balanced"
if "assistant_max_review_rate" not in st.session_state:
    st.session_state.assistant_max_review_rate = 0.20
if "assistant_min_approve_rate" not in st.session_state:
    st.session_state.assistant_min_approve_rate = 0.50
if "assistant_use_ollama" not in st.session_state:
    st.session_state.assistant_use_ollama = False


with st.sidebar:
    st.markdown("## 💡 Policy Assistant")
    st.markdown(
        '<div class="small-muted">Recommendations are based on the default scoring dataset so that policy scenarios stay consistent and comparable.</div>',
        unsafe_allow_html=True
    )

    with st.form("assistant_form", clear_on_submit=False):
        goal_draft = st.selectbox(
            "Preferred business goal",
            ["Balanced", "Growth", "Conservative", "Operations-first"],
            index=["Balanced", "Growth", "Conservative", "Operations-first"].index(st.session_state.assistant_goal),
            help="Select the business objective used to rank threshold candidates."
        )

        max_review_rate_draft = st.slider(
            "Maximum review rate target",
            0.05, 0.50, float(st.session_state.assistant_max_review_rate), 0.01,
            help="Maximum acceptable proportion of applications routed to manual review."
        )

        min_approve_rate_draft = st.slider(
            "Minimum approve rate target",
            0.20, 0.90, float(st.session_state.assistant_min_approve_rate), 0.01,
            help="Minimum desired auto-approval proportion."
        )

        use_ollama_draft = st.checkbox(
            "Use Ollama Cloud explanation",
            value=st.session_state.assistant_use_ollama,
            help="Optional advisory AI explanation using Ollama's hosted API."
        )

        assistant_submit = st.form_submit_button("Update policy assistant")

    if assistant_submit:
        st.session_state.assistant_goal = goal_draft
        st.session_state.assistant_max_review_rate = max_review_rate_draft
        st.session_state.assistant_min_approve_rate = min_approve_rate_draft
        st.session_state.assistant_use_ollama = use_ollama_draft

    def _get_secret(name: str, default: str = "") -> str:
        try:
            return st.secrets.get(name, default)
        except Exception:
            return default

    ollama_model_name = st.text_input(
        "Ollama model name",
        value=_get_secret("OLLAMA_MODEL", "gpt-oss:120b"),
        help="Hosted Ollama model name."
    )

goal = st.session_state.assistant_goal
max_review_rate = st.session_state.assistant_max_review_rate
min_approve_rate = st.session_state.assistant_min_approve_rate
use_ollama = st.session_state.assistant_use_ollama


recommendation_df = recommend_thresholds(
    grid=scenario_grid,
    goal=goal,
    max_review_rate=max_review_rate,
    min_approve_rate=min_approve_rate,
)

assistant_diag = diagnose_assistant_constraints(
    grid=scenario_grid,
    recommendation_df=recommendation_df,
    max_review_rate=max_review_rate,
    min_approve_rate=min_approve_rate,
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

with st.sidebar:
    with st.container(border=True):
        st.caption("ASSISTANT DIAGNOSTICS")
        st.write(assistant_diag["message"])

        if assistant_diag["review_target_binding"] is not None:
            c1, c2 = st.columns(2)
            c1.metric("Review target gap", f"{assistant_diag['review_gap']:.1%}")
            c2.metric("Approve target gap", f"{assistant_diag['approve_gap']:.1%}")

    if st.button("Apply recommended thresholds", use_container_width=True):
        apply_top_recommendation(recommendation_df)
        st.rerun()

    with st.expander("Explain the business goals", expanded=False):
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

    with st.expander("How the policy assistant derives recommendations", expanded=False):
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
        st.info(ai_text)

# --------------------------------------------------
# Threshold controls
# --------------------------------------------------
left_col, right_col = st.columns([1.15, 0.85])

with left_col:
    st.subheader("Threshold controls")

    default_t_low = float(frozen_thresholds["t_low"])
    default_t_high = float(frozen_thresholds["t_high"])

    if st.session_state.active_t_low is None:
        st.session_state.active_t_low = default_t_low
    if st.session_state.active_t_high is None:
        st.session_state.active_t_high = default_t_high

    use_frozen_thresholds = st.checkbox(
        "Use frozen engine thresholds",
        value=st.session_state.threshold_mode_frozen,
        key="threshold_mode_checkbox"
    )

    st.session_state.threshold_mode_frozen = use_frozen_thresholds

    with st.form("threshold_form", clear_on_submit=False):
        draft_t_low = st.slider(
            "Auto-approve threshold (t_low)",
            0.01, 0.40,
            float(default_t_low if use_frozen_thresholds else st.session_state.active_t_low),
            0.01,
            disabled=use_frozen_thresholds
        )

        draft_t_high = st.slider(
            "Auto-reject threshold (t_high)",
            0.05, 0.80,
            float(default_t_high if use_frozen_thresholds else st.session_state.active_t_high),
            0.01,
            disabled=use_frozen_thresholds
        )

        threshold_submit = st.form_submit_button("Apply thresholds")

    if threshold_submit:
        if use_frozen_thresholds:
            st.session_state.active_t_low = default_t_low
            st.session_state.active_t_high = default_t_high
        else:
            st.session_state.active_t_low = draft_t_low
            st.session_state.active_t_high = draft_t_high

    if use_frozen_thresholds:
        t_low = default_t_low
        t_high = default_t_high
    else:
        t_low = st.session_state.active_t_low
        t_high = st.session_state.active_t_high

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
    st.subheader("Current engine state")
    st.metric("Frozen t_low", f"{default_t_low:.2f}")
    st.metric("Frozen t_high", f"{default_t_high:.2f}")
    st.markdown(
        '<div class="small-muted">The sidebar assistant suggests policy scenarios. '
        'The threshold form controls the actual execution thresholds for this run.</div>',
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Input dataset
# --------------------------------------------------
st.subheader("Input dataset")

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

    if TEMPLATE_PATH.exists():
        template_sha = file_sha256(TEMPLATE_PATH)
        template_df = load_csv_cached(str(TEMPLATE_PATH), template_sha)
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
            "No address/state fields were detected. "
            "Scoring may still run, but geographic-derived features may be weaker."
        )

elif use_default_dataset:
    input_df = default_df
    input_name = str(DEFAULT_INPUT_PATH)
else:
    st.info("Upload a file or enable the default dataset.")
    st.stop()

# --------------------------------------------------
# Execute engine
# --------------------------------------------------
run_button = st.button("Execute underwriting engine", type="primary")

if run_button:
    if uploaded_file is None and use_default_dataset:
        scored_df = apply_decisions_from_pd(
            raw_df=default_df,
            pd_scores=default_pd_scores,
            t_low=t_low,
            t_high=t_high,
        )
    else:
        uploaded_pd_scores = model.predict_proba(input_df)[:, 1]
        scored_df = apply_decisions_from_pd(
            raw_df=input_df,
            pd_scores=uploaded_pd_scores,
            t_low=t_low,
            t_high=t_high,
        )

    st.session_state.scored_df = scored_df
    st.session_state.last_run_info = {
        "input_name": input_name,
        "rows_scored": len(input_df),
        "t_low_used": t_low,
        "t_high_used": t_high,
        "use_frozen_thresholds": st.session_state.threshold_mode_frozen,
    }

if "scored_df" in st.session_state:
    scored_df = st.session_state.scored_df
    summary = build_summary(scored_df)

    execution_ai_text = None
    if use_ollama:
        execution_prompt = build_execution_explainer_prompt(summary, st.session_state.last_run_info)
        execution_ai_text = ollama_explanation(execution_prompt, model_name=ollama_model_name)

    st.success("Scoring completed successfully.")
    st.json(st.session_state.last_run_info)

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

    if execution_ai_text:
        with st.expander("AI explanation of this execution", expanded=False):
            st.write(execution_ai_text)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Applicant report",
        "Scenario comparison",
        "Audit / metadata",
        "Downloads",
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

        with st.expander("Debug / parity diagnostics", expanded=False):
            st.subheader("Runtime identity")
            st.json(runtime_identity)

            st.subheader("Default parity summary")
            st.json(default_parity_summary)

            st.subheader("App build")
            st.code(APP_BUILD)

        st.subheader("Thresholds used for this run")
        st.json({
            "input_dataset": st.session_state.last_run_info["input_name"],
            "t_low_used": st.session_state.last_run_info["t_low_used"],
            "t_high_used": st.session_state.last_run_info["t_high_used"],
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