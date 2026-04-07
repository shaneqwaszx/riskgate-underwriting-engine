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
    CUSTOM_PROFILE_NAME,
    PROFILE_SETTINGS,
    get_assistant_profile_settings,
)

import io

from riskgate.explainability import (
    make_xgb_contrib_pack,
    global_contrib_table,
    local_contrib_table,
)

from riskgate.fairness import (
    get_candidate_group_cols,
    build_group_outcome_table,
    build_feat_assessment_table,
    build_all_group_outcome_tables,
)

from riskgate.drift import build_drift_report
from riskgate.history import append_run_history, run_history_df
from riskgate.reviewer import append_override_log, load_override_log

APP_BUILD = "2026-03-11-clean-reset-v1"

DEFAULT_BUNDLE_PATH = PROJECT_ROOT / "artifacts" / "final_model_bundle.joblib"
DEFAULT_INPUT_PATH = PROJECT_ROOT / "data" / "scored" / "new_applications_to_score.csv"
TEMPLATE_PATH = PROJECT_ROOT / "data" / "scored" / "upload_template.csv"
OVERRIDE_LOG_PATH = PROJECT_ROOT / "data" / "scored" / "reviewer_overrides.csv"

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
TEXT_UPLOAD_COLUMNS = [
    "application_id",
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
    "address",
    "zip_code",
    "state",
    "addr_state",
]

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

if "run_history" not in st.session_state:
    st.session_state.run_history = []

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


@st.cache_data(show_spinner=False)
def load_uploaded_csv_cached(file_bytes: bytes) -> pd.DataFrame:
    dtype_map = {col: "string" for col in TEXT_UPLOAD_COLUMNS}
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=dtype_map)
    return sanitize_uploaded_input(df)


@st.cache_data(show_spinner=False)
def score_uploaded_csv_cached(bundle_path: str, bundle_sig: tuple, file_bytes: bytes):
    bundle = load_bundle_cached(bundle_path, bundle_sig)
    model = bundle["model"]
    df = load_uploaded_csv_cached(file_bytes)
    pd_scores = model.predict_proba(df)[:, 1]

    return {
        "df": df,
        "pd_scores": pd_scores,
        "file_hash": hashlib.sha256(file_bytes).hexdigest(),
    }

@st.cache_data(show_spinner=False)
def precompute_fairness_tables(raw_df: pd.DataFrame, scored_df: pd.DataFrame, min_group_size: int, max_unique: int):
    _, candidate_cols, table_map = build_all_group_outcome_tables(
        raw_df=raw_df,
        scored_df=scored_df,
        min_group_size=min_group_size,
        max_unique=max_unique,
    )
    return candidate_cols, table_map

def build_comparison_table(current_summary: dict, benchmark_summary: dict) -> pd.DataFrame:
    rows = [
        ("Approve rate", current_summary["approve_rate"], benchmark_summary["approve_rate"]),
        ("Review rate", current_summary["review_rate"], benchmark_summary["review_rate"]),
        ("Reject rate", current_summary["reject_rate"], benchmark_summary["reject_rate"]),
        ("Avg PD overall", current_summary["avg_pd_overall"], benchmark_summary["avg_pd_overall"]),
        ("Avg PD approve", current_summary["avg_pd_approve"], benchmark_summary["avg_pd_approve"]),
        ("Risk proxy non-rejected", current_summary["risk_proxy_nonreject"], benchmark_summary["risk_proxy_nonreject"]),
    ]

    out = pd.DataFrame(rows, columns=["metric", "current", "frozen_benchmark"])
    out["delta"] = out["current"] - out["frozen_benchmark"]
    return out

def get_scenario_grid_once(
    default_df: pd.DataFrame,
    default_pd_scores: np.ndarray,
    bundle_sig: tuple,
    input_sig: tuple,
    frozen_thresholds: dict,
):
    cache_key = ("scenario_grid", bundle_sig, input_sig, frozen_thresholds["t_low"], frozen_thresholds["t_high"])

    if st.session_state.get("scenario_grid_key") != cache_key:
        default_loan_amnt = pd.to_numeric(
            default_df.get("loan_amnt", 0),
            errors="coerce"
        ).fillna(0.0).to_numpy()

        base_scored_df = pd.DataFrame({
            "pd_score": default_pd_scores,
            "loan_amnt": default_loan_amnt,
        })

        t_low_candidates = np.sort(np.unique(np.round(
            np.append(np.arange(0.05, 0.21, 0.02), [float(frozen_thresholds["t_low"])]), 2
        )))
        t_high_candidates = np.sort(np.unique(np.round(
            np.append(np.arange(0.15, 0.41, 0.02), [float(frozen_thresholds["t_high"])]), 2
        )))

        st.session_state.scenario_grid = build_scenario_grid_from_scores(
            base_scored_df,
            t_low_values=t_low_candidates,
            t_high_values=t_high_candidates,
        )
        st.session_state.scenario_grid_key = cache_key

    return st.session_state.scenario_grid


def validate_upload_columns(df: pd.DataFrame):
    missing_required = [c for c in REQUIRED_UPLOAD_COLUMNS if c not in df.columns]
    has_location = any(c in df.columns for c in OPTIONAL_LOCATION_COLUMNS)
    return missing_required, has_location

def sanitize_uploaded_input(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Preserve text-like fields and remove pandas nullable-string / pd.NA issues
    for col in out.columns:
        dtype_name = str(out[col].dtype)

        if col in TEXT_UPLOAD_COLUMNS or pd.api.types.is_string_dtype(out[col]) or pd.api.types.is_object_dtype(out[col]):
            out[col] = out[col].astype("object")
            out.loc[pd.isna(out[col]), col] = np.nan

        elif dtype_name in ["Int64", "Int32", "Int16", "Int8", "UInt64", "UInt32", "UInt16", "UInt8", "Float64", "Float32"]:
            out[col] = pd.to_numeric(out[col], errors="coerce")

        elif dtype_name == "boolean":
            out[col] = out[col].astype("object")
            out.loc[pd.isna(out[col]), col] = np.nan

    return out

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
    def fmt_or_na(value, decimals=4):
        return f"{value:.{decimals}f}" if pd.notna(value) else "NA"

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
- average PD approve = {fmt_or_na(summary['avg_pd_approve'])}
- average PD review = {fmt_or_na(summary['avg_pd_review'])}
- average PD reject = {fmt_or_na(summary['avg_pd_reject'])}
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
    frozen_thresholds=frozen_thresholds,
)

runtime_identity = build_runtime_identity(DEFAULT_BUNDLE_PATH, DEFAULT_INPUT_PATH, metadata)
default_parity_summary = compute_default_parity_summary(default_pd_scores, frozen_thresholds)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("RiskGate: Automated Underwriting Policy Engine")
st.caption("Local scoring prototype for PD-based approve / review / reject decisioning")
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
        goal_options = [
            
            "Balanced",
            "Growth",
            "Conservative",
            "Operations-first",
            CUSTOM_PROFILE_NAME,
        ]

        goal_draft = st.selectbox(
            "Preferred business goal",
            goal_options,
            index=goal_options.index(st.session_state.assistant_goal),
            help="Fixed profiles use built-in operating assumptions. Only Custom target-driven uses the sliders below."
        )

        is_custom_profile = (goal_draft == CUSTOM_PROFILE_NAME)

        if not is_custom_profile:
            fixed_spec = PROFILE_SETTINGS[goal_draft]
            st.info(
                f"{goal_draft} uses fixed targets: "
                f"max review {fixed_spec['max_review_rate']:.0%}, "
                f"min approve {fixed_spec['min_approve_rate']:.0%}."
            )
        else:
            st.caption("This is the only profile that uses the target sliders below.")

        max_review_rate_draft = st.slider(
            "Maximum review rate target",
            0.05, 0.50, float(st.session_state.assistant_max_review_rate), 0.01,
            help="Used only when the profile is Custom target-driven.",
            disabled=not is_custom_profile,
        )

        min_approve_rate_draft = st.slider(
            "Minimum approve rate target",
            0.20, 0.90, float(st.session_state.assistant_min_approve_rate), 0.01,
            help="Used only when the profile is Custom target-driven.",
            disabled=not is_custom_profile,
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

selected_profile = st.session_state.assistant_goal
user_max_review_rate = st.session_state.assistant_max_review_rate
user_min_approve_rate = st.session_state.assistant_min_approve_rate
use_ollama = st.session_state.assistant_use_ollama

assistant_profile = get_assistant_profile_settings(
    profile_name=selected_profile,
    user_max_review_rate=user_max_review_rate,
    user_min_approve_rate=user_min_approve_rate,
)

effective_goal = assistant_profile["scoring_goal"]
effective_max_review_rate = assistant_profile["max_review_rate"]
effective_min_approve_rate = assistant_profile["min_approve_rate"]
uses_user_targets = assistant_profile["uses_user_targets"]

recommendation_df = recommend_thresholds(
    grid=scenario_grid,
    goal=effective_goal,
    max_review_rate=effective_max_review_rate,
    min_approve_rate=effective_min_approve_rate,
)

assistant_diag = diagnose_assistant_constraints(
    grid=scenario_grid,
    recommendation_df=recommendation_df,
    max_review_rate=effective_max_review_rate,
    min_approve_rate=effective_min_approve_rate,
)

ai_text = None
if use_ollama:
    prompt = f"""
    You are helping an underwriting analyst choose thresholds for a triage policy.

    Selected profile: {selected_profile}
    Profile mode: {"User-defined custom targets" if uses_user_targets else "Fixed profile assumptions"}
    Effective scoring goal: {effective_goal}
    Effective max review rate target: {effective_max_review_rate:.2f}
    Effective min approve rate target: {effective_min_approve_rate:.2f}

    Top candidate threshold pairs:
    {recommendation_df.to_string(index=False)}

    Provide a short practical recommendation in plain English.
    """
    ai_text = ollama_explanation(prompt, model_name=ollama_model_name)

with st.sidebar:
    with st.container(border=True):
        st.caption("CURRENT DEPLOYED BENCHMARK")
        st.write(
            f"t_low = {float(frozen_thresholds['t_low']):.2f}, "
            f"t_high = {float(frozen_thresholds['t_high']):.2f}"
        )
        st.caption(
            "This is the saved benchmark pair from the final model bundle. "
            "It is shown for reference only and is no longer a selectable assistant profile."
        )
    with st.container(border=True):
        st.caption("ASSISTANT DIAGNOSTICS")

        if uses_user_targets:
            st.caption("Mode: Custom target-driven — using your slider targets.")
        else:
            st.caption(
                f"Mode: {selected_profile} — fixed targets applied "
                f"(max review {effective_max_review_rate:.0%}, "
                f"min approve {effective_min_approve_rate:.0%})."
            )

        st.write(assistant_diag["message"])

        if "top_review_rate" in assistant_diag:
            c1, c2 = st.columns(2)

            c1.metric(
                "Review rate",
                f"{assistant_diag['top_review_rate']:.1%}",
                delta=f"{assistant_diag['review_vs_target']:+.1%} vs max"
            )

            c2.metric(
                "Approve rate",
                f"{assistant_diag['top_approve_rate']:.1%}",
                delta=f"{assistant_diag['approve_vs_target']:+.1%} vs minimum"
            )

            st.caption(
                f"Interpretation: {assistant_diag['review_text']}. {assistant_diag['approve_text']}."
            )

    if st.button("Apply recommended thresholds", use_container_width=True):
        apply_top_recommendation(recommendation_df)
        st.rerun()

    with st.expander("Explain the business goals", expanded=False):
        st.markdown(f"""
        **Balanced**  
        Uses fixed targets of **max review 20%** and **min approve 50%**. It tries to maintain a healthy approval rate while penalizing excessive retained risk and avoiding review overload.

        **Growth**  
        Uses fixed targets of **max review 25%** and **min approve 60%**. It favours higher approvals and conversion while still penalizing retained risk.

        **Conservative**  
        Uses fixed targets of **max review 15%** and **min approve 40%**. It prioritizes lower retained risk even if approval volume falls.

        **Operations-first**  
        Uses fixed targets of **max review 12%** and **min approve 45%**. It prioritizes a smaller review queue so the underwriting team can manage workload more easily.

        **{CUSTOM_PROFILE_NAME}**  
        This is the only profile that uses the sliders above. It lets the user set custom review and approval targets directly.
        """)

    with st.expander("How the policy assistant derives recommendations", expanded=False):
        st.markdown("""
        The assistant evaluates many threshold pairs (`t_low`, `t_high`) and estimates their impact on:
        - approval rate
        - review rate
        - reject rate
        - average PD in each decision bucket
        - retained risk proxy in the non-rejected population

        For **Balanced, Growth, Conservative, and Operations-first**, the assistant uses built-in profile targets.

        For **Custom target-driven**, the assistant uses the review and approval targets set by the user on this page.

        The deployed benchmark pair is shown separately for reference and is not treated as a recommendation profile.
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

input_mode = st.radio(
    "Choose input source",
    ["Default scoring dataset", "Upload CSV"],
    key="riskgate_input_mode",
)

uploaded_file = None
if input_mode == "Upload CSV":
    uploaded_file = st.file_uploader(
        "Upload a raw applications CSV",
        type=["csv"],
        key="riskgate_uploaded_csv",
    )

use_default_dataset = (input_mode == "Default scoring dataset")

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    upload_pack = score_uploaded_csv_cached(str(DEFAULT_BUNDLE_PATH), bundle_sig, file_bytes)

    input_df = upload_pack["df"]
    uploaded_pd_scores = np.array(upload_pack["pd_scores"])
    input_name = uploaded_file.name

    missing_required, has_location = validate_upload_columns(input_df)

    if missing_required:
        st.error("Uploaded CSV is missing required columns: " + ", ".join(missing_required))
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
        active_raw_df = default_df.copy()
        active_pd_scores = np.array(default_pd_scores)
    else:
        active_raw_df = sanitize_uploaded_input(input_df).copy()
        active_pd_scores = model.predict_proba(active_raw_df)[:, 1]

    scored_df = apply_decisions_from_pd(
        raw_df=active_raw_df,
        pd_scores=active_pd_scores,
        t_low=t_low,
        t_high=t_high,
    )

    run_summary = build_summary(scored_df)

    st.session_state.scored_df = scored_df
    st.session_state.last_raw_input_df = active_raw_df
    st.session_state.last_pd_scores = active_pd_scores
    st.session_state.last_run_info = {
        "input_name": input_name,
        "rows_scored": len(active_raw_df),
        "t_low_used": t_low,
        "t_high_used": t_high,
        "use_frozen_thresholds": st.session_state.threshold_mode_frozen,
    }

    append_run_history(
        st.session_state,
        st.session_state.last_run_info,
        run_summary,
    )

if "scored_df" in st.session_state:
    scored_df = st.session_state.scored_df
    summary = build_summary(scored_df)

    execution_ai_text = None
    if use_ollama:
        try:
            execution_prompt = build_execution_explainer_prompt(
                summary,
                st.session_state.last_run_info
            )
            execution_ai_text = ollama_explanation(
                execution_prompt,
                model_name=ollama_model_name
            )
        except Exception as e:
            execution_ai_text = f"Execution explanation unavailable: {e}"

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

        fig_hist.update_layout(
            barmode="overlay",
            xaxis_title="PD score",
            yaxis_title="Count",
            legend_title="Decision"
        )
        fig_hist.update_traces(opacity=0.72)
        fig_hist.update_xaxes(range=[0, 1])

        fig_hist.add_vline(
            x=t_low,
            line_width=2,
            line_dash="dash",
            line_color="green",
            annotation_text=f"low={t_low:.2f}",
            annotation_position="top left"
        )
        fig_hist.add_vline(
            x=t_high,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text=f"high={t_high:.2f}",
            annotation_position="top right"
        )
        fig_hist.add_vrect(
            x0=t_low,
            x1=t_high,
            fillcolor="gold",
            opacity=0.08,
            line_width=0,
            annotation_text="review band",
            annotation_position="top"
        )

        st.plotly_chart(fig_hist, use_container_width=True)

    if execution_ai_text:
        with st.expander("AI explanation of this execution", expanded=False):
            st.write(execution_ai_text)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "Applicant report",
        "Run comparison",
        "Explainability",
        "Fairness / FEAT",
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

        st.markdown("### Reviewer override logging")

        if not display_df.empty:
            review_pool = display_df.copy()
            review_pool["record_id"] = review_pool["record_id"].astype(str)

            visible_options = review_pool["record_id"].head(1000).tolist()

            selected_record_id = st.selectbox(
                "Select record_id to review",
                options=visible_options,
                help="For prototype simplicity, the selector shows the first 1,000 visible rows."
            )

            selected_row = review_pool.loc[
                review_pool["record_id"] == selected_record_id
            ].iloc[0]

            st.write({
                "record_id": selected_row["record_id"],
                "pd_score": float(selected_row["pd_score"]),
                "current_decision": selected_row["decision"],
            })

            with st.form("reviewer_override_form", clear_on_submit=True):
                reviewer_name = st.text_input("Reviewer name", value="analyst_demo")
                override_decision = st.selectbox(
                    "Override decision",
                    options=["approve", "review", "reject"],
                    index=["approve", "review", "reject"].index(selected_row["decision"])
                )
                reviewer_note = st.text_area("Reviewer note")

                override_submit = st.form_submit_button("Log override")

            if override_submit:
                append_override_log(
                    OVERRIDE_LOG_PATH,
                    {
                        "reviewer_name": reviewer_name.strip() or "analyst_demo",
                        "input_name": st.session_state.last_run_info["input_name"],
                        "record_id": selected_row["record_id"],
                        "pd_score": float(selected_row["pd_score"]),
                        "original_decision": selected_row["decision"],
                        "override_decision": override_decision,
                        "reviewer_note": reviewer_note.strip(),
                        "t_low_used": st.session_state.last_run_info["t_low_used"],
                        "t_high_used": st.session_state.last_run_info["t_high_used"],
                    },
                )
                st.success("Reviewer override logged.")
        else:
            st.info("No rows available under the current decision filter.")

    with tab2:
        st.subheader("Current run vs frozen benchmark")

        raw_last = st.session_state.last_raw_input_df
        pd_last = st.session_state.last_pd_scores

        benchmark_scored_df = apply_decisions_from_pd(
            raw_df=raw_last,
            pd_scores=pd_last,
            t_low=default_t_low,
            t_high=default_t_high,
        )
        benchmark_summary = build_summary(benchmark_scored_df)

        compare_df = build_comparison_table(summary, benchmark_summary)
        st.dataframe(compare_df, use_container_width=True)

        compare_chart_df = pd.DataFrame({
            "decision": ["approve", "review", "reject"] * 2,
            "count": [
                summary["approve_n"], summary["review_n"], summary["reject_n"],
                benchmark_summary["approve_n"], benchmark_summary["review_n"], benchmark_summary["reject_n"],
            ],
            "run": ["current"] * 3 + ["frozen_benchmark"] * 3,
        })

        fig_compare = px.bar(
            compare_chart_df,
            x="decision",
            y="count",
            color="run",
            barmode="group",
            title="Decision mix: current vs frozen benchmark"
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with tab3:
        st.subheader("Explainability")
        st.caption(
            "Purpose: inspect which features are driving the model's risk score globally and for an individual applicant. "
            "These are model explanations, not causal explanations, and they should be interpreted as decision-support diagnostics."
        )

        if metadata.get("chosen_model_name") != "xgb":
            st.info("Explainability is enabled for XGBoost-based final artifacts.")
        else:
            raw_last = st.session_state.last_raw_input_df.reset_index(drop=True)
            scored_last = scored_df.reset_index(drop=True)

            with st.form("explainability_form", clear_on_submit=False):
                explain_n = st.slider(
                    "Rows to include in explanation pack",
                    min_value=50,
                    max_value=min(2000, len(raw_last)),
                    value=min(300, len(raw_last)),
                    step=50,
                    help="Choose how many rows to analyze when building the explanation pack."
                )
                explain_submit = st.form_submit_button("Compute explanation pack")

            if explain_submit:
                with st.spinner("Computing XGBoost contribution values..."):
                    explain_raw = raw_last.head(explain_n).copy()
                    explain_scored = scored_last.head(explain_n).copy()

                    contrib_pack = make_xgb_contrib_pack(bundle["model"], explain_raw)

                    st.session_state.explain_raw = explain_raw
                    st.session_state.explain_scored = explain_scored
                    st.session_state.contrib_pack = contrib_pack
                    st.session_state.explain_n_used = explain_n

            if "contrib_pack" in st.session_state:
                st.caption(
                    f"Showing explanations for the last computed pack: first {st.session_state.get('explain_n_used', 'N/A')} rows."
                )

                contrib_pack = st.session_state.contrib_pack
                explain_scored = st.session_state.explain_scored

                global_df = global_contrib_table(contrib_pack).head(15).sort_values("mean_abs_contrib")
                fig_global = px.bar(
                    global_df,
                    x="mean_abs_contrib",
                    y="feature",
                    orientation="h",
                    title="Top global drivers of the model risk score"
                )
                st.plotly_chart(fig_global, use_container_width=True)

                selected_record = st.selectbox(
                    "Select applicant for local explanation",
                    explain_scored["record_id"].astype(str).tolist()
                )

                selected_idx = explain_scored.index[
                    explain_scored["record_id"].astype(str) == selected_record
                ][0]

                local_df = local_contrib_table(contrib_pack, row_idx=int(selected_idx), top_n=10)
                st.dataframe(local_df, use_container_width=True)

                if st.button("Clear explanation pack"):
                    for key in ["explain_raw", "explain_scored", "contrib_pack", "explain_n_used"]:
                        st.session_state.pop(key, None)
                    st.rerun()

    with tab4:
        st.subheader("Fairness / FEAT diagnostics")
        st.caption(
            "Purpose: review whether the current policy is producing materially different outcomes across key business segments. "
            "This is a governance diagnostic and proxy disparity check, not a final fairness certification."
        )

        raw_last = st.session_state.last_raw_input_df

        feat_table = build_feat_assessment_table(
            raw_df=raw_last,
            scored_df=scored_df,
            metadata=metadata,
            has_reason_codes="decision_reason_code" in scored_df.columns,
            has_explanations=("contrib_pack" in st.session_state),
            has_override_logging=OVERRIDE_LOG_PATH.exists(),
        )

        st.markdown("### FEAT-style governance view")
        st.dataframe(feat_table, use_container_width=True)

        st.markdown("### Group outcome disparity view")

        fairness_control_col1, fairness_control_col2 = st.columns(2)

        with fairness_control_col1:
            min_group_size = st.slider("Minimum group size", 50, 1000, 200, 50)

        with fairness_control_col2:
            max_unique = st.slider(
                "Maximum distinct groups shown",
                5,
                50,
                20,
                1,
                help="Higher values allow more granular fields such as state or zip3, but may be noisier."
            )

        candidate_group_cols, fairness_table_map = precompute_fairness_tables(
            raw_df=raw_last,
            scored_df=scored_df,
            min_group_size=min_group_size,
            max_unique=max_unique,
        )

        if candidate_group_cols:
            group_col = st.selectbox("Group column", candidate_group_cols)

            group_table = fairness_table_map.get(group_col, pd.DataFrame())

            if group_table.empty:
                st.warning("No groups met the current minimum size threshold for this field.")
            else:
                st.dataframe(group_table, use_container_width=True)

                chart_df = group_table.melt(
                    id_vars=[group_col, "n"],
                    value_vars=["approve_rate", "review_rate", "reject_rate"],
                    var_name="metric",
                    value_name="value",
                )

                fig_group = px.bar(
                    chart_df,
                    x=group_col,
                    y="value",
                    color="metric",
                    barmode="group",
                    hover_data=["n"],
                    title=f"Outcome rates by {group_col}"
                )
                st.plotly_chart(fig_group, use_container_width=True)

                st.caption(
                    "Only low/medium-cardinality fields present in the current input are shown here. "
                    "Increase 'Maximum distinct groups shown' to expose more granular segment fields."
                )
        else:
            st.info("No suitable grouping columns were found for disparity diagnostics in the current input.")

    with tab5:
        st.subheader("Top recommended scenarios")
        st.dataframe(recommendation_df, use_container_width=True, height=250)

        st.subheader("Full scenario grid")
        st.dataframe(scenario_grid, use_container_width=True, height=350)

    with tab6:
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

        st.markdown("### Input quality / drift")

        raw_last = st.session_state.get("last_raw_input_df", default_df)
        drift_report = build_drift_report(default_df, raw_last)

        st.markdown("#### Schema summary")
        st.dataframe(drift_report["schema_summary"], use_container_width=True)

        ref_only_cols = drift_report["reference_only_cols"]
        cur_only_cols = drift_report["current_only_cols"]

        if ref_only_cols:
            st.warning("Reference-only columns: " + ", ".join(ref_only_cols[:20]))
        if cur_only_cols:
            st.warning("Current-only columns: " + ", ".join(cur_only_cols[:20]))

        st.markdown("#### Missingness change")
        st.dataframe(
            drift_report["missingness_table"].head(20),
            use_container_width=True,
            height=300
        )

        st.markdown("#### Numeric PSI")
        psi_table = drift_report["numeric_psi_table"]
        st.dataframe(psi_table, use_container_width=True, height=250)

        if not psi_table.empty:
            fig_psi = px.bar(
                psi_table.sort_values("psi", ascending=True),
                x="psi",
                y="column",
                color="severity",
                orientation="h",
                title="Numeric drift by PSI"
            )
            st.plotly_chart(fig_psi, use_container_width=True)

        st.markdown("#### Unseen categorical values")
        st.dataframe(
            drift_report["unseen_category_table"],
            use_container_width=True,
            height=250
        )

        st.markdown("### Run history (current session)")
        history_df = run_history_df(st.session_state)
        st.dataframe(history_df, use_container_width=True, height=250)

        if len(history_df) >= 2:
            hist_plot_df = history_df.copy()
            hist_plot_df["run_number"] = range(1, len(hist_plot_df) + 1)

            fig_hist_runs = px.line(
                hist_plot_df,
                x="run_number",
                y=["approve_rate", "review_rate", "reject_rate", "avg_pd_overall"],
                markers=True,
                title="Run history trends"
            )
            st.plotly_chart(fig_hist_runs, use_container_width=True)

        st.markdown("### Reviewer override log")
        override_log_df = load_override_log(OVERRIDE_LOG_PATH)
        st.dataframe(override_log_df, use_container_width=True, height=250)

    with tab7:
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

        history_df = run_history_df(st.session_state)
        if not history_df.empty:
            st.download_button(
                "Download run history CSV",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="run_history.csv",
                mime="text/csv"
            )

        override_log_df = load_override_log(OVERRIDE_LOG_PATH)
        if not override_log_df.empty:
            st.download_button(
                "Download reviewer override log CSV",
                data=override_log_df.to_csv(index=False).encode("utf-8"),
                file_name="reviewer_overrides.csv",
                mime="text/csv"
            )

        raw_last = st.session_state.get("last_raw_input_df", default_df)
        drift_report = build_drift_report(default_df, raw_last)

        for table_name in ["missingness_table", "numeric_psi_table", "unseen_category_table"]:
            df_out = drift_report.get(table_name, pd.DataFrame())
            if not df_out.empty:
                st.download_button(
                    f"Download {table_name}.csv",
                    data=df_out.to_csv(index=False).encode("utf-8"),
                    file_name=f"{table_name}.csv",
                    mime="text/csv"
                )