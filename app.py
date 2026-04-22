from __future__ import annotations

import base64
import hashlib
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# -----------------------------
# Page config and styling
# -----------------------------
st.set_page_config(
    page_title="Punjab Startup Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
:root {
    --blue-900: #103a71;
    --blue-800: #1d4f91;
    --blue-700: #316fc2;
}
.main {
    background: linear-gradient(180deg, #fbfdff 0%, #f2f7ff 100%);
}
.block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: var(--blue-900);
}
.dashboard-banner {
    background: linear-gradient(90deg, var(--blue-900), var(--blue-700));
    padding: 1.15rem 1.35rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(16,58,113,0.14);
}
.metric-card {
    background: white;
    border: 1px solid #e7eefb;
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(31,77,143,0.08);
    height: 100%;
}
.small-note {
    color: #5b677a;
    font-size: 0.92rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f4f8ff 0%, #eaf2ff 100%);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
}
.stTabs [data-baseweb="tab"] {
    background: #eef5ff;
    border-radius: 10px;
    padding: 0.5rem 0.9rem;
}
.stTabs [aria-selected="true"] {
    background: #dbeafe !important;
    color: #123a73 !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------
# Utility helpers
# -----------------------------
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
        return str(value) if value is not None else default
    except Exception:
        return default


GITHUB_TOKEN = get_secret("GITHUB_TOKEN")
GITHUB_REPO = get_secret("GITHUB_REPO")  # owner/repo
GITHUB_BRANCH = get_secret("GITHUB_BRANCH", "main")
GITHUB_DATA_PATH = get_secret("GITHUB_DATA_PATH", "data/students.csv")


def github_configured() -> bool:
    return bool(GITHUB_TOKEN and GITHUB_REPO and GITHUB_BRANCH and GITHUB_DATA_PATH)


LOCAL_CANDIDATES = [
    Path("students.csv"),
    Path("data/students.csv"),
    Path("students.xlsx"),
    Path("data/students.xlsx"),
]


def file_digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


@st.cache_data(show_spinner=False)
def parse_bytes_to_df(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    lower = file_name.lower()
    bio = io.BytesIO(file_bytes)
    if lower.endswith(".csv"):
        return pd.read_csv(bio, low_memory=False)
    if lower.endswith((".xlsx", ".xls")):
        return pd.read_excel(bio)
    raise ValueError("Please upload a CSV or Excel file.")


@st.cache_data(show_spinner=False)
def fetch_github_file(repo: str, branch: str, path: str, token: str):
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"

    base_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.object+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    resp = requests.get(api_url, headers=base_headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    sha = payload.get("sha", "")
    encoding = payload.get("encoding")
    content = payload.get("content")


    # Larger files: GitHub may return encoding="none" and no inline content
    download_url = payload.get("download_url")
    if download_url:
        raw_resp = requests.get(download_url, timeout=60)
        raw_resp.raise_for_status()
        return raw_resp.content, sha

    # Fallback: request raw content directly from the contents API
    raw_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    raw_resp = requests.get(api_url, headers=raw_headers, timeout=60)
    raw_resp.raise_for_status()

    # If GitHub returns raw bytes, use them
    if raw_resp.content:
        return raw_resp.content, sha

    raise RuntimeError("Could not retrieve GitHub file content.")

@st.cache_data(show_spinner=False)
def load_local_file(path_str: str) -> tuple[pd.DataFrame, str]:
    path = Path(path_str)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False), path.as_posix()
    return pd.read_excel(path), path.as_posix()


@st.cache_data(show_spinner=False)
def get_github_sha(repo: str, branch: str, path: str, token: str) -> str:
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    response = requests.get(api_url, headers=headers, timeout=30)
    if response.status_code == 200:
        return response.json().get("sha", "")
    if response.status_code == 404:
        return ""
    raise RuntimeError(f"GitHub SHA lookup failed: {response.status_code} {response.text[:200]}")


def upload_file_to_github(file_bytes: bytes, destination_path: str, commit_message: str) -> str:
    if not github_configured():
        raise RuntimeError("GitHub persistence is not configured in Streamlit secrets.")

    existing_sha = get_github_sha(GITHUB_REPO, GITHUB_BRANCH, destination_path, GITHUB_TOKEN)
    api_url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{destination_path}"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    payload = {
        "message": commit_message,
        "content": base64.b64encode(file_bytes).decode("utf-8"),
        "branch": GITHUB_BRANCH,
    }
    if existing_sha:
        payload["sha"] = existing_sha

    response = requests.put(api_url, headers=headers, data=json.dumps(payload), timeout=60)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"GitHub upload failed: {response.status_code} {response.text[:300]}")
    fetch_github_file.clear()
    get_github_sha.clear()
    return destination_path



def clean_text_value(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
    )



def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    for col in out.columns:
        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_string_dtype(out[col]):
            out[col] = out[col].replace({np.inf: np.nan, -np.inf: np.nan})

    likely_date_cols = [
        "Date of sign up start",
        "Student Created At",
        "Date of activation",
        "Last active date",
    ]
    for col in likely_date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    numeric_hints = [
        "Activities completed",
        "Tasks completed",
        "Points",
        "Badges",
        "Revenue",
        "No of sales",
        "No of offerings",
        "Final Activity Score",
        "Final Point Score",
        "Final Revenue Score",
        "Term 1 Final Score",
        "No of track tried",
    ]
    numeric_cols = [
        c
        for c in out.columns
        if c in numeric_hints or (str(c).startswith("Term ") and "Milestone Completed" in str(c))
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    return out



def pct(part: float, whole: float) -> float:
    return 0.0 if whole == 0 else (part / whole) * 100.0



def percentage_table(series: pd.Series, label: str, top_n: Optional[int] = None) -> pd.DataFrame:
    clean = clean_text_value(series).fillna("Unknown")
    counts = clean.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    total = max(len(clean), 1)
    out = counts.rename_axis(label).reset_index(name="Students")
    out["Percentage"] = (out["Students"] / total * 100).round(2)
    return out



def sanitize_for_display(df: pd.DataFrame, max_rows: Optional[int] = None) -> pd.DataFrame:
    out = df.copy()
    if max_rows is not None:
        out = out.head(max_rows).copy()

    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_float_dtype(out[col]):
            out[col] = out[col].replace([np.inf, -np.inf], np.nan)
        else:
            out[col] = out[col].astype("string")
    return out



def safe_dataframe(df: pd.DataFrame, height: int = 420, max_rows: Optional[int] = None, hide_index: bool = True):
    st.dataframe(
        sanitize_for_display(df, max_rows=max_rows),
        use_container_width=True,
        height=height,
        hide_index=hide_index,
    )



def plot_bar(df_plot: pd.DataFrame, x: str, y: str, title: str, height: int = 430):
    fig = px.bar(df_plot, x=x, y=y, text=y, title=title)
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color="#123a73",
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=50, b=10),
        height=height,
    )
    if y.lower().startswith("percent"):
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    else:
        fig.update_traces(textposition="outside")
    return fig



def plot_pie(df_plot: pd.DataFrame, names: str, values: str, title: str):
    fig = px.pie(df_plot, names=names, values=values, hole=0.45, title=title)
    fig.update_layout(
        paper_bgcolor="white",
        title_font_color="#123a73",
        margin=dict(l=10, r=10, t=50, b=10),
        height=430,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return fig



def faculty_missing_mask(series: pd.Series) -> pd.Series:
    cleaned = clean_text_value(series)
    return cleaned.isna() | cleaned.str.lower().isin(["n/a", "na", "not assigned", "none"])



def build_summary_metrics(df: pd.DataFrame):
    total = len(df)
    idx = df.index

    term_cols = [c for c in df.columns if str(c).startswith("Term ") and str(c).endswith("Milestone Completed")]
    activities = df["Activities completed"].fillna(0) if "Activities completed" in df.columns else pd.Series(0, index=idx)
    tasks = df["Tasks completed"].fillna(0) if "Tasks completed" in df.columns else pd.Series(0, index=idx)
    term1 = df["Term 1 Milestone Completed"].fillna(0) if "Term 1 Milestone Completed" in df.columns else pd.Series(0, index=idx)
    term2 = df["Term 2 Milestone Completed"].fillna(0) if "Term 2 Milestone Completed" in df.columns else pd.Series(0, index=idx)
    activation = df["Date of activation"] if "Date of activation" in df.columns else pd.Series(pd.NaT, index=idx)
    last_active = df["Last active date"] if "Last active date" in df.columns else pd.Series(pd.NaT, index=idx)

    milestone_any = pd.Series(False, index=idx)
    if term_cols:
        milestone_any = df[term_cols].fillna(0).gt(0).any(axis=1)

    today = pd.Timestamp.today().normalize()
    six_month_cutoff = today - pd.Timedelta(days=180)

    engagement_any = (activities > 0) | (tasks > 0) | milestone_any
    ever_activated = activation.notna()
    active_last_6m = activation.notna() & last_active.notna() & (last_active >= six_month_cutoff)
    did_not_start = (~engagement_any) & activation.isna() & last_active.isna()
    term1_done_not_term2 = (term1 > 0) & (term2 <= 0)
    inactive_since_term1 = term1_done_not_term2 & (~active_last_6m)

    metrics = {
        "Total Students": total,
        "% Completed Milestones": round(pct(float(milestone_any.sum()), total), 2),
        "% Didn’t Start": round(pct(float(did_not_start.sum()), total), 2),
        "% Active in Last 6 Months": round(pct(float(active_last_6m.sum()), total), 2),
        "% Did Term 1 but Not Active in Term 2": round(pct(float(inactive_since_term1.sum()), total), 2),
        "Activity Cutoff": six_month_cutoff.strftime("%d %b %Y"),
        "Latest Last Active Date": last_active.max().strftime("%d %b %Y") if last_active.notna().any() else "N/A",
    }

    flags = pd.DataFrame(
        {
            "Completed Milestones": milestone_any,
            "Did Not Start": did_not_start,
            "Active Last 6 Months": active_last_6m,
            "Did Term 1, Not Active in Term 2": inactive_since_term1,
            "Any Engagement": engagement_any,
            "Ever Activated": ever_activated,
        },
        index=idx,
    )
    return metrics, flags, six_month_cutoff



def render_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div style='font-size:0.95rem;color:#5b677a;'>{label}</div>
            <div style='font-size:1.8rem;font-weight:700;color:#123a73;margin-top:0.25rem;'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )



def make_safe_mask(series: pd.Series, value: str) -> pd.Series:
    normalized = clean_text_value(series).fillna("Unknown")
    return normalized == value



def render_drilldown_section(df_source: pd.DataFrame, entity_col: str, section_title: str):
    if entity_col not in df_source.columns:
        st.info(f"{entity_col} column not found.")
        return

    options = sorted(clean_text_value(df_source[entity_col]).fillna("Unknown").unique().tolist())
    selected_entity = st.selectbox(
        f"Search and select {section_title}",
        options=options,
        index=0,
        key=f"drilldown_{entity_col}",
    )

    entity_df = df_source[make_safe_mask(df_source[entity_col], selected_entity)].copy()
    entity_metrics, _, entity_cutoff = build_summary_metrics(entity_df)

    st.markdown(f"### {section_title}: {selected_entity}")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        render_metric("Students", f"{len(entity_df):,}")
    with m2:
        render_metric("Completed Milestones", f"{entity_metrics['% Completed Milestones']:.2f}%")
    with m3:
        render_metric("Didn’t Start", f"{entity_metrics['% Didn’t Start']:.2f}%")
    with m4:
        render_metric("Active in Last 6 Months", f"{entity_metrics['% Active in Last 6 Months']:.2f}%")

    st.caption(f"Fixed 6-month activity cutoff used here: {entity_cutoff.strftime('%d %b %Y')}")

    left, right = st.columns(2)
    with left:
        if "Track Name" in entity_df.columns:
            track_tbl = percentage_table(entity_df["Track Name"], label="Track Name")
            st.plotly_chart(plot_bar(track_tbl.head(15), "Track Name", "Percentage", f"Track Mix in {selected_entity}"), use_container_width=True)
    with right:
        if "Term 1 Final Grade" in entity_df.columns:
            grade_tbl = percentage_table(entity_df["Term 1 Final Grade"], label="Term 1 Final Grade")
            st.plotly_chart(plot_bar(grade_tbl, "Term 1 Final Grade", "Percentage", f"Term 1 Grade Mix in {selected_entity}"), use_container_width=True)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if "Faculty Name" in entity_df.columns:
            missing = faculty_missing_mask(entity_df["Faculty Name"])
            faculty_df = pd.DataFrame(
                {
                    "Faculty Assignment": ["Assigned", "Not Assigned"],
                    "Students": [int((~missing).sum()), int(missing.sum())],
                }
            )
            st.plotly_chart(plot_pie(faculty_df, "Faculty Assignment", "Students", "Faculty Assignment Split"), use_container_width=True)

    with lower_right:
        if "Last active date" in entity_df.columns:
            timeline = (
                entity_df.dropna(subset=["Last active date"])
                .assign(Month=lambda x: x["Last active date"].dt.to_period("M").astype(str))
                .groupby("Month")
                .size()
                .reset_index(name="Students")
            )
            if not timeline.empty:
                fig = px.line(timeline, x="Month", y="Students", markers=True, title=f"Monthly Active Trend - {selected_entity}")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", title_font_color="#123a73", height=430)
                st.plotly_chart(fig, use_container_width=True)

    snapshot_cols = [
        c for c in ["Name", "Email ID", "Track Name", "Term", "Activities completed", "Tasks completed", "Date of activation", "Last active date"]
        if c in entity_df.columns
    ]
    if snapshot_cols:
        st.markdown("#### Student snapshot")
        safe_dataframe(entity_df[snapshot_cols], height=360, max_rows=500)


# -----------------------------
# Data source resolution
# -----------------------------
def resolve_data_source(uploaded_file) -> tuple[pd.DataFrame, str, str]:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df = parse_bytes_to_df(file_bytes, uploaded_file.name)
        origin = f"Uploaded: {uploaded_file.name}"
        file_id = file_digest(file_bytes)
        st.session_state["active_file_bytes"] = file_bytes
        st.session_state["active_file_name"] = uploaded_file.name
        st.session_state["active_file_origin"] = origin
        st.session_state["active_file_id"] = file_id
        return df, origin, file_id

    if "active_file_bytes" in st.session_state and "active_file_name" in st.session_state:
        df = parse_bytes_to_df(st.session_state["active_file_bytes"], st.session_state["active_file_name"])
        return df, st.session_state.get("active_file_origin", "Uploaded in this session"), st.session_state.get("active_file_id", "session")

    if github_configured():
        file_bytes, _sha = fetch_github_file(GITHUB_REPO, GITHUB_BRANCH, GITHUB_DATA_PATH, GITHUB_TOKEN)
        file_name = Path(GITHUB_DATA_PATH).name
        df = parse_bytes_to_df(file_bytes, file_name)
        origin = f"GitHub: {GITHUB_REPO}/{GITHUB_DATA_PATH}@{GITHUB_BRANCH}"
        return df, origin, file_digest(file_bytes)

    for candidate in LOCAL_CANDIDATES:
        if candidate.exists():
            df, source = load_local_file(candidate.as_posix())
            return df, f"Local file: {source}", candidate.as_posix()

    raise FileNotFoundError(
        "No data source found. Upload a CSV/Excel file from the sidebar, or configure GitHub persistence in Streamlit secrets."
    )


# -----------------------------
# Header
# -----------------------------
st.markdown(
    """
    <div class='dashboard-banner'>
        <div style='font-size:2rem;font-weight:800;'>Punjab Startup Dashboard</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls
st.sidebar.header("Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

if uploaded_file is not None and uploaded_file.name:
    if st.sidebar.button("Save uploaded file for future app loads", use_container_width=True):
        if github_configured():
            try:
                target_path = GITHUB_DATA_PATH
                msg = f"Update dashboard data: {uploaded_file.name}"
                upload_file_to_github(uploaded_file.getvalue(), target_path, msg)
                st.sidebar.success(f"Saved to GitHub at `{target_path}`. Future loads will use this file until you upload and save another one.")
            except Exception as exc:
                st.sidebar.error(str(exc))
        else:
            st.sidebar.warning(
                "GitHub persistence is not set up yet. Add `GITHUB_TOKEN`, `GITHUB_REPO`, `GITHUB_BRANCH`, and `GITHUB_DATA_PATH` in Streamlit secrets to enable this."
            )

try:
    raw_df, source_name, source_id = resolve_data_source(uploaded_file)
except Exception as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

# Clean and analyze
df = clean_dataframe(raw_df)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.caption(f"Current source: {source_name}")
st.sidebar.caption(f"Rows loaded: {len(df):,}")
if github_configured():
    st.sidebar.caption("Persistence: GitHub enabled")
else:
    st.sidebar.caption("Persistence: session/local only")

# Filters
st.sidebar.header("Filters")
if "Status" in df.columns:
    # Ignored by request.
    pass

for col in ["University Name", "College Name", "Track Name", "Term"]:
    if col in df.columns:
        options = sorted(clean_text_value(df[col]).dropna().unique().tolist())
        selected = st.sidebar.multiselect(col, options)
        if selected:
            df = df[clean_text_value(df[col]).isin(selected)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Rows after filters: {len(df):,}")

metrics, flags, six_month_cutoff = build_summary_metrics(df)

if "Email ID" in df.columns:
    duplicate_emails = int(df["Email ID"].dropna().duplicated().sum())
else:
    duplicate_emails = None

overview_tab, institutions_tab, engagement_tab, faculty_tab, grades_tab, drilldown_tab, data_tab = st.tabs(
    [
        "Overview",
        "Institutions",
        "Engagement",
        "Faculty",
        "Grades & Terms",
        "Drilldown",
        "Data Explorer",
    ]
)

with overview_tab:
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric("Total Students", f"{metrics['Total Students']:,}")
    with c2:
        render_metric("Completed Milestones", f"{metrics['% Completed Milestones']:.2f}%")
    with c3:
        render_metric("Didn’t Start", f"{metrics['% Didn’t Start']:.2f}%")
    with c4:
        render_metric("Active in Last 6 Months", f"{metrics['% Active in Last 6 Months']:.2f}%")
    with c5:
        render_metric("Term 1 Done, Not Active in Term 2", f"{metrics['% Did Term 1 but Not Active in Term 2']:.2f}%")

    st.caption(
        f"Activity uses only `Date of activation` and `Last active date`. Fixed 180-day cutoff from today: {metrics['Activity Cutoff']}. Latest last-active date in filtered data: {metrics['Latest Last Active Date']}."
    )

    a1, a2 = st.columns(2)
    with a1:
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Completed Milestones",
                    "Didn’t Start",
                    "Active in Last 6 Months",
                    "Term 1 Done, Not Active in Term 2",
                ],
                "Percentage": [
                    metrics["% Completed Milestones"],
                    metrics["% Didn’t Start"],
                    metrics["% Active in Last 6 Months"],
                    metrics["% Did Term 1 but Not Active in Term 2"],
                ],
            }
        )
        st.plotly_chart(plot_bar(summary_df, "Metric", "Percentage", "Core Overview Metrics"), use_container_width=True)

    with a2:
        state_df = pd.DataFrame(
            {
                "Activation State": ["Active in Last 6 Months", "Not Active in Last 6 Months"],
                "Students": [int(flags["Active Last 6 Months"].sum()), int((~flags["Active Last 6 Months"]).sum())],
            }
        )
        st.plotly_chart(plot_pie(state_df, "Activation State", "Students", "Activation Status Distribution"), use_container_width=True)

    st.subheader("Quick insights")
    q1, q2, q3 = st.columns(3)
    with q1:
        if "College Name" in df.columns:
            safe_dataframe(percentage_table(df["College Name"], label="College Name", top_n=10), height=320)
    with q2:
        if "University Name" in df.columns:
            safe_dataframe(percentage_table(df["University Name"], label="University Name", top_n=10), height=320)
    with q3:
        if "Track Name" in df.columns:
            safe_dataframe(percentage_table(df["Track Name"], label="Track Name"), height=320)

    st.subheader("Data quality checks")
    dq1, dq2, dq3 = st.columns(3)
    with dq1:
        if duplicate_emails is not None:
            render_metric("Duplicate Email IDs", f"{duplicate_emails:,}")
    with dq2:
        render_metric("Ever Activated", f"{int(flags['Ever Activated'].sum()):,}")
    with dq3:
        render_metric("Not Active in Last 6 Months", f"{int((~flags['Active Last 6 Months']).sum()):,}")

with institutions_tab:
    left, right = st.columns(2)
    if "College Name" in df.columns:
        college_tbl = percentage_table(df["College Name"], label="College Name", top_n=20)
        with left:
            st.plotly_chart(plot_bar(college_tbl, "College Name", "Percentage", "% of Students in Each College Name", height=500), use_container_width=True)
            safe_dataframe(college_tbl, height=360)
    if "University Name" in df.columns:
        uni_tbl = percentage_table(df["University Name"], label="University Name", top_n=20)
        with right:
            st.plotly_chart(plot_bar(uni_tbl, "University Name", "Percentage", "% of Students in Each University Name", height=500), use_container_width=True)
            safe_dataframe(uni_tbl, height=360)

    st.subheader("Track distribution")
    if "Track Name" in df.columns:
        track_tbl = percentage_table(df["Track Name"], label="Track Name")
        t1, t2 = st.columns([1.2, 1])
        with t1:
            st.plotly_chart(plot_bar(track_tbl, "Track Name", "Percentage", "% of Students in Each Track Name"), use_container_width=True)
        with t2:
            safe_dataframe(track_tbl, height=360)

with engagement_tab:
    st.subheader("Activity and milestone analysis")
    e1, e2 = st.columns(2)

    with e1:
        if "Activities completed" in df.columns:
            activity_series = pd.to_numeric(df["Activities completed"], errors="coerce").fillna(0)
            bucket_df = pd.DataFrame(
                {
                    "Bucket": ["No activity", "Low (1-10)", "Moderate (11-50)", "High (51+)"],
                    "Students": [
                        int((activity_series == 0).sum()),
                        int(activity_series.between(1, 10).sum()),
                        int(activity_series.between(11, 50).sum()),
                        int((activity_series > 50).sum()),
                    ],
                }
            )
            bucket_df["Percentage"] = (bucket_df["Students"] / max(len(df), 1) * 100).round(2)
            st.plotly_chart(plot_bar(bucket_df, "Bucket", "Percentage", "Activity Intensity Split"), use_container_width=True)

    with e2:
        if "Last active date" in df.columns:
            timeline = (
                df.dropna(subset=["Last active date"])
                .assign(Month=lambda x: x["Last active date"].dt.to_period("M").astype(str))
                .groupby("Month")
                .size()
                .reset_index(name="Students")
            )
            if not timeline.empty:
                fig = px.line(timeline, x="Month", y="Students", markers=True, title="Students Active by Month")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", title_font_color="#123a73", height=430)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Milestone progression by term")
    term_milestone_cols = [c for c in df.columns if c.startswith("Term ") and c.endswith("Milestone Completed")]
    if term_milestone_cols:
        milestone_pct_df = pd.DataFrame(
            [
                {
                    "Term Milestone": col.replace(" Milestone Completed", ""),
                    "Percentage": round((pd.to_numeric(df[col], errors="coerce").fillna(0) > 0).mean() * 100, 2),
                }
                for col in term_milestone_cols
            ]
        )
        st.plotly_chart(plot_bar(milestone_pct_df, "Term Milestone", "Percentage", "Students Completing Each Term Milestone"), use_container_width=True)
        safe_dataframe(milestone_pct_df, height=300)

with faculty_tab:
    st.subheader("Faculty assignment coverage")
    if "College Name" in df.columns and "Faculty Name" in df.columns:
        faculty_gap = (
            df.assign(FacultyMissing=faculty_missing_mask(df["Faculty Name"]))
            .groupby("College Name")
            .agg(Total_Students=("College Name", "size"), Faculty_Not_Assigned=("FacultyMissing", "sum"))
            .reset_index()
        )
        faculty_gap["Percentage"] = (faculty_gap["Faculty_Not_Assigned"] / faculty_gap["Total_Students"] * 100).round(2)
        faculty_gap = faculty_gap.sort_values(["Percentage", "Faculty_Not_Assigned"], ascending=[False, False])

        f1, f2 = st.columns([1.35, 1])
        with f1:
            st.plotly_chart(plot_bar(faculty_gap.head(20), "College Name", "Percentage", "% of Students in Each College Where Faculty is N/A", height=520), use_container_width=True)
        with f2:
            safe_dataframe(faculty_gap, height=520)

with grades_tab:
    st.subheader("Term 1 final grade by college")
    if "College Name" in df.columns and "Term 1 Final Grade" in df.columns:
        grade_heat = (
            df.assign(**{"Term 1 Final Grade": clean_text_value(df["Term 1 Final Grade"]).fillna("Unknown")})
            .groupby(["College Name", "Term 1 Final Grade"])
            .size()
            .reset_index(name="Students")
        )
        totals = grade_heat.groupby("College Name")["Students"].transform("sum")
        grade_heat["Percentage"] = (grade_heat["Students"] / totals * 100).round(2)

        top_colleges = clean_text_value(df["College Name"]).fillna("Unknown").value_counts().head(20).index.tolist()
        heat_top = grade_heat[grade_heat["College Name"].isin(top_colleges)]
        pivot = heat_top.pivot(index="College Name", columns="Term 1 Final Grade", values="Percentage").fillna(0)

        if not pivot.empty:
            fig = go.Figure(
                data=go.Heatmap(
                    z=pivot.values,
                    x=pivot.columns.tolist(),
                    y=pivot.index.tolist(),
                    text=np.round(pivot.values, 1),
                    texttemplate="%{text}%",
                    colorscale="Blues",
                )
            )
            fig.update_layout(
                title="% of Students in Each College by Term 1 Final Grade (Top 20 Colleges)",
                paper_bgcolor="white",
                plot_bgcolor="white",
                title_font_color="#123a73",
                height=650,
            )
            st.plotly_chart(fig, use_container_width=True)

        safe_dataframe(grade_heat.sort_values(["College Name", "Percentage"], ascending=[True, False]), height=420, max_rows=2000)

    st.subheader("Term-level comparison")
    if "Term" in df.columns:
        term_dist = percentage_table(df["Term"], label="Term")
        g1, g2 = st.columns([1.1, 1])
        with g1:
            st.plotly_chart(plot_bar(term_dist, "Term", "Percentage", "Student Share by Term"), use_container_width=True)
        with g2:
            safe_dataframe(term_dist, height=260)

with drilldown_tab:
    st.subheader("Searchable college and university drilldown")
    st.caption("Pick one college or one university to see a focused breakdown.")
    drill_mode = st.radio("Drill down by", ["College Name", "University Name"], horizontal=True)
    if drill_mode == "College Name":
        render_drilldown_section(df, "College Name", "College")
    else:
        render_drilldown_section(df, "University Name", "University")

with data_tab:
    st.subheader("Data explorer")
    safe_dataframe(df, height=420, max_rows=5000)

    csv_buffer = io.StringIO()
    sanitize_for_display(df).to_csv(csv_buffer, index=False)
    st.download_button(
        "Download filtered data as CSV",
        data=csv_buffer.getvalue(),
        file_name="filtered_students_data.csv",
        mime="text/csv",
    )

    st.markdown("### Column completeness")
    completeness = pd.DataFrame(
        {
            "Column": df.columns,
            "Non-null %": ((df.notna().sum() / max(len(df), 1)) * 100).round(2),
            "Missing %": ((df.isna().sum() / max(len(df), 1)) * 100).round(2),
        }
    ).sort_values("Missing %", ascending=False)
    safe_dataframe(completeness, height=420)

with st.expander("Metric definitions used in this dashboard"):
    st.markdown(
        """
        - **Completed milestones**: a student has a value greater than 0 in at least one `Term X Milestone Completed` column.
        - **Didn’t start**: `Activities completed = 0`, `Tasks completed = 0`, no term milestone completed, no activation date, and no last active date.
        - **Active in last 6 months**: `Date of activation` is present and `Last active date` falls within the last 180 days from today.
        - **Did semester/term 1 but not active in 2**: `Term 1 Milestone Completed > 0`, `Term 2 Milestone Completed = 0`, and not active in the last 6 months.
        - **Faculty not assigned**: `Faculty Name` is blank or marked as `N/A`, `NA`, or similar.
        - **GitHub persistence**: if configured in Streamlit secrets, the uploaded file can be saved to GitHub and used automatically on later app loads until another file is uploaded and saved.
        """
    )
