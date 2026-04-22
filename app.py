
import base64
import io
import json
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st


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
    --blue-100: #eff6ff;
    --blue-050: #f8fbff;
    --white: #ffffff;
    --slate: #5b677a;
    --border: #e6eef9;
}
.main {
    background: linear-gradient(180deg, #fbfdff 0%, #f2f7ff 100%);
}
.block-container {
    padding-top: 1.15rem;
    padding-bottom: 2rem;
}
h1, h2, h3 {
    color: var(--blue-900);
}
.dashboard-banner {
    background: linear-gradient(90deg, var(--blue-900), var(--blue-700));
    padding: 1.1rem 1.3rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 10px 30px rgba(16,58,113,0.14);
}
.metric-card {
    background: white;
    border: 1px solid var(--border);
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(31,77,143,0.08);
    height: 100%;
}
.metric-label {
    color: var(--slate);
    font-size: 0.92rem;
    margin-bottom: 0.25rem;
}
.metric-value {
    color: var(--blue-900);
    font-size: 1.8rem;
    font-weight: 700;
    line-height: 1.1;
}
.metric-sub {
    color: var(--slate);
    font-size: 0.85rem;
    margin-top: 0.35rem;
}
.small-note {
    color: var(--slate);
    font-size: 0.9rem;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f6faff 0%, #edf5ff 100%);
    border-right: 1px solid var(--border);
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

TITLE = "Punjab Startup Dashboard"

DATE_COLUMNS = [
    "Date of sign up start",
    "Student Created At",
    "Date of activation",
    "Last active date",
]

TEXT_COLUMNS = [
    "Name",
    "Email ID",
    "College Name",
    "University Name",
    "Faculty Name",
    "Track Name",
    "Business Name",
    "Business Category",
    "Term 1 Final Grade",
]

NUMERIC_COLUMNS = [
    "No of track tried",
    "Term",
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
]

MILESTONE_COLUMNS = [f"Term {i} Milestone Completed" for i in range(1, 11)]

DISPLAY_COLUMNS = [
    "Name",
    "Email ID",
    "College Name",
    "University Name",
    "Faculty Name",
    "Track Name",
    "Term",
    "Date of activation",
    "Last active date",
    "Activities completed",
    "Tasks completed",
] + MILESTONE_COLUMNS[:4] + ["Term 1 Final Grade"]


def safe_text(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def normalize_blank_series(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().replace(
        {"": pd.NA, "nan": pd.NA, "None": pd.NA, "NAN": pd.NA}
    )


def parse_dates(series: pd.Series) -> pd.Series:
    """Robust date parsing across pandas versions and mixed values."""
    s = series.copy()
    s = s.replace({0: np.nan, "0": np.nan, "0.0": np.nan, "": np.nan})
    parsed = pd.to_datetime(s, errors="coerce")

    remaining = parsed.isna()
    if remaining.any():
        numeric = pd.to_numeric(s.where(remaining), errors="coerce")
        numeric_mask = numeric.notna()
        if numeric_mask.any():
            excel_dates = pd.to_datetime("1899-12-30") + pd.to_timedelta(
                numeric[numeric_mask], unit="D"
            )
            parsed.loc[numeric_mask.index[numeric_mask]] = excel_dates.values
    return parsed


def positive_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.fillna(0).gt(0)


def non_empty_non_zero_mask(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    is_non_zero_numeric = numeric.notna() & numeric.ne(0)
    text = normalize_blank_series(series)
    has_text = text.notna() & ~text.isin(["0", "0.0"])
    return is_non_zero_numeric | has_text


def format_pct(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "0.0%"
    return f"{(numerator / denominator) * 100:.1f}%"


def make_display_safe(df: pd.DataFrame) -> pd.DataFrame:
    safe_df = df.copy()
    for col in safe_df.columns:
        if pd.api.types.is_datetime64_any_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].dt.strftime("%Y-%m-%d")
        elif pd.api.types.is_timedelta64_dtype(safe_df[col]):
            safe_df[col] = safe_df[col].astype(str)
        else:
            safe_df[col] = safe_df[col].map(lambda x: None if pd.isna(x) else str(x))
    return safe_df


def styled_metric(label: str, value: str, sub: str = ""):
    st.markdown(
        f"""
        <div class='metric-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
            <div class='metric-sub'>{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bar(df: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None):
    if df.empty:
        st.info("No data available for this view.")
        return
    fig = px.bar(df, x=x, y=y, title=title, color=color, text=y)
    fig.update_traces(texttemplate="%{text:.1f}" if pd.api.types.is_numeric_dtype(df[y]) else "%{text}", textposition="outside")
    fig.update_layout(
        height=430,
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_pie(df: pd.DataFrame, names: str, values: str, title: str):
    if df.empty:
        st.info("No data available for this view.")
        return
    fig = px.pie(df, names=names, values=values, title=title, hole=0.55)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, use_container_width=True)


def top_share_table(df: pd.DataFrame, column: str, top_n: int = 15) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame(columns=[column, "Students", "% of valid rows"])
    values = normalize_blank_series(df[column])
    counts = values.dropna().value_counts().head(top_n)
    total = counts.sum()
    if total == 0:
        return pd.DataFrame(columns=[column, "Students", "% of valid rows"])
    out = counts.rename_axis(column).reset_index(name="Students")
    out["% of valid rows"] = out["Students"] / total * 100
    return out


def get_github_settings() -> Optional[dict]:
    try:
        token = st.secrets.get("GITHUB_TOKEN")
        repo = st.secrets.get("GITHUB_REPO")
        if not token or not repo:
            return None
        return {
            "token": token,
            "repo": repo,
            "branch": st.secrets.get("GITHUB_BRANCH", "main"),
            "path": st.secrets.get("GITHUB_DATA_PATH", "data/students.csv"),
        }
    except Exception:
        return None


def fetch_github_file(repo: str, branch: str, path: str, token: str) -> Tuple[bytes, str]:
    api_url = f"https://api.github.com/repos/{repo}/contents/{path}?ref={branch}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.object+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    resp = requests.get(api_url, headers=headers, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    sha = payload.get("sha", "")

    if payload.get("encoding") == "base64" and payload.get("content"):
        return base64.b64decode(payload["content"]), sha

    download_url = payload.get("download_url")
    if download_url:
        raw_resp = requests.get(download_url, timeout=60)
        raw_resp.raise_for_status()
        return raw_resp.content, sha

    raw_headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.raw+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    raw_resp = requests.get(api_url, headers=raw_headers, timeout=60)
    raw_resp.raise_for_status()
    if raw_resp.content:
        return raw_resp.content, sha

    raise RuntimeError("Could not retrieve GitHub file content.")


def save_github_file(repo: str, branch: str, path: str, token: str, content: bytes, commit_message: str):
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    sha = None
    get_resp = requests.get(f"{url}?ref={branch}", headers=headers, timeout=30)
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")
    elif get_resp.status_code not in (404,):
        get_resp.raise_for_status()

    payload = {
        "message": commit_message,
        "content": base64.b64encode(content).decode("utf-8"),
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=headers, data=json.dumps(payload), timeout=60)
    put_resp.raise_for_status()
    return put_resp.json()


def read_uploaded_or_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    lower = filename.lower()
    bio = io.BytesIO(file_bytes)
    if lower.endswith(".csv"):
        return pd.read_csv(bio, low_memory=False)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(bio)
    raise ValueError("Only CSV and Excel files are supported.")


def load_data_from_path(path: str) -> pd.DataFrame:
    lower = path.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(path, low_memory=False)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type for {path}")


@st.cache_data(show_spinner=False)
def preprocess_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()
    df.columns = [safe_text(c) for c in df.columns]

    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = normalize_blank_series(df[col])

    for col in DATE_COLUMNS:
        if col in df.columns:
            df[col] = parse_dates(df[col])

    for col in NUMERIC_COLUMNS + [c for c in MILESTONE_COLUMNS if c in df.columns]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    available_milestones = [c for c in MILESTONE_COLUMNS if c in df.columns]

    if available_milestones:
        milestone_positive = pd.concat([positive_mask(df[c]) for c in available_milestones], axis=1)
        df["Milestones Completed Count"] = milestone_positive.sum(axis=1)
    else:
        df["Milestones Completed Count"] = 0

    df["Activities Positive"] = positive_mask(df["Activities completed"]) if "Activities completed" in df.columns else False
    df["Tasks Positive"] = positive_mask(df["Tasks completed"]) if "Tasks completed" in df.columns else False

    df["Active Student"] = (
        df["Milestones Completed Count"].gt(0)
        | df["Activities Positive"]
        | df["Tasks Positive"]
    )

    df["Ever Activated"] = df["Date of activation"].notna() if "Date of activation" in df.columns else False

    if "Last active date" in df.columns:
        cutoff = pd.Timestamp.today().normalize() - pd.Timedelta(days=180)
        df["Active in Last 6 Months"] = df["Last active date"].notna() & df["Last active date"].ge(cutoff)
    else:
        df["Active in Last 6 Months"] = False

    df["Did Not Start"] = ~(
        df["Active Student"]
        | df["Ever Activated"]
        | df["Active in Last 6 Months"]
    )

    term1_positive = positive_mask(df["Term 1 Milestone Completed"]) if "Term 1 Milestone Completed" in df.columns else pd.Series(False, index=df.index)
    term2_positive = positive_mask(df["Term 2 Milestone Completed"]) if "Term 2 Milestone Completed" in df.columns else pd.Series(False, index=df.index)
    df["Did Term 1 but not active in 2"] = term1_positive & ~term2_positive

    if "Faculty Name" in df.columns:
        faculty_norm = normalize_blank_series(df["Faculty Name"])
        df["Faculty Unassigned"] = faculty_norm.isna() | faculty_norm.str.upper().eq("N/A")
    else:
        df["Faculty Unassigned"] = False

    return df


def choose_data_source() -> Tuple[pd.DataFrame, str, Optional[bytes], Optional[str]]:
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV or Excel",
        type=["csv", "xlsx", "xls"],
        help="Upload a fresh file anytime. You can also save the uploaded file to GitHub for future loads.",
    )

    github_cfg = get_github_settings()

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        df = read_uploaded_or_bytes(file_bytes, uploaded_file.name)
        st.session_state["uploaded_file_bytes"] = file_bytes
        st.session_state["uploaded_file_name"] = uploaded_file.name
        return df, f"Uploaded file: {uploaded_file.name}", file_bytes, uploaded_file.name

    if "uploaded_file_bytes" in st.session_state and "uploaded_file_name" in st.session_state:
        file_bytes = st.session_state["uploaded_file_bytes"]
        file_name = st.session_state["uploaded_file_name"]
        df = read_uploaded_or_bytes(file_bytes, file_name)
        return df, f"Current session file: {file_name}", file_bytes, file_name

    if github_cfg:
        try:
            raw_bytes, _ = fetch_github_file(
                repo=github_cfg["repo"],
                branch=github_cfg["branch"],
                path=github_cfg["path"],
                token=github_cfg["token"],
            )
            filename = github_cfg["path"].split("/")[-1]
            df = read_uploaded_or_bytes(raw_bytes, filename)
            return df, f"GitHub saved file: {github_cfg['path']}", None, None
        except Exception as exc:
            st.sidebar.warning(f"GitHub load skipped: {exc}")

    local_candidates = [
        "data/students.csv",
        "students.csv",
        "data/students.xlsx",
        "students.xlsx",
    ]
    for candidate in local_candidates:
        try:
            df = load_data_from_path(candidate)
            return df, f"Repository file: {candidate}", None, None
        except Exception:
            continue

    st.error("No data source found. Upload a CSV/Excel file, or keep the saved file in `data/students.csv`.")
    st.stop()


raw_df, source_label, uploaded_bytes, uploaded_name = choose_data_source()
df = preprocess_data(raw_df)
github_cfg = get_github_settings()

st.markdown(
    f"""
    <div class='dashboard-banner'>
        <h1 style='margin:0; color:white;'>{TITLE}</h1>
        <div style='opacity:0.95; margin-top:0.35rem;'>Source: {source_label}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Controls")

    if uploaded_bytes is not None and uploaded_name is not None:
        if github_cfg:
            if st.button("Save uploaded file for future app loads", use_container_width=True):
                save_name = uploaded_name.split("/")[-1]
                path = github_cfg["path"]
                if "." in path and path.split(".")[-1].lower() != save_name.split(".")[-1].lower():
                    path = "/".join(path.split("/")[:-1] + [save_name]) if "/" in path else save_name
                try:
                    save_github_file(
                        repo=github_cfg["repo"],
                        branch=github_cfg["branch"],
                        path=path,
                        token=github_cfg["token"],
                        content=uploaded_bytes,
                        commit_message=f"Update dashboard data: {save_name}",
                    )
                    st.success(f"Saved to GitHub at `{path}`")
                except Exception as exc:
                    st.error(f"GitHub save failed: {exc}")
        else:
            st.info("Add GitHub secrets to enable saving uploaded files for future app loads.")

    college_options = sorted(df["College Name"].dropna().unique().tolist()) if "College Name" in df.columns else []
    university_options = sorted(df["University Name"].dropna().unique().tolist()) if "University Name" in df.columns else []
    track_options = sorted(df["Track Name"].dropna().unique().tolist()) if "Track Name" in df.columns else []
    grade_options = sorted(df["Term 1 Final Grade"].dropna().unique().tolist()) if "Term 1 Final Grade" in df.columns else []

    selected_colleges = st.multiselect("College Name", college_options)
    selected_universities = st.multiselect("University Name", university_options)
    selected_tracks = st.multiselect("Track Name", track_options)
    selected_grades = st.multiselect("Term 1 Final Grade", grade_options)
    only_active_students = st.checkbox("Only active students", value=False)

filtered_df = df.copy()
if selected_colleges and "College Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["College Name"].isin(selected_colleges)]
if selected_universities and "University Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["University Name"].isin(selected_universities)]
if selected_tracks and "Track Name" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Track Name"].isin(selected_tracks)]
if selected_grades and "Term 1 Final Grade" in filtered_df.columns:
    filtered_df = filtered_df[filtered_df["Term 1 Final Grade"].isin(selected_grades)]
if only_active_students:
    filtered_df = filtered_df[filtered_df["Active Student"]]

n = len(filtered_df)
active_count = int(filtered_df["Active Student"].sum()) if "Active Student" in filtered_df.columns else 0
not_started_count = int(filtered_df["Did Not Start"].sum()) if "Did Not Start" in filtered_df.columns else 0
ever_activated_count = int(filtered_df["Ever Activated"].sum()) if "Ever Activated" in filtered_df.columns else 0
active_6m_count = int(filtered_df["Active in Last 6 Months"].sum()) if "Active in Last 6 Months" in filtered_df.columns else 0
term1_not2_count = int(filtered_df["Did Term 1 but not active in 2"].sum()) if "Did Term 1 but not active in 2" in filtered_df.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    styled_metric("Total Students", f"{n:,}", "Rows in current filtered view")
with c2:
    styled_metric("Active Students", format_pct(active_count, n), f"{active_count:,} students")
with c3:
    styled_metric("Didn't Start", format_pct(not_started_count, n), f"{not_started_count:,} students")
with c4:
    styled_metric("Ever Activated", format_pct(ever_activated_count, n), f"{ever_activated_count:,} students")
with c5:
    styled_metric("Active in Last 6 Months", format_pct(active_6m_count, n), f"{active_6m_count:,} students")

st.markdown(
    "<div class='small-note'>`Status` is ignored. Participation metrics treat 0 values the same as blank values and only count values above 0.</div>",
    unsafe_allow_html=True,
)

overview_tab, milestones_tab, drilldown_tab, explorer_tab = st.tabs(
    ["Overview", "Milestones", "Drilldown", "Data Explorer"]
)

with overview_tab:
    left, right = st.columns([1.15, 0.85])
    with left:
        st.markdown("### Overview")
        summary_df = pd.DataFrame(
            {
                "Metric": [
                    "Active Students",
                    "Didn't Start",
                    "Ever Activated",
                    "Active in Last 6 Months",
                    "Did Term 1 but not active in 2",
                ],
                "Percentage": [
                    active_count / n * 100 if n else 0,
                    not_started_count / n * 100 if n else 0,
                    ever_activated_count / n * 100 if n else 0,
                    active_6m_count / n * 100 if n else 0,
                    term1_not2_count / n * 100 if n else 0,
                ],
            }
        )
        render_bar(summary_df, "Metric", "Percentage", "Core student percentages")
    with right:
        st.markdown("### Term Distribution")
        if "Term" in filtered_df.columns:
            term_numeric = pd.to_numeric(filtered_df["Term"], errors="coerce")
            term_df = filtered_df[term_numeric.fillna(0).gt(0)].copy()
            if not term_df.empty:
                term_df["Term Clean"] = pd.to_numeric(term_df["Term"], errors="coerce").astype("Int64").astype(str)
                term_counts = term_df["Term Clean"].value_counts().sort_index().rename_axis("Term").reset_index(name="Students")
                render_pie(term_counts, "Term", "Students", "Students by Term (0 ignored)")
            else:
                st.info("No term values above 0.")
        else:
            st.info("Term column not available.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### College Name distribution")
        college_share = top_share_table(filtered_df, "College Name", top_n=20)
        render_bar(college_share, "College Name", "% of valid rows", "Top colleges by student share")
    with col2:
        st.markdown("### University Name distribution")
        uni_share = top_share_table(filtered_df, "University Name", top_n=20)
        render_bar(uni_share, "University Name", "% of valid rows", "Top universities by student share")

    col3, col4 = st.columns(2)
    with col3:
        st.markdown("### Track Name distribution")
        track_share = top_share_table(filtered_df, "Track Name", top_n=20)
        render_bar(track_share, "Track Name", "% of valid rows", "Top tracks by student share")
    with col4:
        st.markdown("### Faculty not assigned by college")
        if {"College Name", "Faculty Unassigned"}.issubset(filtered_df.columns):
            valid = filtered_df[filtered_df["College Name"].notna()].copy()
            faculty_df = (
                valid.groupby("College Name", dropna=False)["Faculty Unassigned"]
                .mean()
                .mul(100)
                .sort_values(ascending=False)
                .head(20)
                .rename("% Faculty N/A")
                .reset_index()
            )
            render_bar(faculty_df, "College Name", "% Faculty N/A", "Colleges with highest unassigned faculty share")
        else:
            st.info("Faculty or college columns not available.")

    st.markdown("### College-wise Term 1 Final Grade")
    if {"College Name", "Term 1 Final Grade"}.issubset(filtered_df.columns):
        grade_df = filtered_df[["College Name", "Term 1 Final Grade"]].dropna().copy()
        if not grade_df.empty:
            top_colleges = grade_df["College Name"].value_counts().head(12).index
            grade_df = grade_df[grade_df["College Name"].isin(top_colleges)]
            grade_grouped = grade_df.groupby(["College Name", "Term 1 Final Grade"]).size().reset_index(name="Students")
            fig = px.bar(
                grade_grouped,
                x="College Name",
                y="Students",
                color="Term 1 Final Grade",
                barmode="stack",
                title="Top colleges by Term 1 Final Grade mix",
            )
            fig.update_layout(height=460, xaxis_title=None, yaxis_title=None, margin=dict(l=10, r=10, t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No valid grade rows available.")

with milestones_tab:
    st.markdown("### Milestones")
    total_students = len(filtered_df)
    available_milestones = [c for c in MILESTONE_COLUMNS if c in filtered_df.columns]

    if available_milestones:
        cards1 = st.columns(min(5, len(available_milestones)))
        for idx, col in enumerate(available_milestones[:5]):
            count = int(positive_mask(filtered_df[col]).sum())
            with cards1[idx]:
                styled_metric(col.replace(" Completed", ""), format_pct(count, total_students), f"{count:,} students")

        if len(available_milestones) > 5:
            cards2 = st.columns(len(available_milestones[5:10]))
            for idx, col in enumerate(available_milestones[5:10]):
                count = int(positive_mask(filtered_df[col]).sum())
                with cards2[idx]:
                    styled_metric(col.replace(" Completed", ""), format_pct(count, total_students), f"{count:,} students")

        milestone_dist = (
            filtered_df["Milestones Completed Count"]
            .value_counts()
            .sort_index()
            .rename_axis("Milestones Completed")
            .reset_index(name="Students")
        )
        milestone_dist = milestone_dist[milestone_dist["Milestones Completed"] > 0].copy()
        if not milestone_dist.empty:
            milestone_dist["Percentage"] = milestone_dist["Students"] / total_students * 100
            render_bar(milestone_dist, "Milestones Completed", "Percentage", "Students by number of milestones completed")
            st.dataframe(make_display_safe(milestone_dist), use_container_width=True, height=260)
        else:
            st.info("No milestone completions above 0 found in the current filtered view.")
    else:
        st.info("No milestone columns found in the data.")

with drilldown_tab:
    st.markdown("### Drilldown")
    mode = st.radio("Choose drilldown type", ["College Name", "University Name"], horizontal=True)
    if mode in filtered_df.columns:
        options = sorted(filtered_df[mode].dropna().unique().tolist())
    else:
        options = []

    selected = st.selectbox(f"Select {mode}", options) if options else None
    if selected:
        subset = filtered_df[filtered_df[mode] == selected].copy()
        s_n = len(subset)

        d1, d2, d3, d4 = st.columns(4)
        with d1:
            styled_metric("Students", f"{s_n:,}")
        with d2:
            styled_metric("Active Students", format_pct(int(subset["Active Student"].sum()), s_n))
        with d3:
            styled_metric("Didn't Start", format_pct(int(subset["Did Not Start"].sum()), s_n))
        with d4:
            styled_metric("Active in Last 6 Months", format_pct(int(subset["Active in Last 6 Months"].sum()), s_n))

        a, b = st.columns(2)
        with a:
            track_df = top_share_table(subset, "Track Name", top_n=12)
            render_bar(track_df, "Track Name", "% of valid rows", f"Track mix in {selected}")
        with b:
            grade_small = top_share_table(subset, "Term 1 Final Grade", top_n=12)
            render_bar(grade_small, "Term 1 Final Grade", "% of valid rows", f"Term 1 Final Grade mix in {selected}")

        if "Last active date" in subset.columns:
            monthly = subset.dropna(subset=["Last active date"]).copy()
            if not monthly.empty:
                monthly["Activity Month"] = monthly["Last active date"].dt.to_period("M").astype(str)
                monthly_df = monthly.groupby("Activity Month").size().reset_index(name="Students")
                render_bar(monthly_df, "Activity Month", "Students", f"Recent activity trend in {selected}")

        st.markdown("### Student snapshot")
        available_display_cols = [c for c in DISPLAY_COLUMNS if c in subset.columns]
        st.dataframe(make_display_safe(subset[available_display_cols].head(1000)), use_container_width=True, height=420)

with explorer_tab:
    st.markdown("### Data Explorer")
    q = st.text_input("Search by name or email")
    explorer_df = filtered_df.copy()

    if q:
        pattern = q.strip().lower()
        name_match = explorer_df["Name"].fillna("").str.lower().str.contains(pattern, regex=False) if "Name" in explorer_df.columns else False
        email_match = explorer_df["Email ID"].fillna("").str.lower().str.contains(pattern, regex=False) if "Email ID" in explorer_df.columns else False
        explorer_df = explorer_df[name_match | email_match]

    qc1, qc2, qc3 = st.columns(3)
    with qc1:
        duplicate_emails = explorer_df["Email ID"].dropna().duplicated().sum() if "Email ID" in explorer_df.columns else 0
        styled_metric("Duplicate email rows", f"{int(duplicate_emails):,}")
    with qc2:
        styled_metric("Rows shown", f"{len(explorer_df):,}")
    with qc3:
        positive_term_rows = int(pd.to_numeric(explorer_df.get("Term", pd.Series(dtype=float)), errors="coerce").fillna(0).gt(0).sum()) if "Term" in explorer_df.columns else 0
        styled_metric("Rows with term > 0", f"{positive_term_rows:,}")

    completeness_rows = []
    for col in [
        "Email ID",
        "College Name",
        "University Name",
        "Faculty Name",
        "Track Name",
        "Date of activation",
        "Last active date",
        "Term 1 Final Grade",
    ]:
        if col in explorer_df.columns:
            if pd.api.types.is_datetime64_any_dtype(explorer_df[col]):
                present = int(explorer_df[col].notna().sum())
            else:
                present = int(non_empty_non_zero_mask(explorer_df[col]).sum())
            completeness_rows.append(
                {
                    "Column": col,
                    "Present rows": present,
                    "Completeness %": (present / len(explorer_df) * 100) if len(explorer_df) else 0,
                }
            )

    if completeness_rows:
        completeness_df = pd.DataFrame(completeness_rows).sort_values("Completeness %", ascending=False)
        render_bar(completeness_df, "Column", "Completeness %", "Column completeness (0 treated as empty where relevant)")

    available_display_cols = [c for c in DISPLAY_COLUMNS if c in explorer_df.columns]
    st.dataframe(make_display_safe(explorer_df[available_display_cols].head(5000)), use_container_width=True, height=460)
