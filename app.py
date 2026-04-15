import io
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Student Progress Dashboard",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling
# -----------------------------
CUSTOM_CSS = """
<style>
:root {
    --blue-900: #123a73;
    --blue-800: #1f4d8f;
    --blue-700: #2e63b3;
    --blue-200: #dbeafe;
    --blue-100: #eff6ff;
    --white: #ffffff;
    --slate: #5b677a;
}

.main {
    background: linear-gradient(180deg, #f8fbff 0%, #eef5ff 100%);
}

.block-container {
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}

h1, h2, h3 {
    color: var(--blue-900);
}

.dashboard-banner {
    background: linear-gradient(90deg, var(--blue-900), var(--blue-700));
    padding: 1.2rem 1.4rem;
    border-radius: 18px;
    color: white;
    margin-bottom: 1rem;
    box-shadow: 0 8px 24px rgba(18,58,115,0.12);
}

.metric-card {
    background: white;
    border: 1px solid #e6eefb;
    padding: 1rem;
    border-radius: 18px;
    box-shadow: 0 8px 24px rgba(31,77,143,0.08);
}

.small-note {
    color: #5b677a;
    font-size: 0.92rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f2f7ff 0%, #e7f0ff 100%);
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
# Helpers
# -----------------------------
def load_data(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file, low_memory=False)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(file)
    raise ValueError("Please upload a CSV or Excel file.")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    text_cols = [
        "Name", "Email ID", "College Name", "University Name", "Faculty Name",
        "Track Name", "Status", "Term 1 Final Grade", "Business Category", "Business Name"
    ]
    for col in text_cols:
        if col in out.columns:
            out[col] = out[col].astype(str).replace({"nan": np.nan, "None": np.nan}).str.strip()

    numeric_cols = [
        "Activities completed", "Tasks completed", "Term 1 Milestone Completed",
        "Term 2 Milestone Completed", "Term 3 Milestone Completed",
        "Term 4 Milestone Completed", "Term 5 Milestone Completed", "Term 6 Milestone Completed",
        "Term 7 Milestone Completed", "Term 8 Milestone Completed", "Term 9 Milestone Completed",
        "Term 10 Milestone Completed", "Points", "Badges", "Revenue", "No of sales",
        "No of offerings", "Final Activity Score", "Final Point Score", "Final Revenue Score",
        "Term 1 Final Score", "No of track tried"
    ]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    date_cols = ["Date of sign up start", "Student Created At", "Date of activation", "Last active date"]
    for col in date_cols:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if "Term" in out.columns:
        out["Term"] = out["Term"].astype(str).str.strip()

    return out


def pct(part: float, whole: float) -> float:
    if whole == 0:
        return 0.0
    return (part / whole) * 100.0


def percentage_table(series: pd.Series, top_n: Optional[int] = None, label: str = "Category") -> pd.DataFrame:
    clean = series.fillna("Unknown").replace("", "Unknown")
    counts = clean.value_counts(dropna=False)
    if top_n is not None:
        counts = counts.head(top_n)
    total = clean.shape[0]
    df_out = counts.rename_axis(label).reset_index(name="Students")
    df_out["Percentage"] = (df_out["Students"] / total * 100).round(2)
    return df_out


def plot_bar(df_plot: pd.DataFrame, x: str, y: str, title: str, color: Optional[str] = None):
    fig = px.bar(
        df_plot,
        x=x,
        y=y,
        color=color,
        text=y,
        title=title,
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        title_font_color="#123a73",
        xaxis_title=None,
        yaxis_title=None,
        margin=dict(l=10, r=10, t=50, b=10),
        height=430,
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
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


def build_summary_metrics(df: pd.DataFrame):
    total = len(df)

    term_cols = [c for c in df.columns if c.startswith("Term ") and c.endswith("Milestone Completed")]
    milestone_any = pd.Series(False, index=df.index)
    if term_cols:
        milestone_any = df[term_cols].fillna(0).gt(0).any(axis=1)

    activities = df["Activities completed"].fillna(0) if "Activities completed" in df.columns else pd.Series(0, index=df.index)
    tasks = df["Tasks completed"].fillna(0) if "Tasks completed" in df.columns else pd.Series(0, index=df.index)
    term1 = df["Term 1 Milestone Completed"].fillna(0) if "Term 1 Milestone Completed" in df.columns else pd.Series(0, index=df.index)
    term2 = df["Term 2 Milestone Completed"].fillna(0) if "Term 2 Milestone Completed" in df.columns else pd.Series(0, index=df.index)

    created = df["Student Created At"] if "Student Created At" in df.columns else pd.Series(pd.NaT, index=df.index)
    last_active = df["Last active date"] if "Last active date" in df.columns else pd.Series(pd.NaT, index=df.index)

    reference_date = last_active.max()
    if pd.isna(reference_date):
        reference_date = pd.Timestamp.today().normalize()
    six_month_cutoff = reference_date - pd.Timedelta(days=180)

    did_not_start = (activities <= 0) & (tasks <= 0) & (~milestone_any)
    active_last_6m = last_active.notna() & (last_active >= six_month_cutoff)
    term1_not_term2 = (term1 > 0) & (term2 <= 0)
    inactive_since_term1 = term1_not_term2 & (~active_last_6m)

    metrics = {
        "Total Students": total,
        "% Completed Milestones": round(pct(milestone_any.sum(), total), 2),
        "% Didn’t Start": round(pct(did_not_start.sum(), total), 2),
        "% Active in Last 6 Months": round(pct(active_last_6m.sum(), total), 2),
        "% Did Term 1 but Not Active in Term 2": round(pct(inactive_since_term1.sum(), total), 2),
        "Reference Date for Activity": reference_date.strftime("%d %b %Y") if not pd.isna(reference_date) else "N/A",
    }

    flags = pd.DataFrame({
        "Completed Milestones": milestone_any,
        "Did Not Start": did_not_start,
        "Active Last 6 Months": active_last_6m,
        "Did Term 1, Not Active in Term 2": inactive_since_term1,
    })

    return metrics, flags, six_month_cutoff




def make_safe_mask(series: pd.Series, value: str) -> pd.Series:
    normalized = series.fillna("Unknown").astype(str).replace({"": "Unknown"}).str.strip()
    return normalized == value


def render_drilldown_section(df_source: pd.DataFrame, entity_col: str, section_title: str):
    if entity_col not in df_source.columns:
        st.info(f"{entity_col} column not found.")
        return

    clean_entities = (
        df_source[entity_col]
        .fillna("Unknown")
        .astype(str)
        .replace({"": "Unknown"})
        .str.strip()
    )
    entity_options = sorted(clean_entities.unique().tolist())
    if not entity_options:
        st.info(f"No values available for {entity_col}.")
        return

    selected_entity = st.selectbox(
        f"Search and select {section_title}",
        options=entity_options,
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

    st.caption(f"Drilldown activity cutoff: {entity_cutoff.strftime('%d %b %Y')}")

    top_left, top_right = st.columns(2)
    with top_left:
        if "Track Name" in entity_df.columns:
            track_tbl = percentage_table(entity_df["Track Name"], label="Track Name")
            st.plotly_chart(
                plot_bar(track_tbl.head(15), "Track Name", "Percentage", f"Track Mix in {selected_entity}"),
                use_container_width=True,
            )
        elif "College Name" in entity_df.columns and entity_col != "College Name":
            college_tbl = percentage_table(entity_df["College Name"], label="College Name")
            st.plotly_chart(
                plot_bar(college_tbl.head(15), "College Name", "Percentage", f"College Mix in {selected_entity}"),
                use_container_width=True,
            )

    with top_right:
        if "Term 1 Final Grade" in entity_df.columns:
            grade_tbl = percentage_table(entity_df["Term 1 Final Grade"], label="Term 1 Final Grade")
            st.plotly_chart(
                plot_bar(grade_tbl.head(15), "Term 1 Final Grade", "Percentage", f"Term 1 Final Grade in {selected_entity}"),
                use_container_width=True,
            )
        elif "Status" in entity_df.columns:
            status_tbl = percentage_table(entity_df["Status"], label="Status")
            st.plotly_chart(
                plot_pie(status_tbl, "Status", "Students", f"Status Distribution in {selected_entity}"),
                use_container_width=True,
            )

    lower_left, lower_right = st.columns(2)
    with lower_left:
        if "Faculty Name" in entity_df.columns:
            faculty_na = entity_df["Faculty Name"].isna() | entity_df["Faculty Name"].astype(str).str.strip().isin(["", "N/A", "n/a", "NA", "nan"])
            faculty_summary = pd.DataFrame({
                "Faculty Assignment": ["Assigned", "Not Assigned"],
                "Percentage": [round((~faculty_na).mean() * 100, 2), round(faculty_na.mean() * 100, 2)],
            })
            st.plotly_chart(
                plot_bar(faculty_summary, "Faculty Assignment", "Percentage", f"Faculty Assignment in {selected_entity}"),
                use_container_width=True,
            )

    with lower_right:
        if "Last active date" in entity_df.columns:
            monthly = (
                entity_df.dropna(subset=["Last active date"])
                .assign(ActivityMonth=entity_df.dropna(subset=["Last active date"])["Last active date"].dt.to_period("M").astype(str))
                .groupby("ActivityMonth")
                .size()
                .reset_index(name="Students")
            )
            if not monthly.empty:
                fig = px.line(monthly, x="ActivityMonth", y="Students", markers=True, title=f"Monthly Activity Trend in {selected_entity}")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", title_font_color="#123a73", height=430)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Snapshot table")
    snapshot_cols = [c for c in ["Name", "Email ID", "College Name", "University Name", "Track Name", "Faculty Name", "Term", "Term 1 Final Grade", "Activities completed", "Tasks completed", "Last active date"] if c in entity_df.columns]
    st.dataframe(entity_df[snapshot_cols].head(250), use_container_width=True, hide_index=True)


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


# -----------------------------
# App header
# -----------------------------
st.markdown(
    """
    <div class='dashboard-banner'>
        <div style='font-size:2rem;font-weight:800;'>Student Progress Dashboard</div>
        <div style='opacity:0.95;margin-top:0.25rem;'>Light blue, dark blue, and white Streamlit dashboard for milestone, activity, faculty assignment, college, university, track, and grade analysis.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.header("Upload & Filters")
uploaded = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx", "xls"],
)

sample_path = "students.csv"
use_sample = False
if uploaded is None:
    if st.sidebar.button("Use current sample file"):
        use_sample = True
    st.info("Upload your file from the sidebar, or click **Use current sample file** to load the attached dataset.")

if uploaded is None and not use_sample:
    st.stop()

try:
    if uploaded is not None:
        raw_df = load_data(uploaded)
        source_name = uploaded.name
    else:
        raw_df = pd.read_csv(sample_path, low_memory=False)
        source_name = sample_path
except Exception as exc:
    st.error(f"Could not read the file: {exc}")
    st.stop()


df = clean_dataframe(raw_df)

# Sidebar filters
if "University Name" in df.columns:
    uni_options = sorted(df["University Name"].dropna().astype(str).unique().tolist())
    selected_unis = st.sidebar.multiselect("University Name", uni_options)
    if selected_unis:
        df = df[df["University Name"].isin(selected_unis)]

if "College Name" in df.columns:
    college_options = sorted(df["College Name"].dropna().astype(str).unique().tolist())
    selected_colleges = st.sidebar.multiselect("College Name", college_options)
    if selected_colleges:
        df = df[df["College Name"].isin(selected_colleges)]

if "Track Name" in df.columns:
    track_options = sorted(df["Track Name"].dropna().astype(str).unique().tolist())
    selected_tracks = st.sidebar.multiselect("Track Name", track_options)
    if selected_tracks:
        df = df[df["Track Name"].isin(selected_tracks)]

if "Status" in df.columns:
    status_options = sorted(df["Status"].dropna().astype(str).unique().tolist())
    selected_status = st.sidebar.multiselect("Status", status_options)
    if selected_status:
        df = df[df["Status"].isin(selected_status)]

if "Term" in df.columns:
    term_options = sorted(df["Term"].dropna().astype(str).unique().tolist())
    selected_terms = st.sidebar.multiselect("Term", term_options)
    if selected_terms:
        df = df[df["Term"].isin(selected_terms)]

st.sidebar.markdown("---")
st.sidebar.caption(f"Rows after filters: {len(df):,}")
st.sidebar.caption(f"Source: {source_name}")

metrics, flags, six_month_cutoff = build_summary_metrics(df)

overview_tab, institutions_tab, engagement_tab, faculty_tab, grades_tab, drilldown_tab, data_tab = st.tabs([
    "Overview",
    "Institutions",
    "Engagement",
    "Faculty",
    "Grades & Terms",
    "Drilldown",
    "Data Explorer",
])

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
        render_metric("Did Term 1, Not Active in Term 2", f"{metrics['% Did Term 1 but Not Active in Term 2']:.2f}%")

    st.markdown(
        f"<div class='small-note'>Activity recency is measured against the latest <b>Last active date</b> in the filtered data. Current cutoff: <b>{six_month_cutoff.strftime('%d %b %Y')}</b>.</div>",
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        summary_df = pd.DataFrame({
            "Metric": [
                "Completed Milestones",
                "Didn’t Start",
                "Active in Last 6 Months",
                "Did Term 1, Not Active in Term 2",
            ],
            "Percentage": [
                metrics["% Completed Milestones"],
                metrics["% Didn’t Start"],
                metrics["% Active in Last 6 Months"],
                metrics["% Did Term 1 but Not Active in Term 2"],
            ],
        })
        st.plotly_chart(plot_bar(summary_df, "Metric", "Percentage", "Key Student Progress Percentages"), use_container_width=True)
    with col_b:
        if "Status" in df.columns:
            status_tbl = percentage_table(df["Status"], label="Status")
            st.plotly_chart(plot_pie(status_tbl, "Status", "Students", "Status Distribution"), use_container_width=True)
        else:
            st.info("Status column not found.")

    st.subheader("Quick insights")
    insight_cols = st.columns(3)
    with insight_cols[0]:
        if "College Name" in df.columns:
            college_tbl = percentage_table(df["College Name"], top_n=10, label="College Name")
            st.dataframe(college_tbl, use_container_width=True, hide_index=True)
    with insight_cols[1]:
        if "University Name" in df.columns:
            uni_tbl = percentage_table(df["University Name"], label="University Name")
            st.dataframe(uni_tbl, use_container_width=True, hide_index=True)
    with insight_cols[2]:
        if "Track Name" in df.columns:
            track_tbl = percentage_table(df["Track Name"], label="Track Name")
            st.dataframe(track_tbl, use_container_width=True, hide_index=True)

with institutions_tab:
    left, right = st.columns(2)
    if "College Name" in df.columns:
        college_tbl = percentage_table(df["College Name"], top_n=20, label="College Name")
        with left:
            st.plotly_chart(plot_bar(college_tbl, "College Name", "Percentage", "% of Students in Each College Name"), use_container_width=True)
            st.dataframe(college_tbl, use_container_width=True, hide_index=True)

    if "University Name" in df.columns:
        uni_tbl = percentage_table(df["University Name"], label="University Name")
        with right:
            st.plotly_chart(plot_bar(uni_tbl, "University Name", "Percentage", "% of Students in Each University Name"), use_container_width=True)
            st.dataframe(uni_tbl, use_container_width=True, hide_index=True)

    st.subheader("Track distribution")
    if "Track Name" in df.columns:
        track_tbl = percentage_table(df["Track Name"], label="Track Name")
        a, b = st.columns([1.2, 1])
        with a:
            st.plotly_chart(plot_bar(track_tbl, "Track Name", "Percentage", "% of Students in Each Track Name"), use_container_width=True)
        with b:
            st.dataframe(track_tbl, use_container_width=True, hide_index=True)

with engagement_tab:
    st.subheader("Activity and milestone analysis")
    e1, e2 = st.columns(2)
    with e1:
        if "Activities completed" in df.columns and "Tasks completed" in df.columns:
            bucket_df = pd.DataFrame({
                "Bucket": [
                    "No start",
                    "Low activity (1-10)",
                    "Moderate activity (11-50)",
                    "High activity (51+)"
                ],
                "Students": [
                    int((df["Activities completed"].fillna(0) == 0).sum()),
                    int(df["Activities completed"].fillna(0).between(1, 10).sum()),
                    int(df["Activities completed"].fillna(0).between(11, 50).sum()),
                    int((df["Activities completed"].fillna(0) > 50).sum()),
                ]
            })
            bucket_df["Percentage"] = (bucket_df["Students"] / len(df) * 100).round(2)
            st.plotly_chart(plot_bar(bucket_df, "Bucket", "Percentage", "Activity Intensity Split"), use_container_width=True)

    with e2:
        timeline_col = "Last active date" if "Last active date" in df.columns else None
        if timeline_col:
            timeline = (
                df.dropna(subset=[timeline_col])
                .groupby(df[timeline_col].dt.to_period("M").astype(str))
                .size()
                .reset_index(name="Students")
                .rename(columns={timeline_col: "Month", 0: "Students"})
            )
            if len(timeline) > 0:
                fig = px.line(timeline, x=timeline.columns[0], y="Students", markers=True, title="Students Active by Month")
                fig.update_layout(paper_bgcolor="white", plot_bgcolor="white", title_font_color="#123a73", height=430)
                st.plotly_chart(fig, use_container_width=True)

    st.subheader("Milestone progression by term")
    term_milestone_cols = [c for c in df.columns if c.startswith("Term ") and c.endswith("Milestone Completed")]
    if term_milestone_cols:
        milestone_pct = []
        for col in term_milestone_cols:
            milestone_pct.append({
                "Term Milestone": col.replace(" Milestone Completed", ""),
                "Percentage": round((df[col].fillna(0) > 0).mean() * 100, 2)
            })
        milestone_pct_df = pd.DataFrame(milestone_pct)
        st.plotly_chart(plot_bar(milestone_pct_df, "Term Milestone", "Percentage", "Students Completing Each Term Milestone"), use_container_width=True)
        st.dataframe(milestone_pct_df, use_container_width=True, hide_index=True)

with faculty_tab:
    st.subheader("Faculty assignment coverage")
    if "College Name" in df.columns and "Faculty Name" in df.columns:
        faculty_na = df["Faculty Name"].isna() | df["Faculty Name"].astype(str).str.strip().isin(["", "N/A", "n/a", "NA", "nan"])
        faculty_gap = (
            df.assign(FacultyMissing=faculty_na)
            .groupby("College Name")
            .agg(Total_Students=("College Name", "size"), Faculty_Not_Assigned=("FacultyMissing", "sum"))
            .reset_index()
        )
        faculty_gap["Percentage"] = (faculty_gap["Faculty_Not_Assigned"] / faculty_gap["Total_Students"] * 100).round(2)
        faculty_gap = faculty_gap.sort_values(["Percentage", "Faculty_Not_Assigned"], ascending=[False, False])

        f1, f2 = st.columns([1.35, 1])
        with f1:
            st.plotly_chart(
                plot_bar(faculty_gap.head(20), "College Name", "Percentage", "% of Students in Each College Where Faculty is N/A"),
                use_container_width=True,
            )
        with f2:
            st.dataframe(faculty_gap, use_container_width=True, hide_index=True)

with grades_tab:
    st.subheader("Term 1 final grade by college")
    if "College Name" in df.columns and "Term 1 Final Grade" in df.columns:
        grade_heat = (
            df.assign(**{"Term 1 Final Grade": df["Term 1 Final Grade"].fillna("Unknown")})
            .groupby(["College Name", "Term 1 Final Grade"])
            .size()
            .reset_index(name="Students")
        )
        totals = grade_heat.groupby("College Name")["Students"].transform("sum")
        grade_heat["Percentage"] = (grade_heat["Students"] / totals * 100).round(2)

        top_colleges = df["College Name"].value_counts().head(20).index.tolist()
        heat_top = grade_heat[grade_heat["College Name"].isin(top_colleges)]
        pivot = heat_top.pivot(index="College Name", columns="Term 1 Final Grade", values="Percentage").fillna(0)

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

        st.dataframe(
            grade_heat.sort_values(["College Name", "Percentage"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Term-level comparison")
    if "Term" in df.columns:
        term_dist = percentage_table(df["Term"], label="Term")
        g1, g2 = st.columns([1.1, 1])
        with g1:
            st.plotly_chart(plot_bar(term_dist, "Term", "Percentage", "Student Share by Term"), use_container_width=True)
        with g2:
            st.dataframe(term_dist, use_container_width=True, hide_index=True)


with drilldown_tab:
    st.subheader("Searchable college and university drilldown")
    st.caption("Use this when the full institution charts are too crowded. Pick one college or one university to see a focused breakdown.")

    drill_mode = st.radio("Drill down by", ["College Name", "University Name"], horizontal=True)

    if drill_mode == "College Name":
        render_drilldown_section(df, "College Name", "College")
    else:
        render_drilldown_section(df, "University Name", "University")

with data_tab:
    st.subheader("Data explorer")
    st.dataframe(df, use_container_width=True, height=420)

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download filtered data as CSV",
        data=csv_buffer.getvalue(),
        file_name="filtered_students_data.csv",
        mime="text/csv",
    )

    st.markdown("### Column completeness")
    completeness = pd.DataFrame({
        "Column": df.columns,
        "Non-null %": ((df.notna().sum() / len(df)) * 100).round(2),
        "Missing %": ((df.isna().sum() / len(df)) * 100).round(2),
    }).sort_values("Missing %", ascending=False)
    st.dataframe(completeness, use_container_width=True, hide_index=True)

with st.expander("Metric definitions used in this dashboard"):
    st.markdown(
        """
        - **Completed milestones**: a student has a value greater than 0 in at least one `Term X Milestone Completed` column.
        - **Didn’t start**: `Activities completed = 0`, `Tasks completed = 0`, and no term milestone completed.
        - **Active in last 6 months**: `Last active date` falls within 180 days of the latest last-active date in the filtered dataset.
        - **Did semester/term 1 but not active in 2**: `Term 1 Milestone Completed > 0`, `Term 2 Milestone Completed = 0`, and not active in the last 6 months.
        - **Faculty not assigned**: `Faculty Name` is blank or marked as `N/A`, `NA`, or similar.
        """
    )
