# ============================================================
# USDA DIGITAL PATHWAY & AI INSIGHTS SUITE
# app.py
# ============================================================

from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="USDA Digital Pathway & AI Insights Suite",
    layout="wide"
)

st.title("USDA Digital Pathway & AI Insights Suite")
st.caption(
    "Executive dashboard for USDA 2025 web analytics, Rural Development page behavior, "
    "and AI-enabled service prioritization."
)


# ============================================================
# SETTINGS
# ============================================================

DATA_FILE = Path("organized_clean_long_data_full_USDA.xlsx")
DATA_FOLDER = Path("organized_clean_long_data_full_USDA")

ALLOWED_DEVICES = ["desktop", "mobile", "tablet", "smart tv"]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
        .str.replace("/", "_", regex=False)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
    )
    return df


def find_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        for key in keywords:
            if key in col_lower:
                return col
    return None


def convert_numeric_columns(df):
    df = df.copy()

    for col in df.columns:
        if col not in ["source_file", "source_sheet"]:
            converted = pd.to_numeric(df[col], errors="coerce")

            if converted.notna().sum() >= len(df) * 0.25:
                df[col] = converted

    return df


def format_number(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value:,.0f}"
    except Exception:
        return "N/A"


def load_data():
    loaded = []
    inventory = []

    possible_files = []

    if DATA_FILE.exists():
        possible_files.append(DATA_FILE)

    if DATA_FOLDER.exists() and DATA_FOLDER.is_dir():
        possible_files.extend(list(DATA_FOLDER.glob("*.xlsx")))
        possible_files.extend(list(DATA_FOLDER.glob("*.xls")))
        possible_files.extend(list(DATA_FOLDER.glob("*.csv")))

    if not possible_files:
        st.error(
            "No USDA data file was found. Upload `organized_clean_long_data_full_USDA.xlsx` "
            "or place files inside the `organized_clean_long_data_full_USDA` folder."
        )
        st.stop()

    for file in possible_files:
        try:
            if file.suffix.lower() == ".csv":
                temp = pd.read_csv(file)
                temp = normalize_columns(temp)
                temp["source_file"] = file.name
                loaded.append(temp)

                inventory.append({
                    "file": file.name,
                    "sheet": "N/A",
                    "rows": temp.shape[0],
                    "columns": temp.shape[1],
                    "status": "Loaded"
                })

            elif file.suffix.lower() in [".xlsx", ".xls"]:
                excel = pd.ExcelFile(file)

                for sheet in excel.sheet_names:
                    temp = pd.read_excel(file, sheet_name=sheet)
                    temp = normalize_columns(temp)
                    temp["source_file"] = file.name
                    temp["source_sheet"] = sheet
                    loaded.append(temp)

                    inventory.append({
                        "file": file.name,
                        "sheet": sheet,
                        "rows": temp.shape[0],
                        "columns": temp.shape[1],
                        "status": "Loaded"
                    })

        except Exception as e:
            inventory.append({
                "file": file.name,
                "sheet": "Unknown",
                "rows": 0,
                "columns": 0,
                "status": f"Error: {e}"
            })

    if not loaded:
        st.error("Data files were detected, but none could be loaded successfully.")
        st.stop()

    df = pd.concat(loaded, ignore_index=True, sort=False)
    return df, pd.DataFrame(inventory)


@st.cache_data
def cached_load_data():
    return load_data()


def build_insights(df, page_col, users_col, sessions_col):
    insights = []

    if df.empty:
        return ["No rows are available after filtering."]

    if page_col and users_col and page_col in df.columns and users_col in df.columns:
        page_summary = (
            df.groupby(page_col, dropna=False)[users_col]
            .sum()
            .sort_values(ascending=False)
        )

        if not page_summary.empty and page_summary.sum() > 0:
            top_page = page_summary.index[0]
            top_value = page_summary.iloc[0]
            share = top_value / page_summary.sum() * 100

            insights.append(
                f"**{top_page}** is the highest-activity page, representing about "
                f"**{share:.1f}%** of filtered user activity."
            )

    if sessions_col and sessions_col in df.columns:
        insights.append(
            f"The filtered view contains **{format_number(df[sessions_col].sum())} sessions**, "
            "which indicates the scale of digital demand."
        )

    if page_col and page_col in df.columns:
        insights.append(
            f"The current view includes **{df[page_col].nunique():,} unique page titles**, "
            "useful for identifying where users may need clearer navigation."
        )

    return insights


# ============================================================
# LOAD DATA
# ============================================================

df, inventory_df = cached_load_data()
df = convert_numeric_columns(df)


# ============================================================
# COLUMN DETECTION
# ============================================================

page_col = find_col(df, ["page_title", "page", "title"])

# Required date column from the project file:
date_col = find_col(df, ["traffic_date_2025_assumed"])

if not date_col:
    date_col = find_col(df, ["traffic_date", "date"])

country_col = find_col(df, ["country"])
device_col = find_col(df, ["device"])
channel_col = find_col(df, ["channel", "source", "medium"])

users_col = find_col(df, ["active_users", "total_users", "users", "visitors"])
sessions_col = find_col(df, ["sessions", "visits"])
events_col = find_col(df, ["event_count", "events"])
views_col = find_col(df, ["views_per_session", "views"])
duration_col = find_col(df, ["average_session_duration", "avg_session_duration", "duration"])
bounce_col = find_col(df, ["bounce_rate", "bounce"])
exit_col = find_col(df, ["exits", "exit"])
returning_col = find_col(df, ["returning_users", "returning"])


# ============================================================
# DATE CLEANING: TRAFFIC DATE (2025 ASSUMED)
# ============================================================

if date_col and date_col in df.columns:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Executive Filters")

filtered_df = df.copy()

if date_col and date_col in filtered_df.columns and filtered_df[date_col].notna().sum() > 0:
    min_date = filtered_df[date_col].min().date()
    max_date = filtered_df[date_col].max().date()

    selected_dates = st.sidebar.date_input(
        "Traffic Date (2025 Assumed)",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        start_date, end_date = selected_dates

        filtered_df = filtered_df[
            (filtered_df[date_col].dt.date >= start_date) &
            (filtered_df[date_col].dt.date <= end_date)
        ]

if device_col and device_col in filtered_df.columns:
    filtered_df[device_col] = (
        filtered_df[device_col]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    filtered_df = filtered_df[
        filtered_df[device_col].isin(ALLOWED_DEVICES)
    ]

    selected_devices = st.sidebar.multiselect(
        "Device Category",
        sorted(filtered_df[device_col].dropna().unique()),
        default=[]
    )

    if selected_devices:
        filtered_df = filtered_df[
            filtered_df[device_col].isin(selected_devices)
        ]

if country_col and country_col in filtered_df.columns:
    selected_countries = st.sidebar.multiselect(
        "Country",
        sorted(filtered_df[country_col].dropna().astype(str).unique()),
        default=[]
    )

    if selected_countries:
        filtered_df = filtered_df[
            filtered_df[country_col].astype(str).isin(selected_countries)
        ]

if channel_col and channel_col in filtered_df.columns:
    selected_channels = st.sidebar.multiselect(
        "Channel / Source",
        sorted(filtered_df[channel_col].dropna().astype(str).unique()),
        default=[]
    )

    if selected_channels:
        filtered_df = filtered_df[
            filtered_df[channel_col].astype(str).isin(selected_channels)
        ]

page_search = st.sidebar.text_input("Search Page Title")

if page_col and page_search:
    filtered_df = filtered_df[
        filtered_df[page_col]
        .astype(str)
        .str.lower()
        .str.contains(page_search.lower(), na=False)
    ]

top_n = st.sidebar.slider("Top N Pages", 5, 50, 15)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "Executive Overview",
    "USDA Web Usage",
    "Rural Development Segments",
    "Data Quality / Methodology"
])


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tab1:
    st.header("Executive Overview")

    st.write(
        "This view highlights the most decision-relevant indicators for USDA executives: "
        "traffic scale, user demand, Rural Development share, top pages, and digital service "
        "opportunities for AI-enabled support."
    )

    total_users = filtered_df[users_col].sum() if users_col and users_col in filtered_df.columns else np.nan
    total_sessions = filtered_df[sessions_col].sum() if sessions_col and sessions_col in filtered_df.columns else np.nan
    total_events = filtered_df[events_col].sum() if events_col and events_col in filtered_df.columns else np.nan
    unique_pages = filtered_df[page_col].nunique() if page_col and page_col in filtered_df.columns else np.nan

    if page_col and page_col in filtered_df.columns:
        rd_df = filtered_df[
            filtered_df[page_col]
            .astype(str)
            .str.lower()
            .str.contains("rural development", na=False)
        ].copy()
    else:
        rd_df = pd.DataFrame()

    rd_users = rd_df[users_col].sum() if users_col and users_col in rd_df.columns else np.nan
    rd_share = rd_users / total_users * 100 if pd.notna(total_users) and total_users > 0 else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Users", format_number(total_users))
    c2.metric("Total Sessions", format_number(total_sessions))
    c3.metric("Total Events", format_number(total_events))
    c4.metric("Unique Pages", format_number(unique_pages))
    c5.metric("Rural Dev Share", f"{rd_share:.1f}%" if pd.notna(rd_share) else "N/A")

    st.subheader("Executive Insights")

    for insight in build_insights(filtered_df, page_col, users_col, sessions_col):
        st.info(insight)

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df.head(500), use_container_width=True)

    st.download_button(
        "Download Filtered Data",
        filtered_df.to_csv(index=False).encode("utf-8"),
        "filtered_usda_web_analytics.csv",
        "text/csv"
    )


# ============================================================
# TAB 2: USDA WEB USAGE
# ============================================================

with tab2:
    st.header("USDA Web Usage")

    if filtered_df.empty:
        st.warning("No data available after filtering.")

    else:
        if date_col and users_col and date_col in filtered_df.columns:
            st.subheader("Traffic Trend by Traffic Date (2025 Assumed)")

            trend_df = (
                filtered_df.dropna(subset=[date_col])
                .groupby(date_col)[users_col]
                .sum()
                .reset_index()
                .sort_values(date_col)
            )

            if not trend_df.empty:
                fig = px.line(
                    trend_df,
                    x=date_col,
                    y=users_col,
                    markers=True,
                    title="USDA User Activity Over Time",
                    labels={
                        date_col: "Traffic Date (2025 Assumed)",
                        users_col: "Users"
                    }
                )

                st.plotly_chart(fig, use_container_width=True)

        if page_col and users_col and page_col in filtered_df.columns:
            st.subheader("Top Pages by User Activity")

            top_pages = (
                filtered_df.groupby(page_col, dropna=False)[users_col]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            fig = px.bar(
                top_pages,
                x=users_col,
                y=page_col,
                orientation="h",
                title="Top USDA Pages by Users",
                labels={
                    users_col: "Users",
                    page_col: "Page Title"
                }
            )

            fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                height=650
            )

            st.plotly_chart(fig, use_container_width=True)

        if device_col and users_col and device_col in filtered_df.columns:
            st.subheader("Device-Level Activity")

            device_df = (
                filtered_df.groupby(device_col)[users_col]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig = px.pie(
                device_df,
                names=device_col,
                values=users_col,
                title="User Activity by Device Category",
                hole=0.35
            )

            fig.update_traces(
                textinfo="percent+label",
                hovertemplate="<b>%{label}</b><br>Users: %{value:,.0f}<br>Share: %{percent}<extra></extra>"
            )

            st.plotly_chart(fig, use_container_width=True)

        if country_col and users_col and country_col in filtered_df.columns:
            st.subheader("Top Countries by User Activity")

            country_df = (
                filtered_df.groupby(country_col)[users_col]
                .sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )

            fig = px.bar(
                country_df,
                x=country_col,
                y=users_col,
                title="Top Countries by Users",
                labels={
                    country_col: "Country",
                    users_col: "Users"
                }
            )

            st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3: RURAL DEVELOPMENT SEGMENTS
# ============================================================

with tab3:
    st.header("Rural Development Segments")

    if not page_col or page_col not in filtered_df.columns:
        st.warning("Page title column was not detected.")

    else:
        rd_df = filtered_df[
            filtered_df[page_col]
            .astype(str)
            .str.lower()
            .str.contains("rural development", na=False)
        ].copy()

        if rd_df.empty:
            st.warning(
                "No page titles containing 'rural development' were found after filters."
            )

        else:
            st.success(
                f"{len(rd_df):,} rows matched page titles containing 'rural development'."
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Rural Development Users",
                format_number(rd_df[users_col].sum()) if users_col and users_col in rd_df.columns else "N/A"
            )

            c2.metric(
                "Rural Development Sessions",
                format_number(rd_df[sessions_col].sum()) if sessions_col and sessions_col in rd_df.columns else "N/A"
            )

            c3.metric(
                "Rural Development Pages",
                f"{rd_df[page_col].nunique():,}"
            )

            segment_metrics = []

            for col in [
                users_col,
                sessions_col,
                events_col,
                views_col,
                duration_col,
                bounce_col,
                exit_col,
                returning_col
            ]:
                if col and col in rd_df.columns:
                    rd_df[col] = pd.to_numeric(rd_df[col], errors="coerce")
                    if rd_df[col].notna().sum() > 0:
                        segment_metrics.append(col)

            segment_metrics = list(dict.fromkeys(segment_metrics))

            if len(segment_metrics) < 2:
                st.warning("At least two numeric metrics are needed for clustering.")

            else:
                selected_metrics = st.multiselect(
                    "Metrics for Segmentation",
                    segment_metrics,
                    default=segment_metrics[:min(6, len(segment_metrics))]
                )

                if len(selected_metrics) < 2:
                    st.warning("Select at least two metrics.")

                else:
                    rd_page_segments = (
                        rd_df.groupby(page_col, dropna=False)[selected_metrics]
                        .sum()
                        .reset_index()
                    )

                    if users_col and sessions_col and users_col in rd_page_segments.columns and sessions_col in rd_page_segments.columns:
                        rd_page_segments["sessions_per_user"] = (
                            rd_page_segments[sessions_col] /
                            rd_page_segments[users_col].replace(0, np.nan)
                        ).fillna(0)

                    if users_col and users_col in rd_page_segments.columns:
                        total_rd_users = rd_page_segments[users_col].sum()
                        rd_page_segments["share_of_rd_users"] = (
                            rd_page_segments[users_col] / total_rd_users
                            if total_rd_users > 0 else 0
                        )

                    clustering_cols = [
                        col for col in rd_page_segments.columns
                        if col != page_col and pd.api.types.is_numeric_dtype(rd_page_segments[col])
                    ]

                    if len(rd_page_segments) < 2:
                        st.warning("Not enough Rural Development pages for clustering after filters.")

                    elif len(clustering_cols) < 2:
                        st.warning("Not enough usable numeric columns for clustering.")

                    else:
                        max_clusters = min(8, len(rd_page_segments))

                        k = st.slider(
                            "Number of Segments",
                            min_value=2,
                            max_value=max_clusters,
                            value=min(4, max_clusters)
                        )

                        X = (
                            rd_page_segments[clustering_cols]
                            .replace([np.inf, -np.inf], np.nan)
                            .fillna(0)
                        )

                        X_transformed = X.copy()

                        for col in X_transformed.columns:
                            if X_transformed[col].min() >= 0 and X_transformed[col].max() > 10:
                                X_transformed[col] = np.log1p(X_transformed[col])

                        try:
                            scaler = StandardScaler()
                            X_scaled = scaler.fit_transform(X_transformed)

                            kmeans = KMeans(
                                n_clusters=min(k, len(rd_page_segments)),
                                random_state=42,
                                n_init=20
                            )

                            rd_page_segments["segment"] = kmeans.fit_predict(X_scaled)

                            if users_col and users_col in rd_page_segments.columns:
                                segment_order = (
                                    rd_page_segments.groupby("segment")[users_col]
                                    .sum()
                                    .sort_values(ascending=False)
                                    .index
                                    .tolist()
                                )
                            else:
                                segment_order = (
                                    rd_page_segments["segment"]
                                    .value_counts()
                                    .index
                                    .tolist()
                                )

                            segment_labels = [
                                "High-Priority Digital Access Pages",
                                "Broad Navigation / Utility Pages",
                                "Niche Service Information Pages",
                                "Lower-Traffic Improvement Candidates",
                                "High-Exit Attention Pages",
                                "Repeat-Visitor Support Pages",
                                "Emerging Demand Pages",
                                "Specialized Program Pages"
                            ]

                            label_map = {
                                seg: segment_labels[i]
                                for i, seg in enumerate(segment_order)
                            }

                            rd_page_segments["segment_label"] = rd_page_segments["segment"].map(label_map)

                            st.subheader("Segment Opportunity Map")

                            x_metric = sessions_col if sessions_col in rd_page_segments.columns else clustering_cols[0]
                            y_metric = views_col if views_col in rd_page_segments.columns else clustering_cols[1]
                            size_metric = users_col if users_col in rd_page_segments.columns else clustering_cols[0]

                            fig = px.scatter(
                                rd_page_segments,
                                x=x_metric,
                                y=y_metric,
                                size=size_metric,
                                color="segment_label",
                                hover_name=page_col,
                                hover_data=clustering_cols[:8],
                                title="Rural Development Segments: Traffic vs Engagement",
                                labels={
                                    x_metric: x_metric.replace("_", " ").title(),
                                    y_metric: y_metric.replace("_", " ").title(),
                                    size_metric: size_metric.replace("_", " ").title(),
                                    "segment_label": "Segment"
                                }
                            )

                            fig.update_layout(height=650)
                            st.plotly_chart(fig, use_container_width=True)

                            st.subheader("Executive Segment Profiles")

                            segment_profile = (
                                rd_page_segments
                                .groupby("segment_label")
                                .agg(
                                    pages=(page_col, "count"),
                                    total_activity=(size_metric, "sum")
                                )
                                .reset_index()
                                .sort_values("total_activity", ascending=False)
                            )

                            st.dataframe(segment_profile, use_container_width=True)

                            st.subheader("AI-Enabled Service Prioritization")

                            recommendations = []

                            for _, row in segment_profile.iterrows():
                                segment = row["segment_label"]

                                if "High-Priority" in segment:
                                    rec = "Prioritize chatbot FAQs, guided navigation, and high-volume self-service support."
                                elif "Navigation" in segment:
                                    rec = "Improve menu pathways, search labels, and related-service links."
                                elif "High-Exit" in segment:
                                    rec = "Review exits and add clearer next-step prompts."
                                elif "Repeat" in segment:
                                    rec = "Add task-oriented support for users returning to complete actions."
                                else:
                                    rec = "Monitor content clarity and evaluate whether service instructions are easy to follow."

                                recommendations.append({
                                    "Segment": segment,
                                    "Pages": row["pages"],
                                    "Total Activity": row["total_activity"],
                                    "Executive Recommendation": rec
                                })

                            st.dataframe(pd.DataFrame(recommendations), use_container_width=True)

                            st.subheader("Segmented Rural Development Pages")

                            display_cols = [page_col, "segment_label"] + clustering_cols

                            st.dataframe(
                                rd_page_segments[display_cols].sort_values(
                                    by=size_metric,
                                    ascending=False
                                ),
                                use_container_width=True
                            )

                            st.download_button(
                                "Download Rural Development Segmentation",
                                rd_page_segments.to_csv(index=False).encode("utf-8"),
                                "rural_development_segmentation.csv",
                                "text/csv"
                            )

                        except Exception as e:
                            st.warning("Clustering could not be completed with the current filters.")
                            st.caption(f"Technical note: {e}")


# ============================================================
# TAB 4: DATA QUALITY / METHODOLOGY
# ============================================================

with tab4:
    st.header("Data Quality / Methodology")

    st.subheader("Data Inventory")
    st.dataframe(inventory_df, use_container_width=True)

    st.subheader("Detected Important Columns")

    detected_df = pd.DataFrame({
        "Feature Needed for Executive Dashboard": [
            "Traffic Date (2025 Assumed)",
            "Page Title",
            "Device Category",
            "Country",
            "Channel / Source",
            "Users",
            "Sessions",
            "Event Count",
            "Views per Session",
            "Average Session Duration",
            "Bounce Rate",
            "Exits",
            "Returning Users"
        ],
        "Detected Column": [
            date_col,
            page_col,
            device_col,
            country_col,
            channel_col,
            users_col,
            sessions_col,
            events_col,
            views_col,
            duration_col,
            bounce_col,
            exit_col,
            returning_col
        ],
        "Executive Purpose": [
            "Shows web activity across the assumed 2025 traffic period",
            "Used as the benchmark for top-page and Rural Development analysis",
            "Shows access behavior across desktop, mobile, tablet, and smart TV",
            "Shows geographic usage patterns",
            "Shows acquisition or navigation source if available",
            "Measures audience reach",
            "Measures visit volume",
            "Measures interaction volume",
            "Acts as an engagement proxy",
            "Acts as a time-on-site engagement proxy",
            "Signals possible navigation or content mismatch",
            "Signals where users leave the site",
            "Signals repeated use or returning demand"
        ]
    })

    st.dataframe(detected_df, use_container_width=True)

    st.subheader("Dataset Profile")

    profile_df = pd.DataFrame({
        "column": filtered_df.columns,
        "dtype": [str(filtered_df[col].dtype) for col in filtered_df.columns],
        "missing_values": [filtered_df[col].isna().sum() for col in filtered_df.columns],
        "missing_rate": [filtered_df[col].isna().mean() for col in filtered_df.columns],
        "unique_values": [filtered_df[col].nunique(dropna=True) for col in filtered_df.columns]
    })

    st.dataframe(profile_df, use_container_width=True)

    st.subheader("Methodology Notes")

    st.markdown(
        """
        **Most necessary executive features included in this dashboard:**

        1. **Traffic Date (2025 Assumed)** to show when web activity occurred.
        2. **Page Title** as the benchmark for identifying high-demand pages.
        3. **Users and Sessions** to measure traffic scale.
        4. **Device Category** to understand digital access patterns across desktop, mobile, tablet, and smart TV.
        5. **Engagement proxies** such as views per session, average session duration, bounce rate, exits, and returning users.
        6. **Rural Development page filter** using page titles that contain `rural development`.
        7. **Clustering and segmentation** to group Rural Development pages into actionable executive categories.
        8. **AI-service prioritization** to identify where chatbot support, guided navigation, or clearer pathways may help users.

        The Page Benchmark Comparison tab was removed because it was less necessary than direct USDA-wide usage,
        Rural Development segmentation, and AI prioritization.
        """
    )

    st.download_button(
        "Download Data Quality Profile",
        profile_df.to_csv(index=False).encode("utf-8"),
        "data_quality_profile.csv",
        "text/csv"
    )
