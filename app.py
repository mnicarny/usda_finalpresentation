# ============================================================
# USDA WEB ANALYTICS & AI INSIGHTS DASHBOARD
# app.py
# ============================================================

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

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
    "Executive-ready web analytics dashboard for USDA-wide usage patterns, "
    "Rural Development page behavior, and AI-enabled service prioritization."
)


# ============================================================
# GLOBAL SETTINGS
# ============================================================

DATA_FOLDER = Path("organized_clean_long_data_full_USDA")
DATA_FILE = Path("organized_clean_long_data_full_USDA.xlsx")


# ============================================================
# UTILITY FUNCTIONS
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
    )
    return df


def find_col(df, keywords):
    for col in df.columns:
        col_lower = col.lower()
        for key in keywords:
            if key in col_lower:
                return col
    return None


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def load_all_data():
    loaded = []
    inventory = []

    possible_files = []

    if DATA_FOLDER.exists() and DATA_FOLDER.is_dir():
        possible_files.extend(list(DATA_FOLDER.glob("*.csv")))
        possible_files.extend(list(DATA_FOLDER.glob("*.xlsx")))
        possible_files.extend(list(DATA_FOLDER.glob("*.xls")))

    if DATA_FILE.exists():
        possible_files.append(DATA_FILE)

    if not possible_files:
        st.error(
            "No data files were found. Please make sure the folder "
            "`organized_clean_long_data_full_USDA` or the file "
            "`organized_clean_long_data_full_USDA.xlsx` is uploaded."
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
                    "type": "CSV",
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
                        "type": f"Excel sheet: {sheet}",
                        "rows": temp.shape[0],
                        "columns": temp.shape[1],
                        "status": "Loaded"
                    })

        except Exception as e:
            inventory.append({
                "file": file.name,
                "type": file.suffix,
                "rows": 0,
                "columns": 0,
                "status": f"Error: {e}"
            })

    if not loaded:
        st.error("Files were detected, but none could be loaded successfully.")
        st.stop()

    df = pd.concat(loaded, ignore_index=True, sort=False)
    return df, pd.DataFrame(inventory)


@st.cache_data
def cached_load_data():
    return load_all_data()


def make_kpi_card(label, value, note=None):
    st.metric(label, value)
    if note:
        st.caption(note)


def format_number(value):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{value:,.0f}"
    except Exception:
        return "N/A"


def get_numeric_columns(df):
    numeric_cols = []
    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            numeric_cols.append(col)
    return numeric_cols


def convert_numeric_columns(df):
    df = df.copy()
    for col in df.columns:
        if col not in ["source_file", "source_sheet"]:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0 and converted.notna().sum() >= len(df) * 0.25:
                df[col] = converted
    return df


def build_auto_insights(filtered_df, page_col, users_col, sessions_col):
    insights = []

    if filtered_df.empty:
        return ["No rows are available after filtering."]

    if page_col and users_col and page_col in filtered_df.columns and users_col in filtered_df.columns:
        page_summary = (
            filtered_df.groupby(page_col, dropna=False)[users_col]
            .sum()
            .sort_values(ascending=False)
        )

        if not page_summary.empty:
            top_page = page_summary.index[0]
            top_value = page_summary.iloc[0]
            total_value = page_summary.sum()

            if total_value > 0:
                share = top_value / total_value * 100
                insights.append(
                    f"The top page is **{top_page}**, representing about **{share:.1f}%** "
                    f"of filtered user activity."
                )

    if sessions_col and sessions_col in filtered_df.columns:
        total_sessions = filtered_df[sessions_col].sum()
        insights.append(
            f"The filtered data contains **{format_number(total_sessions)} sessions**, "
            "which can help identify where USDA digital demand is concentrated."
        )

    if page_col and page_col in filtered_df.columns:
        unique_pages = filtered_df[page_col].nunique()
        insights.append(
            f"There are **{unique_pages:,} unique page titles** in the current filtered view."
        )

    if not insights:
        insights.append(
            "The available fields are limited, so the dashboard can summarize activity "
            "but cannot compute deeper engagement insights."
        )

    return insights


# ============================================================
# LOAD DATA
# ============================================================

df, inventory_df = cached_load_data()
df = convert_numeric_columns(df)

original_df = df.copy()


# ============================================================
# AUTO-DETECT IMPORTANT COLUMNS
# ============================================================

page_col = find_col(df, ["page_title", "page title", "title", "page"])
date_col = find_col(df, ["date", "day", "month"])
country_col = find_col(df, ["country"])
device_col = find_col(df, ["device"])
channel_col = find_col(df, ["channel", "source", "medium"])
cluster_col = find_col(df, ["cluster", "segment"])

users_col = find_col(df, ["active_users", "total_users", "users", "visitors"])
sessions_col = find_col(df, ["sessions", "visits"])
events_col = find_col(df, ["event_count", "events"])
views_col = find_col(df, ["views_per_session", "views", "screen_page_views"])
duration_col = find_col(df, ["average_session_duration", "avg_session_duration", "duration"])
bounce_col = find_col(df, ["bounce_rate", "bounce"])
exit_col = find_col(df, ["exits", "exit"])
returning_col = find_col(df, ["returning_users", "returning"])


# ============================================================
# SIDEBAR FILTERS
# ============================================================

st.sidebar.header("Dashboard Filters")

filtered_df = df.copy()

if date_col and date_col in filtered_df.columns:
    filtered_df[date_col] = pd.to_datetime(filtered_df[date_col], errors="coerce")

    if filtered_df[date_col].notna().sum() > 0:
        min_date = filtered_df[date_col].min().date()
        max_date = filtered_df[date_col].max().date()

        selected_dates = st.sidebar.date_input(
            "Date range",
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

if country_col and country_col in filtered_df.columns:
    countries = sorted(filtered_df[country_col].dropna().astype(str).unique())
    selected_countries = st.sidebar.multiselect(
        "Country",
        countries,
        default=[]
    )
    if selected_countries:
        filtered_df = filtered_df[
            filtered_df[country_col].astype(str).isin(selected_countries)
        ]

if device_col and device_col in filtered_df.columns:
    devices = sorted(filtered_df[device_col].dropna().astype(str).unique())
    selected_devices = st.sidebar.multiselect(
        "Device category",
        devices,
        default=[]
    )
    if selected_devices:
        filtered_df = filtered_df[
            filtered_df[device_col].astype(str).isin(selected_devices)
        ]

if channel_col and channel_col in filtered_df.columns:
    channels = sorted(filtered_df[channel_col].dropna().astype(str).unique())
    selected_channels = st.sidebar.multiselect(
        "Channel / source",
        channels,
        default=[]
    )
    if selected_channels:
        filtered_df = filtered_df[
            filtered_df[channel_col].astype(str).isin(selected_channels)
        ]

page_search = st.sidebar.text_input("Search page title")

if page_col and page_search:
    filtered_df = filtered_df[
        filtered_df[page_col]
        .astype(str)
        .str.lower()
        .str.contains(page_search.lower(), na=False)
    ]

top_n = st.sidebar.slider("Top N pages / categories", 5, 50, 15)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Executive Overview",
    "USDA-wide Web Analytics",
    "Rural Development Cluster Profile",
    "Page Benchmark Comparison",
    "Data Quality / Methodology"
])


# ============================================================
# TAB 1: EXECUTIVE OVERVIEW
# ============================================================

with tab1:
    st.header("Executive Overview")

    st.write(
        "This dashboard summarizes USDA public web activity and highlights Rural Development "
        "page patterns that may benefit from AI-enabled support such as chatbots, guided navigation, "
        "and clearer digital service pathways."
    )

    total_users = filtered_df[users_col].sum() if users_col in filtered_df.columns else np.nan
    total_sessions = filtered_df[sessions_col].sum() if sessions_col in filtered_df.columns else np.nan
    total_events = filtered_df[events_col].sum() if events_col in filtered_df.columns else np.nan

    rd_mask = (
        filtered_df[page_col]
        .astype(str)
        .str.lower()
        .str.contains("rural development", na=False)
        if page_col and page_col in filtered_df.columns
        else pd.Series(False, index=filtered_df.index)
    )

    rd_df = filtered_df[rd_mask].copy()

    rd_users = rd_df[users_col].sum() if users_col in rd_df.columns else np.nan
    rd_share = (rd_users / total_users * 100) if users_col and total_users and total_users > 0 else np.nan

    unique_pages = filtered_df[page_col].nunique() if page_col in filtered_df.columns else np.nan

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        make_kpi_card("Total Users", format_number(total_users), "Sum of detected user metric")

    with c2:
        make_kpi_card("Total Sessions", format_number(total_sessions), "Sum of detected sessions metric")

    with c3:
        make_kpi_card("Total Events", format_number(total_events), "Sum of detected event metric")

    with c4:
        make_kpi_card("Unique Pages", format_number(unique_pages), "Distinct detected page titles")

    with c5:
        if pd.notna(rd_share):
            make_kpi_card("Rural Dev Share", f"{rd_share:.1f}%", "Share of users from Rural Development pages")
        else:
            make_kpi_card("Rural Dev Share", "N/A", "Requires page title and user metric")

    st.subheader("Key Insights")

    insights = build_auto_insights(filtered_df, page_col, users_col, sessions_col)

    for insight in insights:
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
# TAB 2: USDA-WIDE WEB ANALYTICS
# ============================================================

with tab2:
    st.header("USDA-wide Web Analytics")

    if filtered_df.empty:
        st.warning("No data available after applying filters.")

    else:
        if page_col and users_col and page_col in filtered_df.columns and users_col in filtered_df.columns:
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
                title="Top Pages by Users",
                labels={
                    users_col: "Users",
                    page_col: "Page Title"
                }
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=600)
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Page title and/or user metric was not detected.")

        if date_col and users_col and date_col in filtered_df.columns and users_col in filtered_df.columns:
            st.subheader("Trend Over Time")

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
                    title="User Activity Trend Over Time",
                    labels={
                        date_col: "Date",
                        users_col: "Users"
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Date values are unavailable after filtering.")

        if device_col and users_col and device_col in filtered_df.columns and users_col in filtered_df.columns:
            st.subheader("Device-Level Activity")

            device_df = (
                filtered_df.groupby(device_col, dropna=False)[users_col]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig = px.pie(
                device_df,
                names=device_col,
                values=users_col,
                title="User Activity by Device Category"
            )
            st.plotly_chart(fig, use_container_width=True)

        if country_col and users_col and country_col in filtered_df.columns and users_col in filtered_df.columns:
            st.subheader("Country-Level Activity")

            country_df = (
                filtered_df.groupby(country_col, dropna=False)[users_col]
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
# TAB 3: RURAL DEVELOPMENT CLUSTER PROFILE
# ============================================================

with tab3:
    st.header("Rural Development Cluster Profile")

    if not page_col or page_col not in filtered_df.columns:
        st.warning("Page title column was not detected, so Rural Development pages cannot be identified.")

    else:
        rd_df = filtered_df[
            filtered_df[page_col]
            .astype(str)
            .str.lower()
            .str.contains("rural development", na=False)
        ].copy()

        if rd_df.empty:
            st.warning(
                "No page titles containing 'rural development' were found after filters. "
                "Please loosen the sidebar filters or clear the page search."
            )

        else:
            st.success(f"{len(rd_df):,} rows matched page titles containing 'rural development'.")

            c1, c2, c3 = st.columns(3)

            with c1:
                if users_col and users_col in rd_df.columns:
                    st.metric("Rural Development Users", format_number(rd_df[users_col].sum()))
                else:
                    st.metric("Rural Development Users", "N/A")

            with c2:
                if sessions_col and sessions_col in rd_df.columns:
                    st.metric("Rural Development Sessions", format_number(rd_df[sessions_col].sum()))
                else:
                    st.metric("Rural Development Sessions", "N/A")

            with c3:
                st.metric("Rural Development Pages", f"{rd_df[page_col].nunique():,}")

            possible_metrics = [
                users_col,
                sessions_col,
                events_col,
                views_col,
                duration_col,
                bounce_col,
                exit_col,
                returning_col
            ]

            segment_metrics = []

            for col in possible_metrics:
                if col and col in rd_df.columns:
                    rd_df[col] = pd.to_numeric(rd_df[col], errors="coerce")
                    if rd_df[col].notna().sum() > 0 and col not in segment_metrics:
                        segment_metrics.append(col)

            if len(segment_metrics) < 2:
                st.warning(
                    "Not enough numeric metrics were detected for clustering. "
                    "At least two numeric metrics are required."
                )

            else:
                st.subheader("Segmentation Inputs")

                selected_metrics = st.multiselect(
                    "Choose metrics used for clustering",
                    segment_metrics,
                    default=segment_metrics[:min(6, len(segment_metrics))]
                )

                if len(selected_metrics) < 2:
                    st.warning("Please select at least two metrics for clustering.")

                else:
                    rd_page_segments = (
                        rd_df.groupby(page_col, dropna=False)[selected_metrics]
                        .sum()
                        .reset_index()
                    )

                    if users_col and users_col in rd_page_segments.columns and sessions_col and sessions_col in rd_page_segments.columns:
                        rd_page_segments["sessions_per_user"] = (
                            rd_page_segments[sessions_col] /
                            rd_page_segments[users_col].replace(0, np.nan)
                        ).fillna(0)

                    if users_col and users_col in rd_page_segments.columns:
                        total_rd_users = rd_page_segments[users_col].sum()
                        rd_page_segments["share_of_rd_users"] = (
                            rd_page_segments[users_col] / total_rd_users
                            if total_rd_users > 0
                            else 0
                        )

                    clustering_cols = [
                        col for col in rd_page_segments.columns
                        if col != page_col and pd.api.types.is_numeric_dtype(rd_page_segments[col])
                    ]

                    n_pages = len(rd_page_segments)

                    if n_pages < 2:
                        st.warning(
                            "Not enough Rural Development pages after filtering to run clustering. "
                            "Please loosen the filters or include more pages."
                        )

                    elif len(clustering_cols) < 2:
                        st.warning("Not enough usable numeric columns for clustering.")

                    else:
                        max_possible_clusters = min(8, n_pages)

                        k = st.slider(
                            "Number of segments",
                            min_value=2,
                            max_value=max_possible_clusters,
                            value=min(4, max_possible_clusters),
                            help=(
                                "The maximum number of clusters automatically adjusts based on "
                                "the number of Rural Development pages available after filtering."
                            )
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

                            safe_k = min(k, X_scaled.shape[0])

                            if safe_k < 2:
                                st.warning(
                                    "Clustering cannot run because fewer than two pages are available."
                                )

                            else:
                                if safe_k < k:
                                    st.warning(
                                        f"Only {X_scaled.shape[0]} pages are available, so clusters were "
                                        f"automatically reduced from {k} to {safe_k}."
                                    )

                                kmeans = KMeans(
                                    n_clusters=safe_k,
                                    random_state=42,
                                    n_init=20
                                )

                                rd_page_segments["segment"] = kmeans.fit_predict(X_scaled)

                                if users_col and users_col in rd_page_segments.columns:
                                    segment_rank = (
                                        rd_page_segments.groupby("segment")[users_col]
                                        .sum()
                                        .sort_values(ascending=False)
                                        .index
                                        .tolist()
                                    )
                                else:
                                    segment_rank = (
                                        rd_page_segments["segment"]
                                        .value_counts()
                                        .index
                                        .tolist()
                                    )

                                executive_names = [
                                    "High-Priority Digital Access Pages",
                                    "Broad Utility / Navigation Pages",
                                    "Niche but Important Service Pages",
                                    "Low-Traffic Improvement Candidates",
                                    "High-Exit Attention Pages",
                                    "Returning-User Support Pages",
                                    "Emerging Demand Pages",
                                    "Specialized Information Pages",
                                ]

                                segment_name_map = {
                                    seg: executive_names[i]
                                    for i, seg in enumerate(segment_rank)
                                }

                                rd_page_segments["segment_label"] = rd_page_segments["segment"].map(segment_name_map)

                                st.success(
                                    f"{len(rd_page_segments):,} Rural Development pages were grouped into "
                                    f"{safe_k} executive segments."
                                )

                                st.subheader("Segment Opportunity Map")

                                x_metric = sessions_col if sessions_col in rd_page_segments.columns else clustering_cols[0]

                                y_metric = None
                                for candidate in [views_col, duration_col, returning_col, bounce_col, exit_col]:
                                    if candidate and candidate in rd_page_segments.columns:
                                        y_metric = candidate
                                        break

                                if y_metric is None:
                                    y_metric = clustering_cols[1]

                                size_metric = users_col if users_col in rd_page_segments.columns else clustering_cols[0]

                                fig = px.scatter(
                                    rd_page_segments,
                                    x=x_metric,
                                    y=y_metric,
                                    size=size_metric,
                                    color="segment_label",
                                    hover_name=page_col,
                                    hover_data=clustering_cols[:8],
                                    title="Rural Development Segments: Traffic vs Engagement Opportunity",
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

                                profile_metrics = clustering_cols.copy()

                                segment_profile_mean = (
                                    rd_page_segments.groupby("segment_label")[profile_metrics]
                                    .mean()
                                    .reset_index()
                                )

                                if users_col and users_col in rd_page_segments.columns:
                                    segment_size = (
                                        rd_page_segments.groupby("segment_label")
                                        .agg(
                                            pages=(page_col, "count"),
                                            total_users=(users_col, "sum")
                                        )
                                        .reset_index()
                                    )
                                else:
                                    segment_size = (
                                        rd_page_segments.groupby("segment_label")
                                        .agg(pages=(page_col, "count"))
                                        .reset_index()
                                    )

                                segment_profile = segment_size.merge(
                                    segment_profile_mean,
                                    on="segment_label",
                                    how="left"
                                )

                                st.dataframe(segment_profile, use_container_width=True)

                                st.subheader("Segment Behavior Heatmap")

                                heatmap_cols = segment_profile.select_dtypes(include=np.number).columns.tolist()

                                if heatmap_cols:
                                    heatmap_scaled = segment_profile.copy()

                                    for col in heatmap_cols:
                                        min_val = heatmap_scaled[col].min()
                                        max_val = heatmap_scaled[col].max()

                                        if max_val != min_val:
                                            heatmap_scaled[col] = (
                                                (heatmap_scaled[col] - min_val) /
                                                (max_val - min_val)
                                            )
                                        else:
                                            heatmap_scaled[col] = 0

                                    fig = px.imshow(
                                        heatmap_scaled[heatmap_cols],
                                        y=heatmap_scaled["segment_label"],
                                        x=[c.replace("_", " ").title() for c in heatmap_cols],
                                        aspect="auto",
                                        title="Relative Segment Strengths by Metric"
                                    )
                                    fig.update_layout(height=500)
                                    st.plotly_chart(fig, use_container_width=True)

                                st.subheader("AI-Enabled Service Prioritization")

                                recommendation_rows = []

                                for _, row in segment_profile.iterrows():
                                    segment_name = row["segment_label"]
                                    pages_count = int(row["pages"])

                                    recommendation = "Monitor content clarity and review navigation pathways."

                                    if bounce_col and bounce_col in row.index and row[bounce_col] > segment_profile[bounce_col].median():
                                        recommendation = (
                                            "Prioritize guided navigation or chatbot support to reduce early exits "
                                            "and help users find the correct service path."
                                        )

                                    elif exit_col and exit_col in row.index and row[exit_col] > segment_profile[exit_col].median():
                                        recommendation = (
                                            "Improve next-step prompts, related links, and page-level service pathways."
                                        )

                                    elif users_col and "total_users" in row.index and row["total_users"] >= segment_profile["total_users"].median():
                                        recommendation = (
                                            "Prioritize chatbot FAQs and self-service support because this segment "
                                            "concentrates visible user demand."
                                        )

                                    elif returning_col and returning_col in row.index and row[returning_col] > segment_profile[returning_col].median():
                                        recommendation = (
                                            "Add task-oriented support for repeat visitors who may be returning "
                                            "to complete service-related actions."
                                        )

                                    recommendation_rows.append({
                                        "Segment": segment_name,
                                        "Pages": pages_count,
                                        "Executive Recommendation": recommendation
                                    })

                                rec_df = pd.DataFrame(recommendation_rows)
                                st.dataframe(rec_df, use_container_width=True)

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
                                    "Download Rural Development Segmentation Results",
                                    rd_page_segments.to_csv(index=False).encode("utf-8"),
                                    "rural_development_segmentation_results.csv",
                                    "text/csv"
                                )

                                st.subheader("Auto-Generated Executive Insight")

                                top_segment = (
                                    rd_page_segments.groupby("segment_label")[size_metric]
                                    .sum()
                                    .sort_values(ascending=False)
                                    .index[0]
                                )

                                top_page = (
                                    rd_page_segments.sort_values(by=size_metric, ascending=False)
                                    .iloc[0][page_col]
                                )

                                st.info(
                                    f"The largest Rural Development opportunity segment is "
                                    f"**{top_segment}**, based on **{size_metric.replace('_', ' ')}**. "
                                    f"The highest-priority page in the filtered Rural Development set is "
                                    f"**{top_page}**. These pages are strong candidates for AI-enabled support "
                                    f"because they concentrate user demand and can benefit from clearer navigation, "
                                    f"automated FAQs, and guided service pathways."
                                )

                        except Exception as e:
                            st.warning(
                                "Clustering could not be completed with the current filters. "
                                "Please loosen filters or select different metrics."
                            )
                            st.caption(f"Technical note: {e}")


# ============================================================
# TAB 4: PAGE BENCHMARK COMPARISON
# ============================================================

with tab4:
    st.header("Page Benchmark Comparison")

    if not page_col or not users_col:
        st.warning("Page title and user metric are required for benchmark comparison.")

    elif filtered_df.empty:
        st.warning("No data available after filtering.")

    else:
        page_summary = (
            filtered_df.groupby(page_col, dropna=False)
            .agg(
                users=(users_col, "sum")
            )
            .reset_index()
        )

        if sessions_col and sessions_col in filtered_df.columns:
            session_summary = (
                filtered_df.groupby(page_col, dropna=False)[sessions_col]
                .sum()
                .reset_index()
            )
            page_summary = page_summary.merge(session_summary, on=page_col, how="left")

        page_summary["is_rural_development"] = (
            page_summary[page_col]
            .astype(str)
            .str.lower()
            .str.contains("rural development", na=False)
        )

        comparison = (
            page_summary.groupby("is_rural_development")
            .agg(
                pages=(page_col, "count"),
                total_users=("users", "sum"),
                avg_users_per_page=("users", "mean")
            )
            .reset_index()
        )

        comparison["group"] = comparison["is_rural_development"].map({
            True: "Rural Development Pages",
            False: "Other USDA Pages"
        })

        st.subheader("Rural Development vs Other USDA Pages")

        fig = px.bar(
            comparison,
            x="group",
            y="total_users",
            title="Total Users: Rural Development vs Other USDA Pages",
            labels={
                "group": "Page Group",
                "total_users": "Total Users"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.bar(
            comparison,
            x="group",
            y="avg_users_per_page",
            title="Average Users per Page",
            labels={
                "group": "Page Group",
                "avg_users_per_page": "Average Users per Page"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(comparison, use_container_width=True)


# ============================================================
# TAB 5: DATA QUALITY / METHODOLOGY
# ============================================================

with tab5:
    st.header("Data Quality / Methodology")

    st.subheader("Data Inventory")
    st.dataframe(inventory_df, use_container_width=True)

    st.subheader("Detected Columns")

    detected = pd.DataFrame({
        "Field": [
            "Page title",
            "Date",
            "Country",
            "Device",
            "Channel",
            "Cluster / segment",
            "Users",
            "Sessions",
            "Events",
            "Views",
            "Duration",
            "Bounce rate",
            "Exits",
            "Returning users"
        ],
        "Detected column": [
            page_col,
            date_col,
            country_col,
            device_col,
            channel_col,
            cluster_col,
            users_col,
            sessions_col,
            events_col,
            views_col,
            duration_col,
            bounce_col,
            exit_col,
            returning_col
        ]
    })

    st.dataframe(detected, use_container_width=True)

    st.subheader("Dataset Profile")

    profile = pd.DataFrame({
        "column": filtered_df.columns,
        "dtype": [str(filtered_df[col].dtype) for col in filtered_df.columns],
        "missing_values": [filtered_df[col].isna().sum() for col in filtered_df.columns],
        "missing_rate": [filtered_df[col].isna().mean() for col in filtered_df.columns],
        "unique_values": [filtered_df[col].nunique(dropna=True) for col in filtered_df.columns]
    })

    st.dataframe(profile, use_container_width=True)

    st.subheader("Methodology Notes")

    st.markdown(
        """
        - Data files are loaded dynamically from `organized_clean_long_data_full_USDA` or `organized_clean_long_data_full_USDA.xlsx`.
        - Column names are normalized by stripping spaces, lowercasing, and replacing spaces with underscores.
        - Rural Development pages are identified only through page titles containing the phrase `rural development`.
        - Clustering is performed at the page-title level, not raw-row level, so executives can interpret segments as service/page groups.
        - KMeans clustering uses standardized numeric metrics and log transformation for high-volume traffic metrics.
        - The number of clusters automatically adjusts when filters reduce the available number of pages.
        - If required metrics are missing, the dashboard shows warnings instead of fabricating values.
        """
    )

    st.download_button(
        "Download Data Quality Profile",
        profile.to_csv(index=False).encode("utf-8"),
        "data_quality_profile.csv",
        "text/csv"
    )
