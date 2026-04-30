# ===============================
# USDA STREAMLIT DASHBOARD APP
# ===============================

import os
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(layout="wide")
st.title("USDA Web Analytics & AI Insights Dashboard")
st.caption("Executive dashboard: USDA-wide analytics + Rural Development clustering")


# ===============================
# LOAD DATA
# ===============================
DATA_PATH = Path("organized_clean_long_data_full_USDA.xlsx")

@st.cache_data
def load_data():
    try:
        df = pd.read_excel(DATA_PATH)
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

df = load_data()

if df is None:
    st.stop()


# ===============================
# DATA OVERVIEW
# ===============================
st.subheader("Data Overview")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))


# ===============================
# AUTO DETECT KEY COLUMNS
# ===============================
def find_col(keywords):
    for col in df.columns:
        for k in keywords:
            if k in col:
                return col
    return None

users_col = find_col(["user"])
sessions_col = find_col(["session"])
page_col = find_col(["page", "title"])
date_col = find_col(["date"])
rd_col = find_col(["rural", "development"])


# ===============================
# KPI SECTION
# ===============================
st.subheader("Executive KPIs")

col1, col2, col3 = st.columns(3)

with col1:
    if users_col:
        st.metric("Total Users", int(df[users_col].sum()))
    else:
        st.warning("Users column not found")

with col2:
    if sessions_col:
        st.metric("Total Sessions", int(df[sessions_col].sum()))
    else:
        st.warning("Sessions column not found")

with col3:
    st.metric("Rows", len(df))


# ===============================
# FILTERS
# ===============================
st.sidebar.header("Filters")

if page_col:
    selected_pages = st.sidebar.multiselect(
        "Select Pages",
        df[page_col].dropna().unique()
    )
    if selected_pages:
        df = df[df[page_col].isin(selected_pages)]


# ===============================
# USDA-WIDE ANALYTICS
# ===============================
st.header("USDA-Wide Web Analytics")

if page_col and users_col:
    top_pages = (
        df.groupby(page_col)[users_col]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )

    fig = px.bar(
        top_pages,
        x=users_col,
        y=page_col,
        orientation="h",
        title="Top Pages by Users"
    )

    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Missing page/users columns")


# ===============================
# TREND OVER TIME
# ===============================
if date_col and users_col:
    trend = df.groupby(date_col)[users_col].sum().reset_index()

    fig = px.line(
        trend,
        x=date_col,
        y=users_col,
        title="User Trend Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)


# ===============================
# CLUSTERING (IF POSSIBLE)
# ===============================
st.header("User Segmentation (Clustering)")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

if len(numeric_cols) >= 2:
    try:
        X = df[numeric_cols].fillna(0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["cluster"] = kmeans.fit_predict(X_scaled)

        st.success("Clustering applied successfully")

        fig = px.scatter(
            df,
            x=numeric_cols[0],
            y=numeric_cols[1],
            color="cluster",
            title="Cluster Visualization"
        )
        st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.warning(f"Clustering failed: {e}")
else:
    st.warning("Not enough numeric columns for clustering")


# ===============================
# RURAL DEVELOPMENT VIEW (REVISED)
# ===============================
st.header("Rural Development Analysis")

if page_col:
    # Filter rows where page title contains "rural development"
    rd_df = df[
        df[page_col]
        .astype(str)
        .str.lower()
        .str.contains("rural development", na=False)
    ]

    if rd_df.empty:
        st.warning("No page titles containing 'rural development' found.")
    else:
        st.success(f"{len(rd_df):,} rows matched 'rural development' in page titles.")

        # KPIs
        col1, col2 = st.columns(2)

        with col1:
            if users_col:
                st.metric("Total Users (Rural Dev)", int(rd_df[users_col].fillna(0).sum()))

        with col2:
            if sessions_col:
                st.metric("Total Sessions (Rural Dev)", int(rd_df[sessions_col].fillna(0).sum()))

        # Top pages chart
        if users_col:
            rd_pages = (
                rd_df.groupby(page_col)[users_col]
                .sum()
                .sort_values(ascending=False)
                .head(15)
                .reset_index()
            )

            fig = px.bar(
                rd_pages,
                x=users_col,
                y=page_col,
                orientation="h",
                title="Top Rural Development Pages by Users"
            )

            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        # Table + download
        st.subheader("Filtered Rural Development Data")
        st.dataframe(rd_df, use_container_width=True)

        st.download_button(
            "Download Rural Development Data",
            rd_df.to_csv(index=False).encode("utf-8"),
            "rural_development_data.csv",
            "text/csv"
        )

else:
    st.warning("Page title column not found.")

# ===============================
# RURAL DEVELOPMENT CLUSTERING + SEGMENTATION
# ===============================
st.header("Rural Development Clustering & Segmentation")

if page_col:
    rd_df = df[
        df[page_col]
        .astype(str)
        .str.lower()
        .str.contains("rural development", na=False)
    ].copy()

    if rd_df.empty:
        st.warning("No Rural Development page titles found for clustering.")
    else:
        st.caption(
            "This section segments Rural Development pages based on traffic, engagement, "
            "retention, and exit behavior to help prioritize AI-enabled services."
        )

        # Candidate metrics for segmentation
        possible_metrics = [
            users_col,
            sessions_col,
            find_col(["active_users"]),
            find_col(["total_users"]),
            find_col(["event_count"]),
            find_col(["views_per_session"]),
            find_col(["average_session_duration"]),
            find_col(["bounce_rate"]),
            find_col(["exits"]),
            find_col(["returning_users"]),
        ]

        segment_metrics = []
        for col in possible_metrics:
            if col and col in rd_df.columns and pd.api.types.is_numeric_dtype(rd_df[col]):
                if col not in segment_metrics:
                    segment_metrics.append(col)

        if len(segment_metrics) < 2:
            st.warning(
                "Not enough numeric Rural Development metrics were found for clustering. "
                "At least two numeric metrics are needed."
            )
        else:
            st.subheader("Segmentation Inputs")
            selected_metrics = st.multiselect(
                "Choose metrics used for clustering",
                segment_metrics,
                default=segment_metrics[: min(6, len(segment_metrics))]
            )

            k = st.slider(
                "Number of segments",
                min_value=2,
                max_value=min(8, max(2, len(rd_df) // 5)),
                value=min(4, max(2, len(rd_df) // 10)),
                help="Higher values create more detailed segments; lower values create broader executive categories."
            )

            if len(selected_metrics) < 2:
                st.warning("Select at least two metrics for clustering.")
            else:
                # Aggregate to page level so segmentation is interpretable for executives
                rd_page_segments = (
                    rd_df.groupby(page_col, dropna=False)[selected_metrics]
                    .sum()
                    .reset_index()
                )

                # Basic derived ratios when possible
                if users_col in rd_page_segments.columns and sessions_col in rd_page_segments.columns:
                    rd_page_segments["sessions_per_user"] = (
                        rd_page_segments[sessions_col] /
                        rd_page_segments[users_col].replace(0, np.nan)
                    ).fillna(0)

                if users_col in rd_page_segments.columns:
                    total_rd_users = rd_page_segments[users_col].sum()
                    rd_page_segments["share_of_rd_users"] = (
                        rd_page_segments[users_col] / total_rd_users
                        if total_rd_users > 0 else 0
                    )

                # Clean numeric clustering matrix
                clustering_cols = [
                    col for col in rd_page_segments.columns
                    if col != page_col and pd.api.types.is_numeric_dtype(rd_page_segments[col])
                ]

                X = rd_page_segments[clustering_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

                # Log transform traffic-heavy metrics to reduce dominance of very large pages
                X_transformed = X.copy()
                for col in X_transformed.columns:
                    if X_transformed[col].min() >= 0 and X_transformed[col].max() > 10:
                        X_transformed[col] = np.log1p(X_transformed[col])

                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_transformed)

                kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
                rd_page_segments["segment"] = kmeans.fit_predict(X_scaled)

                # Rank segments by size/traffic for friendlier executive labels
                if users_col in rd_page_segments.columns:
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

                segment_name_map = {}
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

                for i, seg in enumerate(segment_rank):
                    segment_name_map[seg] = executive_names[i]

                rd_page_segments["segment_label"] = rd_page_segments["segment"].map(segment_name_map)

                st.success(
                    f"{len(rd_page_segments):,} Rural Development pages were grouped into {k} executive segments."
                )

                # Segment profile table
                st.subheader("Executive Segment Profiles")

                profile_metrics = [col for col in clustering_cols if col in rd_page_segments.columns]

                segment_profile = (
                    rd_page_segments.groupby("segment_label")[profile_metrics]
                    .mean()
                    .reset_index()
                )

                if users_col in rd_page_segments.columns:
                    size_profile = (
                        rd_page_segments.groupby("segment_label")
                        .agg(
                            pages=(page_col, "count"),
                            total_users=(users_col, "sum")
                        )
                        .reset_index()
                    )
                else:
                    size_profile = (
                        rd_page_segments.groupby("segment_label")
                        .agg(pages=(page_col, "count"))
                        .reset_index()
                    )

                segment_profile = size_profile.merge(segment_profile, on="segment_label", how="left")

                st.dataframe(segment_profile, use_container_width=True)

                # Bubble chart: traffic vs engagement proxy
                st.subheader("Segment Opportunity Map")

                x_metric = sessions_col if sessions_col in rd_page_segments.columns else clustering_cols[0]
                y_metric = None

                for candidate in ["views_per_session", "average_session_duration", "returning_users"]:
                    if candidate in rd_page_segments.columns:
                        y_metric = candidate
                        break

                if y_metric is None:
                    y_metric = clustering_cols[1] if len(clustering_cols) > 1 else clustering_cols[0]

                size_metric = users_col if users_col in rd_page_segments.columns else clustering_cols[0]

                fig = px.scatter(
                    rd_page_segments,
                    x=x_metric,
                    y=y_metric,
                    size=size_metric,
                    color="segment_label",
                    hover_name=page_col,
                    hover_data=clustering_cols[:8],
                    title="Rural Development Page Segments: Traffic vs Engagement Opportunity",
                    labels={
                        x_metric: x_metric.replace("_", " ").title(),
                        y_metric: y_metric.replace("_", " ").title(),
                        size_metric: size_metric.replace("_", " ").title(),
                        "segment_label": "Segment"
                    },
                )

                fig.update_layout(height=650)
                st.plotly_chart(fig, use_container_width=True)

                # Heatmap for executive comparison
                st.subheader("Segment Behavior Heatmap")

                heatmap_df = segment_profile.copy()

                numeric_heatmap_cols = heatmap_df.select_dtypes(include=np.number).columns.tolist()
                if numeric_heatmap_cols:
                    heatmap_scaled = heatmap_df.copy()
                    for col in numeric_heatmap_cols:
                        min_val = heatmap_scaled[col].min()
                        max_val = heatmap_scaled[col].max()
                        if max_val != min_val:
                            heatmap_scaled[col] = (heatmap_scaled[col] - min_val) / (max_val - min_val)
                        else:
                            heatmap_scaled[col] = 0

                    fig = px.imshow(
                        heatmap_scaled[numeric_heatmap_cols],
                        y=heatmap_scaled["segment_label"],
                        x=[c.replace("_", " ").title() for c in numeric_heatmap_cols],
                        aspect="auto",
                        title="Relative Segment Strengths by Metric"
                    )

                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                # AI prioritization logic
                st.subheader("AI-Enabled Service Prioritization")

                recommendation_rows = []

                for _, row in segment_profile.iterrows():
                    segment_name = row["segment_label"]
                    pages_count = int(row["pages"])

                    total_users_value = row["total_users"] if "total_users" in row else np.nan

                    recommendation = "Monitor and improve content clarity."

                    if "bounce_rate" in row.index and row["bounce_rate"] > segment_profile["bounce_rate"].median():
                        recommendation = "Prioritize guided navigation or chatbot support to reduce early exits."
                    elif "exits" in row.index and row["exits"] > segment_profile["exits"].median():
                        recommendation = "Review page pathways and add clearer next-step prompts."
                    elif users_col and "total_users" in row.index and row["total_users"] >= segment_profile["total_users"].median():
                        recommendation = "Prioritize chatbot FAQs and high-volume self-service support."
                    elif "returning_users" in row.index and row["returning_users"] > segment_profile["returning_users"].median():
                        recommendation = "Add task-oriented support for repeat visitors."

                    recommendation_rows.append({
                        "Segment": segment_name,
                        "Pages": pages_count,
                        "Traffic": round(total_users_value, 2) if not pd.isna(total_users_value) else "N/A",
                        "Executive Interpretation": recommendation,
                    })

                rec_df = pd.DataFrame(recommendation_rows)
                st.dataframe(rec_df, use_container_width=True)

                # Page-level segmented table
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

                # Evidence-based insight panel
                st.subheader("Auto-Generated Executive Insights")

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
                    f"The largest Rural Development opportunity segment is **{top_segment}**, "
                    f"based on **{size_metric.replace('_', ' ')}**. "
                    f"The highest-priority page in the filtered Rural Development set is **{top_page}**. "
                    "These pages are strong candidates for AI-enabled support because they concentrate user demand "
                    "and can benefit from clearer navigation, automated FAQs, and guided service pathways."
                )

else:
    st.warning("Page title column not found. Rural Development clustering cannot be completed.")
# ===============================
# DATA TABLE + DOWNLOAD
# ===============================
st.header("Detailed Data")

st.dataframe(df)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download Data",
    csv,
    "usda_data.csv",
    "text/csv"
)


# ===============================
# INSIGHTS
# ===============================
st.header("Key Insights")

insights = []

if users_col and page_col:
    top_page = (
        df.groupby(page_col)[users_col]
        .sum()
        .idxmax()
    )
    insights.append(f"Top performing page: {top_page}")

if "cluster" in df.columns:
    insights.append("User segments identified via clustering")

if insights:
    for i in insights:
        st.write("-", i)
else:
    st.write("Not enough data to generate insights")


# ===============================
# METHODOLOGY
# ===============================
st.header("Methodology")

st.write("""
- Data cleaned and normalized dynamically
- Columns auto-detected (no hardcoding)
- KPIs computed from available fields
- KMeans clustering used for segmentation
- Graceful handling for missing fields
""")
