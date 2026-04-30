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
