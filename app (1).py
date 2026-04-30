import os
import io
import re
import math
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:
    px = None
    PLOTLY_AVAILABLE = False

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


st.set_page_config(page_title="USDA Digital Pathway & AI Insights Suite", layout="wide")

if not PLOTLY_AVAILABLE:
    st.error("Plotly is not installed. Add `plotly` to requirements.txt and redeploy the app.")
    st.stop()


PROJECT_FOLDER = Path("organized_clean_long_data_full_USDA")
PROJECT_FILE_STEMS = [
    Path("organized_clean_long_data_full_USDA.xlsx"),
    Path("organized_clean_long_data_full_USDA.csv"),
    Path("organized_clean_long_data_full_USDA.parquet"),
]

PREFERRED_METRICS = [
    "sessions",
    "active_users",
    "total_users",
    "event_count",
    "views_per_session",
    "average_session_duration",
    "bounce_rate",
    "exits",
    "returning_users",
]


st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .metric-card {
        border: 1px solid rgba(49, 51, 63, 0.14);
        border-radius: 18px;
        padding: 18px 18px 14px 18px;
        background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(248,250,252,0.98));
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.04);
        min-height: 116px;
    }
    .metric-label {
        font-size: 0.82rem;
        color: #64748b;
        margin-bottom: 0.35rem;
        line-height: 1.2rem;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 750;
        color: #0f172a;
        line-height: 2rem;
    }
    .metric-note {
        font-size: 0.76rem;
        color: #64748b;
        margin-top: 0.45rem;
    }
    .insight-box {
        border-left: 5px solid #4f772d;
        background: #f7fbf4;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .warning-box {
        border-left: 5px solid #b7791f;
        background: #fffaf0;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }
    .small-muted {color:#64748b; font-size:0.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_column_name(col):
    col = str(col).strip().lower()
    col = re.sub(r"[^\w\s]+", " ", col)
    col = re.sub(r"\s+", "_", col)
    return col.strip("_")


def clean_dataframe(df):
    df = df.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["", "nan", "None", "NaT"]), col] = np.nan
    for col in df.columns:
        if df[col].dtype == "object":
            converted = pd.to_numeric(df[col].str.replace(",", "", regex=False), errors="coerce")
            if converted.notna().mean() >= 0.75:
                df[col] = converted
    for col in df.columns:
        if any(token in col for token in ["date", "day", "month", "year"]):
            if "month" not in col and "day" not in col and "year" not in col:
                parsed = pd.to_datetime(df[col], errors="coerce")
                if parsed.notna().mean() >= 0.50:
                    df[col] = parsed
    return df


def load_one_file(path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return {"default": clean_dataframe(pd.read_csv(path))}
    if suffix in [".xlsx", ".xls"]:
        sheets = pd.read_excel(path, sheet_name=None)
        return {sheet: clean_dataframe(frame) for sheet, frame in sheets.items()}
    if suffix == ".parquet":
        return {"default": clean_dataframe(pd.read_parquet(path))}
    return {}


@st.cache_data(show_spinner=False)
def load_project_data():
    detected_files = []
    frames = {}

    if PROJECT_FOLDER.exists() and PROJECT_FOLDER.is_dir():
        for ext in ["*.csv", "*.xlsx", "*.xls", "*.parquet"]:
            detected_files.extend(sorted(PROJECT_FOLDER.rglob(ext)))
    else:
        detected_files = [p for p in PROJECT_FILE_STEMS if p.exists()]

    inventory = []
    for path in detected_files:
        try:
            loaded = load_one_file(path)
            for sheet_name, frame in loaded.items():
                if frame.empty:
                    continue
                key = f"{path.name}::{sheet_name}"
                frame["_source_file"] = path.name
                frame["_source_sheet"] = sheet_name
                frames[key] = frame
                inventory.append(
                    {
                        "file": path.name,
                        "sheet_or_table": sheet_name,
                        "rows": len(frame),
                        "columns": len(frame.columns),
                        "detected_columns": ", ".join(list(frame.columns[:12])),
                    }
                )
        except Exception as exc:
            inventory.append(
                {
                    "file": path.name,
                    "sheet_or_table": "unreadable",
                    "rows": 0,
                    "columns": 0,
                    "detected_columns": f"Error: {exc}",
                }
            )

    if not frames:
        return pd.DataFrame(), pd.DataFrame(inventory)

    combined = pd.concat(frames.values(), ignore_index=True, sort=False)
    combined = combined.drop_duplicates()
    return combined, pd.DataFrame(inventory)


def find_col(columns, candidates, contains_any=None):
    cols = list(columns)
    for cand in candidates:
        if cand in cols:
            return cand
    if contains_any:
        for col in cols:
            if all(token in col for token in contains_any):
                return col
    return None


def detect_schema(df):
    columns = df.columns
    schema = {
        "date": find_col(columns, ["traffic_date_2025_assumed", "traffic_date", "date", "event_date"]),
        "page": find_col(columns, ["page_title_benchmark", "page_title", "title", "page"], ["page"]),
        "path": find_col(columns, ["page_path_and_screen_class", "page_path", "screen_class", "path"]),
        "country": find_col(columns, ["country", "country_name", "region"]),
        "device": find_col(columns, ["device_category", "device", "platform"]),
        "cluster": find_col(columns, ["cluster", "cluster_label", "segment", "segment_label", "user_segment"]),
        "rural": None,
    }

    for col in columns:
        if "rural" in col and "development" in col:
            schema["rural"] = col
            break

    metrics = {}
    for metric in PREFERRED_METRICS:
        if metric in columns:
            metrics[metric] = metric

    for col in columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in metrics.values():
            metrics[col] = col

    return schema, metrics


def add_derived_fields(df, schema, metrics):
    df = df.copy()

    if schema["date"] and schema["date"] in df.columns:
        df[schema["date"]] = pd.to_datetime(df[schema["date"]], errors="coerce")
    elif {"month", "day"}.issubset(set(df.columns)):
        df["_derived_date"] = pd.to_datetime(
            {"year": 2025, "month": pd.to_numeric(df["month"], errors="coerce"), "day": pd.to_numeric(df["day"], errors="coerce")},
            errors="coerce",
        )
        schema["date"] = "_derived_date"

    if schema["page"] and schema["page"] in df.columns:
        page_text = df[schema["page"]].astype(str).str.lower()
        df["_is_rural_development"] = page_text.str.contains("rural development", na=False)
    elif schema["path"] and schema["path"] in df.columns:
        path_text = df[schema["path"]].astype(str).str.lower()
        df["_is_rural_development"] = path_text.str.contains("rural|rd", regex=True, na=False)
    else:
        df["_is_rural_development"] = False

    if "engagement_rate" not in metrics:
        bounce_col = metrics.get("bounce_rate")
        if bounce_col and bounce_col in df.columns:
            df["_engagement_proxy"] = 1 - pd.to_numeric(df[bounce_col], errors="coerce")
            metrics["_engagement_proxy"] = "_engagement_proxy"

    if "sessions" in metrics and "event_count" in metrics:
        sessions = pd.to_numeric(df[metrics["sessions"]], errors="coerce").replace(0, np.nan)
        events = pd.to_numeric(df[metrics["event_count"]], errors="coerce")
        df["_events_per_session"] = events / sessions
        metrics["_events_per_session"] = "_events_per_session"

    if "returning_users" in metrics and "total_users" in metrics:
        total = pd.to_numeric(df[metrics["total_users"]], errors="coerce").replace(0, np.nan)
        returning = pd.to_numeric(df[metrics["returning_users"]], errors="coerce")
        df["_returning_user_rate"] = returning / total
        metrics["_returning_user_rate"] = "_returning_user_rate"

    return df, schema, metrics


def safe_sum(df, col):
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col else np.nan


def safe_mean(df, col):
    return float(pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).mean()) if col else np.nan


def format_number(value, rate=False, seconds=False):
    if value is None or pd.isna(value):
        return "N/A"
    if rate:
        return f"{value:.1%}"
    if seconds:
        return f"{value:,.1f}s"
    if abs(value) >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value/1_000:.1f}K"
    return f"{value:,.0f}" if float(value).is_integer() else f"{value:,.2f}"


def metric_card(label, value, note="", rate=False, seconds=False):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{format_number(value, rate=rate, seconds=seconds)}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def aggregate_by(df, group_cols, metrics):
    agg_map = {}
    for name, col in metrics.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if name in ["bounce_rate", "views_per_session", "average_session_duration", "_engagement_proxy", "_events_per_session", "_returning_user_rate"]:
                agg_map[col] = "mean"
            else:
                agg_map[col] = "sum"
    if not group_cols or not agg_map:
        return pd.DataFrame()
    return df.groupby(group_cols, dropna=False).agg(agg_map).reset_index()


def make_csv_download(df):
    return df.to_csv(index=False).encode("utf-8")


def choose_metric(metrics, default="sessions"):
    keys = list(metrics.keys())
    if not keys:
        return None
    return default if default in keys else keys[0]


def derive_clusters(df, schema, metrics, k=4):
    if schema["cluster"] and schema["cluster"] in df.columns:
        out = df.copy()
        out["_segment_label"] = out[schema["cluster"]].astype(str)
        return out, "Existing cluster or segment field detected in the data."

    if not SKLEARN_AVAILABLE:
        out = df.copy()
        out["_segment_label"] = "Clustering unavailable"
        return out, "scikit-learn is not available, so cluster derivation was skipped."

    feature_candidates = [
        metrics.get("sessions"),
        metrics.get("active_users"),
        metrics.get("event_count"),
        metrics.get("views_per_session"),
        metrics.get("average_session_duration"),
        metrics.get("bounce_rate"),
        metrics.get("exits"),
        metrics.get("returning_users"),
        metrics.get("total_users"),
        metrics.get("_events_per_session"),
        metrics.get("_returning_user_rate"),
    ]
    feature_cols = [c for c in feature_candidates if c and c in df.columns]
    feature_cols = list(dict.fromkeys(feature_cols))

    if len(feature_cols) < 2 or len(df) < 8:
        out = df.copy()
        out["_segment_label"] = "Insufficient data for clustering"
        return out, "Not enough numeric fields or records were available to derive clusters."

    page_col = schema.get("page")
    country_col = schema.get("country")
    device_col = schema.get("device")
    group_cols = [c for c in [page_col, country_col, device_col] if c and c in df.columns]

    if group_cols:
        base = aggregate_by(df, group_cols, {c: c for c in feature_cols})
    else:
        base = df[feature_cols].copy()
        base["_row_id"] = np.arange(len(base))
        group_cols = ["_row_id"]

    X = base[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    n_clusters = max(2, min(int(k), len(X)))
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    base["_derived_cluster_id"] = model.fit_predict(X_scaled)

    if "sessions" in metrics and metrics["sessions"] in base.columns:
        volume_col = metrics["sessions"]
    elif "active_users" in metrics and metrics["active_users"] in base.columns:
        volume_col = metrics["active_users"]
    else:
        volume_col = feature_cols[0]

    cluster_summary = base.groupby("_derived_cluster_id")[feature_cols].mean(numeric_only=True)
    ranked = cluster_summary[volume_col].rank(method="dense", ascending=False).astype(int).to_dict()

    def label_cluster(cluster_id):
        rank = ranked.get(cluster_id, cluster_id + 1)
        engagement_col = metrics.get("_engagement_proxy") or metrics.get("views_per_session") or metrics.get("average_session_duration")
        if engagement_col in cluster_summary.columns:
            engagement_rank = cluster_summary[engagement_col].rank(method="dense", ascending=False).astype(int).to_dict().get(cluster_id, 0)
            return f"Derived Segment {rank}: volume rank {rank}, engagement rank {engagement_rank}"
        return f"Derived Segment {rank}"

    base["_segment_label"] = base["_derived_cluster_id"].map(label_cluster)

    out = df.merge(base[group_cols + ["_segment_label"]], on=group_cols, how="left")
    out["_segment_label"] = out["_segment_label"].fillna("Unassigned")
    return out, "Derived reproducible KMeans segments using available numeric traffic and engagement features."


def get_top_share(df, metric_col, page_col, n=10):
    if not metric_col or not page_col or page_col not in df.columns:
        return np.nan
    grouped = df.groupby(page_col)[metric_col].sum(numeric_only=True).sort_values(ascending=False)
    total = grouped.sum()
    if total == 0 or pd.isna(total):
        return np.nan
    return float(grouped.head(n).sum() / total)


def build_insights(df, schema, metrics):
    insights = []
    primary = metrics.get("sessions") or metrics.get("active_users") or metrics.get("total_users") or choose_metric(metrics)
    page_col = schema.get("page")
    device_col = schema.get("device")
    country_col = schema.get("country")

    if primary and page_col and page_col in df.columns and primary in df.columns:
        page_summary = df.groupby(page_col)[primary].sum(numeric_only=True).sort_values(ascending=False)
        if not page_summary.empty and page_summary.sum() > 0:
            top_page = page_summary.index[0]
            top_share = page_summary.iloc[0] / page_summary.sum()
            insights.append(f"Highest-activity page benchmark is **{top_page}**, accounting for **{top_share:.1%}** of the filtered {primary.replace('_', ' ')}.")

    if primary and device_col and device_col in df.columns and primary in df.columns:
        device_summary = df.groupby(device_col)[primary].sum(numeric_only=True).sort_values(ascending=False)
        if len(device_summary) > 0 and device_summary.sum() > 0:
            insights.append(f"Top device category is **{device_summary.index[0]}**, representing **{device_summary.iloc[0] / device_summary.sum():.1%}** of filtered activity.")

    if "_is_rural_development" in df.columns and primary and primary in df.columns:
        rural_total = safe_sum(df[df["_is_rural_development"]], primary)
        total = safe_sum(df, primary)
        if total > 0:
            insights.append(f"Rural Development represents **{rural_total / total:.1%}** of filtered {primary.replace('_', ' ')}.")

    engagement_col = metrics.get("_engagement_proxy") or metrics.get("views_per_session") or metrics.get("average_session_duration")
    if engagement_col and page_col and page_col in df.columns and engagement_col in df.columns:
        eligible = df.groupby(page_col).agg({engagement_col: "mean", primary: "sum" if primary else "count"}).dropna()
        if not eligible.empty and primary in eligible.columns:
            threshold = eligible[primary].quantile(0.60)
            eligible = eligible[eligible[primary] >= threshold]
        if not eligible.empty:
            top_engaged = eligible[engagement_col].idxmax()
            insights.append(f"Among higher-volume pages, **{top_engaged}** has the strongest average engagement signal based on **{engagement_col.replace('_', ' ')}**.")

    if country_col and country_col in df.columns and primary and primary in df.columns:
        countries = df.groupby(country_col)[primary].sum(numeric_only=True).sort_values(ascending=False)
        if len(countries) > 1 and countries.sum() > 0:
            insights.append(f"Geographic activity is led by **{countries.index[0]}**, followed by **{countries.index[1]}**.")

    if not insights:
        insights.append("The filtered data does not contain enough recognized metrics or dimensions to generate evidence-based insights.")
    return insights[:5]


def chart_layout(fig, height=430):
    fig.update_layout(
        height=height,
        margin=dict(l=20, r=20, t=55, b=25),
        legend_title_text="",
        hovermode="closest",
    )
    return fig


def show_warning(message):
    st.markdown(f"<div class='warning-box'>{message}</div>", unsafe_allow_html=True)


df_raw, inventory = load_project_data()

st.title("USDA Digital Pathway & AI Insights Suite")
st.caption(
    "Executive-ready Streamlit dashboard for two analytical layers: USDA-wide descriptive web analytics and focused Rural Development clustering/profile analysis."
)

if df_raw.empty:
    st.error(
        "No compatible data was loaded. Place CSV, Excel, or Parquet files inside the local folder "
        "`organized_clean_long_data_full_USDA`, or provide a same-named workbook/file beside `app.py`."
    )
    if not inventory.empty:
        st.subheader("Detected file inventory")
        st.dataframe(inventory, use_container_width=True)
    st.stop()

schema, metrics = detect_schema(df_raw)
df, schema, metrics = add_derived_fields(df_raw, schema, metrics)

with st.sidebar:
    st.header("Interactive Controls")

    cluster_k = st.slider("Derived cluster count if needed", min_value=2, max_value=8, value=4, step=1)
    df, cluster_note = derive_clusters(df, schema, metrics, k=cluster_k)

    page_col = schema.get("page")
    country_col = schema.get("country")
    device_col = schema.get("device")
    date_col = schema.get("date")
    primary_metric = choose_metric(metrics)

    selected_metric = st.selectbox("Primary metric", options=list(metrics.keys()), index=list(metrics.keys()).index(primary_metric) if primary_metric in metrics else 0)
    metric_col = metrics[selected_metric]

    top_n = st.slider("Top N", min_value=5, max_value=30, value=10, step=1)

    filtered = df.copy()

    if date_col and date_col in filtered.columns and pd.api.types.is_datetime64_any_dtype(filtered[date_col]):
        min_date = filtered[date_col].min()
        max_date = filtered[date_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            date_range = st.date_input("Date range", value=(min_date.date(), max_date.date()))
            if isinstance(date_range, tuple) and len(date_range) == 2:
                start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
                filtered = filtered[(filtered[date_col] >= start) & (filtered[date_col] <= end)]

    if page_col and page_col in filtered.columns:
        page_search = st.text_input("Search page benchmark or title", "")
        if page_search:
            filtered = filtered[filtered[page_col].astype(str).str.contains(page_search, case=False, na=False)]
        page_options = sorted(filtered[page_col].dropna().astype(str).unique().tolist())
        selected_pages = st.multiselect("Page benchmark filter", page_options, default=[])
        if selected_pages:
            filtered = filtered[filtered[page_col].astype(str).isin(selected_pages)]

    if country_col and country_col in filtered.columns:
        country_options = sorted(filtered[country_col].dropna().astype(str).unique().tolist())
        selected_countries = st.multiselect("Country filter", country_options, default=[])
        if selected_countries:
            filtered = filtered[filtered[country_col].astype(str).isin(selected_countries)]

    if device_col and device_col in filtered.columns:
        device_options = sorted(filtered[device_col].dropna().astype(str).unique().tolist())
        selected_devices = st.multiselect("Device category filter", device_options, default=[])
        if selected_devices:
            filtered = filtered[filtered[device_col].astype(str).isin(selected_devices)]

    segment_options = sorted(filtered["_segment_label"].dropna().astype(str).unique().tolist()) if "_segment_label" in filtered.columns else []
    selected_segments = st.multiselect("Segment / cluster filter", segment_options, default=[])
    if selected_segments:
        filtered = filtered[filtered["_segment_label"].astype(str).isin(selected_segments)]

    rural_only = st.toggle("Rural Development only", value=False)
    if rural_only:
        filtered = filtered[filtered["_is_rural_development"]]

    compare_rural = st.toggle("Show Rural vs USDA benchmarks", value=True)


if filtered.empty:
    st.warning("The current filters returned no rows. Adjust the sidebar controls to restore data.")
    st.stop()

tabs = st.tabs(
    [
        "Executive Overview",
        "USDA-wide Web Analytics",
        "Rural Development Cluster Profile",
        "Page Benchmark Comparison",
        "Data Quality / Methodology",
    ]
)

sessions_col = metrics.get("sessions")
users_col = metrics.get("active_users") or metrics.get("total_users")
engagement_col = metrics.get("_engagement_proxy") or metrics.get("views_per_session") or metrics.get("average_session_duration")
rural_activity = safe_sum(filtered[filtered["_is_rural_development"]], metric_col) if metric_col in filtered.columns else np.nan
total_activity = safe_sum(filtered, metric_col) if metric_col in filtered.columns else np.nan
top_share = get_top_share(filtered, metric_col, page_col, n=top_n)
cluster_count = filtered["_segment_label"].nunique() if "_segment_label" in filtered.columns else np.nan

with tabs[0]:
    st.subheader("Executive Overview")
    st.write(
        "Use this view to monitor overall USDA web usage, identify concentration patterns, and assess how much filtered activity is tied to Rural Development."
    )

    kpi_cols = st.columns(6)
    with kpi_cols[0]:
        metric_card("Total sessions", safe_sum(filtered, sessions_col), "Sum of sessions when available.")
    with kpi_cols[1]:
        metric_card("Users / visitors", safe_sum(filtered, users_col), "Uses active users, otherwise total users.")
    with kpi_cols[2]:
        metric_card("Engagement proxy", safe_mean(filtered, engagement_col), "1 - bounce rate when available.", rate=engagement_col == "_engagement_proxy")
    with kpi_cols[3]:
        metric_card(f"Top {top_n} page share", top_share, f"Share of {selected_metric.replace('_', ' ')}.", rate=True)
    with kpi_cols[4]:
        metric_card("Segments", cluster_count, "Existing labels or derived KMeans segments.")
    with kpi_cols[5]:
        metric_card("Rural Development share", rural_activity / total_activity if total_activity else np.nan, f"Share of {selected_metric.replace('_', ' ')}.", rate=True)

    st.markdown("#### Key Insights")
    for insight in build_insights(filtered, schema, metrics):
        st.markdown(f"<div class='insight-box'>{insight}</div>", unsafe_allow_html=True)

    left, right = st.columns([1.25, 1])
    with left:
        if date_col and date_col in filtered.columns and metric_col in filtered.columns:
            trend = aggregate_by(filtered.dropna(subset=[date_col]), [date_col], {selected_metric: metric_col})
            if not trend.empty:
                fig = px.line(trend.sort_values(date_col), x=date_col, y=metric_col, markers=True, title=f"{selected_metric.replace('_', ' ').title()} Trend")
                st.plotly_chart(chart_layout(fig), use_container_width=True)
            else:
                show_warning("Trend chart could not be computed because date or metric values are missing.")
        else:
            show_warning("No valid date field was detected, so the trend chart is unavailable.")

    with right:
        if page_col and page_col in filtered.columns and metric_col in filtered.columns:
            top_pages = filtered.groupby(page_col, dropna=False)[metric_col].sum(numeric_only=True).sort_values(ascending=False).head(top_n).reset_index()
            fig = px.bar(top_pages, x=metric_col, y=page_col, orientation="h", title=f"Top {top_n} Page Benchmarks")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
        else:
            show_warning("Top page chart requires a page benchmark/title and a numeric metric.")

    st.download_button("Download filtered data", make_csv_download(filtered), "usda_filtered_data.csv", "text/csv")

with tabs[1]:
    st.subheader("USDA-wide Web Analytics")
    st.write("System-wide descriptive analysis of the filtered USDA public website activity.")

    chart_cols = st.columns(2)

    with chart_cols[0]:
        if device_col and device_col in filtered.columns and metric_col in filtered.columns:
            device_summary = filtered.groupby(device_col)[metric_col].sum(numeric_only=True).sort_values(ascending=False).reset_index()
            fig = px.bar(device_summary, x=device_col, y=metric_col, title=f"{selected_metric.replace('_', ' ').title()} by Device Category")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
        else:
            show_warning("Device chart is unavailable because device category or selected metric is missing.")

    with chart_cols[1]:
        if country_col and country_col in filtered.columns and metric_col in filtered.columns:
            country_summary = filtered.groupby(country_col)[metric_col].sum(numeric_only=True).sort_values(ascending=False).head(top_n).reset_index()
            fig = px.bar(country_summary, x=metric_col, y=country_col, orientation="h", title=f"Top {top_n} Countries")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig), use_container_width=True)
        else:
            show_warning("Country chart is unavailable because country or selected metric is missing.")

    if page_col and device_col and page_col in filtered.columns and device_col in filtered.columns and metric_col in filtered.columns:
        matrix = filtered.pivot_table(index=page_col, columns=device_col, values=metric_col, aggfunc="sum", fill_value=0)
        matrix = matrix.loc[matrix.sum(axis=1).sort_values(ascending=False).head(top_n).index]
        fig = px.imshow(matrix, aspect="auto", title=f"Page Benchmark x Device Matrix: {selected_metric.replace('_', ' ').title()}")
        st.plotly_chart(chart_layout(fig, height=520), use_container_width=True)
    else:
        show_warning("Heatmap requires page benchmark, device category, and the selected numeric metric.")

    if page_col and metric_col in filtered.columns:
        scatter_df = aggregate_by(
            filtered,
            [page_col],
            {
                selected_metric: metric_col,
                "sessions": sessions_col or metric_col,
                "engagement": engagement_col or metric_col,
                "users": users_col or metric_col,
            },
        )
        if not scatter_df.empty:
            y_col = engagement_col if engagement_col in scatter_df.columns else metric_col
            size_col = users_col if users_col in scatter_df.columns else metric_col
            fig = px.scatter(
                scatter_df,
                x=metric_col,
                y=y_col,
                size=size_col,
                hover_name=page_col,
                title="Traffic, Engagement, and Audience Size by Page Benchmark",
            )
            st.plotly_chart(chart_layout(fig), use_container_width=True)

with tabs[2]:
    st.subheader("Rural Development Cluster Profile")
    st.write("Focused segment/profile analysis for Rural Development, using existing segment labels when present or derived KMeans segments when needed.")
    st.caption(cluster_note)

    rural_df = filtered[filtered["_is_rural_development"]].copy()
    if rural_df.empty:
        show_warning("No Rural Development rows are present under the current filters. Clear filters or check whether the page/title field contains Rural Development.")
    else:
        profile_metrics = {k: v for k, v in metrics.items() if v in rural_df.columns and pd.api.types.is_numeric_dtype(rural_df[v])}
        segment_summary = aggregate_by(rural_df, ["_segment_label"], profile_metrics)

        if not segment_summary.empty:
            st.dataframe(segment_summary, use_container_width=True)

            if metric_col in segment_summary.columns:
                fig = px.bar(segment_summary.sort_values(metric_col, ascending=False), x="_segment_label", y=metric_col, title="Rural Development Segment Size")
                st.plotly_chart(chart_layout(fig), use_container_width=True)

            profile_cols = [
                c for c in [
                    metrics.get("sessions"),
                    metrics.get("active_users"),
                    metrics.get("event_count"),
                    metrics.get("views_per_session"),
                    metrics.get("average_session_duration"),
                    metrics.get("bounce_rate"),
                    metrics.get("_events_per_session"),
                    metrics.get("_returning_user_rate"),
                ]
                if c and c in segment_summary.columns
            ]
            if len(profile_cols) >= 2:
                radar_base = segment_summary[["_segment_label"] + profile_cols].copy()
                long_profile = radar_base.melt(id_vars="_segment_label", var_name="metric", value_name="value")
                long_profile["metric"] = long_profile["metric"].str.replace("_", " ").str.title()
                fig = px.line_polar(long_profile, r="value", theta="metric", color="_segment_label", line_close=True, title="Cluster Profile Shape")
                st.plotly_chart(chart_layout(fig, height=560), use_container_width=True)

            if page_col and page_col in rural_df.columns and metric_col in rural_df.columns:
                seg_page = rural_df.groupby(["_segment_label", page_col])[metric_col].sum(numeric_only=True).reset_index()
                top_segment_pages = seg_page.sort_values(metric_col, ascending=False).groupby("_segment_label").head(5)
                fig = px.bar(
                    top_segment_pages,
                    x=metric_col,
                    y=page_col,
                    color="_segment_label",
                    orientation="h",
                    title="Top Rural Development Page Patterns by Segment",
                )
                fig.update_yaxes(autorange="reversed")
                st.plotly_chart(chart_layout(fig, height=560), use_container_width=True)

            st.markdown("#### Action-oriented Segment Notes")
            for _, row in segment_summary.iterrows():
                seg_name = row["_segment_label"]
                seg_volume = row[metric_col] if metric_col in row else np.nan
                engagement_value = row[engagement_col] if engagement_col in row else np.nan
                st.markdown(
                    f"<div class='insight-box'><b>{seg_name}</b><br>"
                    f"Filtered volume: <b>{format_number(seg_volume)}</b>. "
                    f"Engagement signal: <b>{format_number(engagement_value, rate=engagement_col == '_engagement_proxy')}</b>. "
                    "Use this segment to prioritize chatbot or guided-navigation support when it combines high volume with weaker engagement or high exit/bounce signals.</div>",
                    unsafe_allow_html=True,
                )

        st.download_button("Download Rural Development profile data", make_csv_download(rural_df), "rural_development_profile.csv", "text/csv")

with tabs[3]:
    st.subheader("Page Benchmark Comparison")
    st.write("Compare Rural Development against broader USDA patterns using the selected metric and available engagement fields.")

    if compare_rural and metric_col in filtered.columns:
        comparison_df = filtered.copy()
        comparison_df["_benchmark_group"] = np.where(comparison_df["_is_rural_development"], "Rural Development", "Other USDA")
        comp = aggregate_by(comparison_df, ["_benchmark_group"], {**metrics, selected_metric: metric_col})
        if not comp.empty:
            st.dataframe(comp, use_container_width=True)
            fig = px.bar(comp, x="_benchmark_group", y=metric_col, title=f"Rural Development vs Other USDA: {selected_metric.replace('_', ' ').title()}")
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        if page_col and page_col in filtered.columns:
            benchmark = filtered.groupby(page_col)[metric_col].sum(numeric_only=True).sort_values(ascending=False).head(top_n).reset_index()
            benchmark["_is_rural_development"] = benchmark[page_col].astype(str).str.lower().str.contains("rural development", na=False)
            fig = px.bar(
                benchmark,
                x=metric_col,
                y=page_col,
                color="_is_rural_development",
                orientation="h",
                title=f"Top {top_n} Page Benchmarks with Rural Development Flag",
            )
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(chart_layout(fig, height=520), use_container_width=True)

    if page_col and metric_col in filtered.columns:
        detail_cols = [c for c in [page_col, country_col, device_col, date_col, "_segment_label", metric_col, engagement_col, sessions_col, users_col] if c and c in filtered.columns]
        detail = filtered[detail_cols].copy()
        st.markdown("#### Detailed filtered benchmark table")
        st.dataframe(detail.head(5000), use_container_width=True)
        st.download_button("Download benchmark table", make_csv_download(detail), "page_benchmark_comparison.csv", "text/csv")
    else:
        show_warning("Page benchmark comparison requires a page/title field and selected numeric metric.")

with tabs[4]:
    st.subheader("Data Quality / Methodology")

    st.markdown("#### Data Inventory")
    st.dataframe(inventory, use_container_width=True)

    st.markdown("#### Detected Schema")
    schema_table = pd.DataFrame(
        [{"field_role": key, "detected_column": value if value else "Not detected"} for key, value in schema.items()]
    )
    st.dataframe(schema_table, use_container_width=True)

    st.markdown("#### Available Metrics")
    metric_table = pd.DataFrame([{"metric_name": key, "source_column": value} for key, value in metrics.items()])
    st.dataframe(metric_table, use_container_width=True)

    st.markdown("#### Dataset Profile")
    profile = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "missing_count": [int(df[c].isna().sum()) for c in df.columns],
            "missing_rate": [float(df[c].isna().mean()) for c in df.columns],
            "unique_values": [int(df[c].nunique(dropna=True)) for c in df.columns],
        }
    )
    st.dataframe(profile, use_container_width=True)

    duplicate_count = int(df.duplicated().sum())
    st.info(
        f"Rows loaded after basic cleaning: {len(df):,}. Columns: {len(df.columns):,}. "
        f"Duplicate rows still present after load-level de-duplication check: {duplicate_count:,}."
    )

    st.markdown(
        """
        #### Methodology Notes
        - Columns are normalized by stripping whitespace, lowercasing, removing punctuation, and replacing spaces with underscores.
        - Numeric fields are detected dynamically and converted when most values can be parsed as numbers.
        - Date fields are detected dynamically; when month and day exist without a date, the app derives a 2025 date because the project context is 2025 web analytics.
        - Rural Development rows are identified from page benchmark/title text when available.
        - Engagement proxy uses `1 - bounce_rate` when bounce rate exists. Otherwise, the app falls back to available engagement-like metrics such as views per session or average session duration.
        - Segment labels are used directly if present. If absent, reproducible KMeans clustering is derived from available numeric traffic and engagement features.
        - Any missing chart or KPI is intentionally shown as a warning instead of using fabricated values.
        """
    )

    st.download_button("Download full cleaned combined data", make_csv_download(df), "usda_cleaned_combined_data.csv", "text/csv")
