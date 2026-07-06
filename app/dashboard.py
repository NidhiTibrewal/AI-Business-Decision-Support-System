"""
app/dashboard.py
=================
Streamlit dashboard: Overview, Forecast, Segmentation, Allocation (with
scenario sliders), and Ask-the-Analyst — the five pages from the project spec.

Run locally:
    streamlit run app/dashboard.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from module1_data_foundation import build_feature_table
from module2_eda_kpis import compute_kpis, generate_insights, weekly_revenue_series, category_leaderboard
from module3_forecasting import train_final_model, predict_with_interval, FEATURES
from module4_segmentation import build_store_features, cluster_stores, label_clusters
from module5_optimization import compute_effective_demand, optimize_allocation, run_scenario, CostParams
from module6_explainability import build_explainer, explain_row, format_driver_summary, HAS_SHAP
from module7_llm_layer import explain_decision, executive_summary
from module8_monitoring import check_mae_threshold

st.set_page_config(page_title="Inventory Allocation Platform", layout="wide")


@st.cache_data
def load_data():
    return build_feature_table(save=False)


@st.cache_resource
def load_model(df):
    return train_final_model(df)


df = load_data()
model, residuals = load_model(df)

page = st.sidebar.radio(
    "Page", ["Overview", "Forecast", "Segmentation", "Allocation", "Ask the Analyst"]
)

# ---------------------------------------------------------------------------
# OVERVIEW
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Overview")
    kpis = compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total revenue", f"₹{kpis['total_revenue']:,.0f}")
    c2.metric("Revenue trend (Q1→Q4)", f"{kpis['revenue_trend_pct_first_vs_last_quarter']:+.1f}%")
    c3.metric("Top-20% series revenue share", f"{kpis['top20pct_series_revenue_share_pct']:.1f}%")
    c4.metric("Avg demand volatility (CV)", f"{kpis['avg_demand_volatility_cv']:.2f}")

    st.subheader("Weekly revenue trend")
    rev = weekly_revenue_series(df)
    st.plotly_chart(px.line(rev, x="week", y="weekly_sales"), use_container_width=True)

    st.subheader("Revenue by category")
    cat = category_leaderboard(df)
    st.plotly_chart(px.bar(cat, x="category_id", y="weekly_sales"), use_container_width=True)

    st.subheader("Business insights")
    for line in generate_insights(df):
        st.write(line)

# ---------------------------------------------------------------------------
# FORECAST
# ---------------------------------------------------------------------------
elif page == "Forecast":
    st.title("Demand Forecast")
    store_id = st.selectbox("Store", sorted(df["store_id"].unique()))
    category_id = st.selectbox("Category", sorted(df["category_id"].unique()))

    series = df[(df.store_id == store_id) & (df.category_id == category_id)].sort_values("week")
    latest_row = series.tail(1)

    fc = predict_with_interval(model, residuals, latest_row[FEATURES])
    forecast_val = fc["forecast"].iloc[0]
    lower, upper = fc["lower_bound"].iloc[0], fc["upper_bound"].iloc[0]
    confidence = fc["confidence_score"].iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Next-week forecast", f"{forecast_val:,.0f} units")
    c2.metric("90% interval", f"[{lower:,.0f}, {upper:,.0f}]")
    c3.metric("Confidence score", f"{confidence:.0%}")

    st.plotly_chart(
        px.line(series, x="week", y="weekly_sales", title="Historical sales"), use_container_width=True
    )

    background = df[FEATURES].sample(min(200, len(df)), random_state=42)
    explainer = build_explainer(model, background)
    shap_vals = explainer(latest_row[FEATURES])
    drivers = explain_row(shap_vals[0], FEATURES, latest_row[FEATURES].iloc[0],
                           base_prediction=forecast_val, top_k=4)
    st.subheader("Why this forecast")
    st.code(format_driver_summary(drivers))
    st.caption(f"Explainability backend: {'SHAP' if HAS_SHAP else 'permutation-importance fallback'}")

# ---------------------------------------------------------------------------
# SEGMENTATION
# ---------------------------------------------------------------------------
elif page == "Segmentation":
    st.title("Store Segmentation")
    store_feats = build_store_features(df)
    clustered, meta = cluster_stores(store_feats)
    labels = label_clusters(clustered)

    st.plotly_chart(
        px.scatter(clustered, x="pca_1", y="pca_2", color=clustered["cluster"].astype(str),
                   hover_data=["store_id", "avg_weekly_sales", "volatility_cv"],
                   title=f"Store clusters (PCA, {sum(meta['explained_variance_ratio']):.0%} variance explained)"),
        use_container_width=True,
    )

    st.subheader("Cluster descriptions")
    for cid, info in labels.items():
        st.write(f"**Cluster {cid}** ({info['n_stores']} stores) — {info['description']}")

    st.dataframe(clustered[["store_id", "cluster", "avg_weekly_sales", "volatility_cv", "margin_proxy"]])

# ---------------------------------------------------------------------------
# ALLOCATION (with scenario sliders)
# ---------------------------------------------------------------------------
elif page == "Allocation":
    st.title("Inventory Allocation — Scenario Analysis")

    latest_week = df["week"].max()
    latest = df[df["week"] == latest_week].reset_index(drop=True)
    fc_all = predict_with_interval(model, residuals, latest[FEATURES])
    fc_all["store_id"] = latest["store_id"].values

    store_forecast = fc_all.groupby("store_id").agg(
        forecast=("forecast", "sum"), lower_bound=("lower_bound", "sum"),
        confidence_score=("confidence_score", "mean"),
    ).reset_index()
    rng = np.random.default_rng(7)
    store_forecast["distance_km"] = rng.uniform(5, 80, len(store_forecast)).round(1)

    st.sidebar.subheader("Scenario controls")
    warehouse_cap = st.sidebar.slider("Warehouse capacity", 10_000, 100_000, 50_000, step=5_000)
    per_store_cap = st.sidebar.slider("Per-store capacity", 500, 10_000, 5_000, step=500)
    holding_cost = st.sidebar.slider("Holding cost / unit / week (₹)", 0.5, 10.0, 2.0, step=0.5)
    transport_cost = st.sidebar.slider("Transport cost / km (₹)", 0.1, 3.0, 0.6, step=0.1)
    penalty = st.sidebar.slider("Lost-sale penalty / unit (₹)", 5.0, 50.0, 18.0, step=1.0)
    budget_cut = st.sidebar.slider("Budget cut vs. base (%)", 0, 50, 0, step=5)

    params = CostParams(
        holding_cost_per_unit=holding_cost, transport_cost_per_km=transport_cost,
        lost_sale_penalty_per_unit=penalty, warehouse_capacity=int(warehouse_cap * (1 - budget_cut / 100)),
        per_store_capacity=per_store_cap,
    )

    effective_demand = compute_effective_demand(store_forecast)
    result = optimize_allocation(
        store_forecast["store_id"], effective_demand.values,
        store_forecast["distance_km"].values, params,
    )

    st.write(f"Solver status: **{result.attrs['solver_status']}** (backend: {result.attrs['solver_backend']})")
    st.plotly_chart(
        px.bar(result.sort_values("allocated_units", ascending=False),
               x="store_id", y=["allocated_units", "projected_shortfall"], barmode="group"),
        use_container_width=True,
    )
    st.dataframe(result.sort_values("allocated_units", ascending=False))

    mae_check = check_mae_threshold(latest["weekly_sales"].values, fc_all["forecast"].values, threshold=400)
    if mae_check["flagged"]:
        st.warning(f"Model drift flag: rolling MAE {mae_check['latest_rolling_mae']} exceeds threshold {mae_check['threshold']}")
    else:
        st.success(f"Model tracking within threshold (rolling MAE: {mae_check['latest_rolling_mae']})")

# ---------------------------------------------------------------------------
# ASK THE ANALYST
# ---------------------------------------------------------------------------
elif page == "Ask the Analyst":
    st.title("Ask the Analyst")
    st.caption("Templated, grounded workflows — the LLM explains real numbers, it doesn't generate them.")

    workflow = st.selectbox(
        "Choose a question type",
        ["Executive Summary", "Explain a store's allocation"],
    )

    if workflow == "Executive Summary":
        if st.button("Generate summary"):
            kpis = compute_kpis(df)
            insights = generate_insights(df)
            st.write(executive_summary(kpis, insights, {}))

    else:
        store_id = st.selectbox("Store", sorted(df["store_id"].unique()))
        if st.button("Explain allocation"):
            latest_week = df["week"].max()
            latest = df[(df["week"] == latest_week) & (df["store_id"] == store_id)]
            fc = predict_with_interval(model, residuals, latest[FEATURES])
            demo_allocation = {
                "store_id": int(store_id),
                "forecast": float(fc["forecast"].sum()),
                "confidence_score": float(fc["confidence_score"].mean()),
            }
            st.write(explain_decision(store_id, demo_allocation, []))
