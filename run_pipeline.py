"""
Runs the full decision path end-to-end, module by module, and prints a
readable trace of each stage's output. 
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd

from module1_data_foundation import build_feature_table
from module2_eda_kpis import compute_kpis, test_promotion_effect, test_holiday_effect, generate_insights
from module3_forecasting import evaluate_walk_forward, train_final_model, predict_with_interval, FEATURES
from module4_segmentation import build_store_features, cluster_stores, label_clusters
from module5_optimization import compute_effective_demand, optimize_allocation, run_scenario, CostParams
from module6_explainability import build_explainer, explain_row, format_driver_summary, HAS_SHAP
from module7_llm_layer import explain_decision
from module8_monitoring import log_run, check_mae_threshold


def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main():
    # ---- Module 1: Data Foundation ----------------------------------------
    section("MODULE 1 — Data Foundation")
    df = build_feature_table(save=True)
    print(f"Feature table: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # ---- Module 2: EDA + KPIs ----------------------------------------------
    section("MODULE 2 — Understand the Business (EDA + KPIs)")
    kpis = compute_kpis(df)
    promo_test = test_promotion_effect(df)
    holiday_test = test_holiday_effect(df)
    insights = generate_insights(df)
    print("KPIs:", kpis)
    print("\nInsights:")
    for line in insights:
        print(" ", line)

    # ---- Module 3: Forecasting ---------------------------------------------
    section("MODULE 3 — Predict Future Demand (walk-forward validated)")
    cv = evaluate_walk_forward(df, n_splits=3, test_weeks=4)
    print("Walk-forward CV, mean MAE/RMSE by model:")
    print(cv.groupby("model")[["mae", "rmse"]].mean().round(2))

    model, residuals = train_final_model(df)
    latest_week = df["week"].max()
    latest = df[df["week"] == latest_week].reset_index(drop=True)
    forecast_df = predict_with_interval(model, residuals, latest[FEATURES])
    forecast_df["store_id"] = latest["store_id"].values
    forecast_df["category_id"] = latest["category_id"].values
    print(f"\nLatest week forecasts (n={len(forecast_df)}), sample:")
    print(forecast_df[["store_id", "category_id", "forecast", "lower_bound", "upper_bound", "confidence_score"]].head())

    # ---- Module 4: Segmentation ---------------------------------------------
    section("MODULE 4 — Understand Store Behavior (Segmentation)")
    store_feats = build_store_features(df)
    clustered, cluster_meta = cluster_stores(store_feats)
    cluster_labels = label_clusters(clustered)
    for cid, info in cluster_labels.items():
        print(f"Cluster {cid} ({info['n_stores']} stores): {info['description']}")

    # ---- Module 5: Optimization (aggregate demand to store level first) ----
    section("MODULE 5 — Recommend Inventory Allocation (confidence-weighted LP)")
    store_forecast = forecast_df.groupby("store_id").agg(
        forecast=("forecast", "sum"),
        lower_bound=("lower_bound", "sum"),
        confidence_score=("confidence_score", "mean"),
    ).reset_index()

    rng = np.random.default_rng(10)
    store_forecast["distance_km"] = rng.uniform(5, 80, len(store_forecast)).round(1)

    effective_demand = compute_effective_demand(store_forecast)
    params = CostParams()  # defaults from the project spec
    allocation = optimize_allocation(
        store_forecast["store_id"], effective_demand.values,
        store_forecast["distance_km"].values, params,
    )
    print(f"Solver status: {allocation.attrs['solver_status']} (backend: {allocation.attrs['solver_backend']})")
    print(allocation.sort_values("allocated_units", ascending=False).head(8))

    print("\nScenario: 10% budget cut (implemented as reduced warehouse capacity for this demo)")
    scenario = run_scenario(
        store_forecast["store_id"],
        store_forecast.assign(effective_demand=effective_demand),
        store_forecast["distance_km"].values,
        params,
        warehouse_capacity=int(params.warehouse_capacity * 0.9),
    )
    print(scenario.sort_values("allocated_units", ascending=False).head(5))

    # ---- Module 6: Explainability -------------------------------------------
    section("MODULE 6 — Explain the Decision (SHAP-based drivers)")
    print(f"SHAP available: {HAS_SHAP}")
    background = df[FEATURES].sample(min(200, len(df)), random_state=42)
    explain = build_explainer(model, background)
    sample_row = latest[FEATURES].iloc[[0]]
    shap_vals = explain(sample_row)
    row_vals = shap_vals[0]
    pred = model.predict(sample_row)[0]
    drivers = explain_row(row_vals, FEATURES, sample_row.iloc[0], base_prediction=pred, top_k=4)
    print(format_driver_summary(drivers))

    # ---- Module 7: LLM layer ------------------------------------------------
    section("MODULE 7 — Explain the Recommendation (templated LLM workflow)")
    top_store = allocation.sort_values("allocated_units", ascending=False).iloc[0].to_dict()
    print(explain_decision(top_store["store_id"], top_store, drivers))

    # ---- Module 8: Monitoring -------------------------------------------------
    section("MODULE 8 — Monitoring & Tracking")
    log_run(
        model_name="xgboost_demand_forecast",
        params={"n_estimators": 300, "max_depth": 4},
        metrics={"mae": float(cv[cv.model == "tree_model"]["mae"].mean())},
        dataset_version="synthetic_v1",
    )
    mae_check = check_mae_threshold(
        latest["weekly_sales"].values, forecast_df["forecast"].values, threshold=400
    )
    print("Monitoring check:", mae_check)

    section("PIPELINE COMPLETE")
    print("All 8 analytical modules ran end-to-end. See README.md for the")
    print("dashboard (app/dashboard.py) and API (app/api.py) that sit on top of this.")


if __name__ == "__main__":
    main()
