"""
app/api.py
===========
FastAPI service exposing the pipeline as endpoints, per the project spec:

    GET  /health
    GET  /forecast?store_id=..&category_id=..
    POST /optimize        (accepts scenario parameters -> re-runs the LP)
    GET  /explain?store_id=..&category_id=..
    POST /ask              (routes to a Module 7 templated workflow)

Run locally:
    uvicorn app.api:app --reload --port 8000

Note: models are trained once at startup and cached in memory (`STATE`).
For a real deployment you'd load a serialized model from Module 8's
tracked runs instead of retraining on boot — kept simple here since the
point of this file is the API surface, not a model registry.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from module1_data_foundation import build_feature_table
from module3_forecasting import train_final_model, predict_with_interval, FEATURES
from module5_optimization import compute_effective_demand, optimize_allocation, CostParams
from module6_explainability import build_explainer, explain_row, HAS_SHAP
from module7_llm_layer import explain_decision, compare_stores, explain_forecast, executive_summary
from module2_eda_kpis import compute_kpis, generate_insights

app = FastAPI(title="Decision-Centric Inventory Allocation API", version="1.0")

STATE = {}


@app.on_event("startup")
def load_state():
    df = build_feature_table(save=False)
    model, residuals = train_final_model(df)
    background = df[FEATURES].sample(min(200, len(df)), random_state=42)
    explainer = build_explainer(model, background)

    STATE["df"] = df
    STATE["model"] = model
    STATE["residuals"] = residuals
    STATE["explainer"] = explainer


@app.get("/health")
def health():
    return {"status": "ok", "rows_loaded": int(len(STATE["df"])) if "df" in STATE else 0}


@app.get("/forecast")
def forecast(store_id: int, category_id: int):
    df = STATE["df"]
    row = df[(df.store_id == store_id) & (df.category_id == category_id)].sort_values("week").tail(1)
    if row.empty:
        raise HTTPException(404, "No data for that store/category")
    result = predict_with_interval(STATE["model"], STATE["residuals"], row[FEATURES])
    return result[["forecast", "lower_bound", "upper_bound", "confidence_score"]].iloc[0].to_dict()


class OptimizeRequest(BaseModel):
    store_ids: list
    effective_demand: list
    distances_km: list
    holding_cost_per_unit: Optional[float] = 2.0
    transport_cost_per_km: Optional[float] = 0.6
    lost_sale_penalty_per_unit: Optional[float] = 18.0
    warehouse_capacity: Optional[int] = 50_000
    per_store_capacity: Optional[int] = 5_000
    budget: Optional[float] = None


@app.post("/optimize")
def optimize(req: OptimizeRequest):
    params = CostParams(
        holding_cost_per_unit=req.holding_cost_per_unit,
        transport_cost_per_km=req.transport_cost_per_km,
        lost_sale_penalty_per_unit=req.lost_sale_penalty_per_unit,
        warehouse_capacity=req.warehouse_capacity,
        per_store_capacity=req.per_store_capacity,
        budget=req.budget,
    )
    result = optimize_allocation(req.store_ids, req.effective_demand, req.distances_km, params)
    return {
        "solver_status": result.attrs["solver_status"],
        "solver_backend": result.attrs["solver_backend"],
        "allocations": result.to_dict(orient="records"),
    }


@app.get("/explain")
def explain(store_id: int, category_id: int):
    df = STATE["df"]
    row = df[(df.store_id == store_id) & (df.category_id == category_id)].sort_values("week").tail(1)
    if row.empty:
        raise HTTPException(404, "No data for that store/category")

    X = row[FEATURES]
    shap_vals = STATE["explainer"](X)
    pred = STATE["model"].predict(X)[0]
    drivers = explain_row(shap_vals[0], FEATURES, X.iloc[0], base_prediction=pred, top_k=4)
    return {"prediction": float(pred), "drivers": drivers, "shap_backend": "shap" if HAS_SHAP else "permutation_fallback"}


class AskRequest(BaseModel):
    workflow: str  # "explain_decision" | "compare_stores" | "explain_forecast" | "executive_summary"
    payload: dict


@app.post("/ask")
def ask(req: AskRequest):
    if req.workflow == "explain_decision":
        answer = explain_decision(req.payload["store_id"], req.payload["allocation"], req.payload.get("drivers", []))
    elif req.workflow == "compare_stores":
        answer = compare_stores(
            req.payload["store_a_id"], req.payload["store_a"],
            req.payload["store_b_id"], req.payload["store_b"],
        )
    elif req.workflow == "explain_forecast":
        answer = explain_forecast(req.payload["forecast"], req.payload.get("drivers", []))
    elif req.workflow == "executive_summary":
        df = STATE["df"]
        kpis = compute_kpis(df)
        insights = generate_insights(df)
        answer = executive_summary(kpis, insights, req.payload.get("allocation_summary", {}))
    else:
        raise HTTPException(400, f"Unknown workflow: {req.workflow}")
    return {"answer": answer}
