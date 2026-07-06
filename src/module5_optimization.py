"""
Module 5 — Recommend Inventory Allocation (Optimization)
==========================================================
THE CENTERPIECE MODULE. Everything else in this project (forecasting,
segmentation, explainability) exists to feed this function a good input,
or to explain its output.

Built as a single parameterized function, `optimize_allocation`, precisely
so that Module 5's own "scenario analysis" feature — a manager moving a
budget slider and getting a new allocation instantly — is just "call this
function again with different arguments".

-----------------------------------------------------------------------------
THE KEY MECHANIC: CONFIDENCE-WEIGHTED DEMAND SHRINKAGE
-----------------------------------------------------------------------------
The optimizer does not feed the raw point forecast into the LP. It shrinks
each store's forecast toward the *lower bound* of its prediction interval,
by an amount controlled by that store's confidence_score (from Module 3):

    effective_demand = lower_bound + confidence_score * (forecast - lower_bound)

-----------------------------------------------------------------------------
THE LP
-----------------------------------------------------------------------------
Decision variables:      x_i  = units allocated to store i              (i = 1..n)
Objective (minimize):    sum_i [ holding_cost * x_i
                                  + transport_cost * distance_i * x_i
                                  + lost_sale_penalty * max(0, effective_demand_i - x_i) ]
    The max(0, ...) term is linearized in the LP via an auxiliary variable
    shortfall_i >= 0 with the constraint shortfall_i >= effective_demand_i - x_i.

Constraints:
    0 <= x_i <= per_store_capacity                    (per-store shelf/backroom limit)
    sum_i x_i <= warehouse_capacity                   (total units available to ship)
    sum_i (unit_cost * x_i) <= budget                 (optional budget cap, for scenarios)

Solver: Google OR-Tools' linear solver (GLOP) is used
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ortools.linear_solver import pywraplp


@dataclass
class CostParams:
    holding_cost_per_unit: float = 2.0        # ₹ / unit / week
    transport_cost_per_km: float = 0.6        # ₹ / km
    lost_sale_penalty_per_unit: float = 18.0  # ₹ / unit of unmet demand
    warehouse_capacity: int = 50_000
    per_store_capacity: int = 5_000
    unit_cost: float = 10.0                   # ₹ / unit, used only if a budget cap is set
    budget: float = None                      # None = unconstrained


# Step 1: confidence-weighted demand shrinkage
def compute_effective_demand(forecast_df: pd.DataFrame) -> pd.Series:
    """
    forecast_df must have columns: forecast, lower_bound, confidence_score.
    Returns the shrunk demand estimate that actually feeds the LP.
    """
    return (
        forecast_df["lower_bound"]
        + forecast_df["confidence_score"] * (forecast_df["forecast"] - forecast_df["lower_bound"])
    )


# Step 2: the LP itself
def optimize_allocation(store_ids, effective_demand, distances_km, params: CostParams) -> pd.DataFrame:
    n = len(store_ids)
    return _solve_with_ortools(store_ids, effective_demand, distances_km, params, n)


def _solve_with_ortools(store_ids, demand, dist, params, n):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    x = [solver.NumVar(0, params.per_store_capacity, f"x_{i}") for i in range(n)]
    shortfall = [solver.NumVar(0, solver.infinity(), f"short_{i}") for i in range(n)]

    for i in range(n):
        solver.Add(shortfall[i] >= demand[i] - x[i])

    solver.Add(solver.Sum(x) <= params.warehouse_capacity)
    if params.budget is not None:
        solver.Add(solver.Sum(x) * params.unit_cost <= params.budget)

    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], params.holding_cost_per_unit + params.transport_cost_per_km * dist[i])
        objective.SetCoefficient(shortfall[i], params.lost_sale_penalty_per_unit)
    objective.SetMinimization()

    status = solver.Solve()
    solved = status == pywraplp.Solver.OPTIMAL

    return _build_result_df(
        store_ids, demand,
        [x[i].solution_value() if solved else 0 for i in range(n)],
        [shortfall[i].solution_value() if solved else demand[i] for i in range(n)],
        solved,
    )

def _build_result_df(store_ids, demand, x_vals, shortfall_vals, solved) -> pd.DataFrame:
    df = pd.DataFrame({
        "store_id": store_ids,
        "effective_demand": np.round(demand, 1),
        "allocated_units": np.round(x_vals, 1),
        "projected_shortfall": np.round(shortfall_vals, 1),
    })
    df["fill_rate_pct"] = np.where(
        df["effective_demand"] > 0,
        (1 - df["projected_shortfall"] / df["effective_demand"]).clip(0, 1) * 100,
        100.0,
    )
    df.attrs["solver_status"] = "optimal" if solved else "infeasible_or_failed"
    df.attrs["solver_backend"] = "ortools" 
    return df


# Step 3: scenario analysis wrapper
def run_scenario(store_ids, forecast_df, distances_km, base_params: CostParams, **overrides) -> pd.DataFrame:
    """
    Re-run the same allocation under changed business constraints.
    """
    scenario_params = CostParams(**{**base_params.__dict__, **overrides})
    demand = compute_effective_demand(forecast_df).values
    return optimize_allocation(store_ids, demand, distances_km, scenario_params)


if __name__ == "__main__":

    # Small worked example mirroring the README's Store A / Store B illustration.
    demo = pd.DataFrame({
        "store_id": ["A", "B", "C"],
        "forecast": [1000, 950, 700],
        "lower_bound": [550, 900, 500],
        "confidence_score": [0.55, 0.95, 0.75],
    })
    demo["effective_demand"] = compute_effective_demand(demo)
    print("\nConfidence-weighted effective demand:")
    print(demo[["store_id", "forecast", "lower_bound", "confidence_score", "effective_demand"]])

    params = CostParams(warehouse_capacity=2000, per_store_capacity=1200)
    distances = [12, 30, 5]
    result = optimize_allocation(demo["store_id"], demo["effective_demand"].values, distances, params)
    print(f"\nBase allocation (status: {result.attrs['solver_status']}):")
    print(result)

    print("\nScenario: 10% budget cut via a tighter warehouse cap")
    scenario_result = run_scenario(
        demo["store_id"], demo, distances, params, warehouse_capacity=int(2000 * 0.9)
    )
    print(scenario_result)
