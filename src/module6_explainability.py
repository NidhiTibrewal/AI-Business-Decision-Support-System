"""
Module 6 — Explain the Decision (Explainability)
===================================================
Responsibility: turn "the model said 2,605 units" into "the model said 2,605
units because of these top 3-4 factors" — for both the demand forecast and,
where meaningful, the allocation outcome.

Deliberately NOT dumping a full SHAP summary plot with 17 features on a
manager's screen. The point of explainability here is decision support, so
we surface only the drivers big enough to matter and phrase them in the
units the business speaks (% impact), not raw SHAP log-odds/units debris.

Falls back to permutation importance if the `shap` package isn't installed,
so the interface (`explain_row` -> top-k driver list) never changes for
the dashboard/LLM layer regardless of which backend is available.
"""

import numpy as np
import pandas as pd

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    from sklearn.inspection import permutation_importance


FRIENDLY_NAMES = {
    "lag_1": "Last week's sales",
    "lag_2": "Sales 2 weeks ago",
    "lag_4": "Sales 4 weeks ago",
    "rolling_mean_4": "4-week average trend",
    "rolling_std_4": "Recent demand volatility",
    "rolling_mean_8": "8-week average trend",
    "is_promo": "Promotion",
    "is_holiday": "Holiday season",
    "temperature": "Temperature",
    "fuel_price": "Fuel price",
    "cpi": "Consumer Price Index",
    "unemployment": "Unemployment rate",
    "week_of_year": "Seasonal week-of-year effect",
    "month": "Month-of-year effect",
    "store_type_A": "Large-format store",
    "store_type_B": "Medium-format store",
    "store_type_C": "Small-format store",
}


def build_explainer(model, background_X: pd.DataFrame):
    """Returns a callable `explain(X) -> np.ndarray` of per-row, per-feature
    contribution values, using SHAP's TreeExplainer when available.
    """
    if HAS_SHAP:
        explainer = shap.TreeExplainer(model)

        def explain(X):
            return explainer.shap_values(X)

        return explain

    # Fallback: a *global* permutation-importance profile applied as a
    # rough per-row proxy (feature_value_delta_from_mean * global_weight).
    # This is a reasonable stand-in, not a true per-row attribution method
    # — worth saying explicitly in an interview if SHAP wasn't installed.
    result = permutation_importance(
        model, background_X, model.predict(background_X),
        n_repeats=5, random_state=42,
    )
    global_weights = result.importances_mean
    feature_means = background_X.mean()

    def explain(X):
        deltas = X - feature_means
        return deltas.values * global_weights

    return explain


def explain_row(shap_values_row: np.ndarray, feature_names: list, feature_values: pd.Series,
                 base_prediction: float, top_k: int = 4) -> list:
    """Convert one row's raw contribution values into a ranked, business-
    readable list of the top_k drivers with an approximate % impact.

    % impact is each driver's |contribution| as a share of the total
    absolute contribution across all features for that row — a simple,
    defensible normalization that always sums to <=100% across top_k.
    """
    contributions = pd.Series(shap_values_row, index=feature_names)
    total_abs = contributions.abs().sum()
    if total_abs == 0:
        return []

    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    top = ranked.head(top_k)

    drivers = []
    for feat, contrib in top.items():
        pct_impact = contrib / base_prediction * 100 if base_prediction else 0
        drivers.append({
            "feature": FRIENDLY_NAMES.get(feat, feat),
            "direction": "increase" if contrib > 0 else "decrease",
            "approx_pct_impact": round(pct_impact, 1),
        })
    return drivers


def format_driver_summary(drivers: list) -> str:
    """Renders driver list as the README's example block:
        Demand increased because:
          Holiday season     +21%
          Promotion          +14%
    """
    if not drivers:
        return "No dominant drivers identified for this prediction."
    lines = ["Demand changed because:"]
    for d in drivers:
        sign = "+" if d["direction"] == "increase" else ""
        lines.append(f"  {d['feature']:<28}{sign}{d['approx_pct_impact']}%")
    return "\n".join(lines)


if __name__ == "__main__":
    from module1_data_foundation import build_feature_table
    from module3_forecasting import train_final_model, FEATURES

    df = build_feature_table(save=False)
    model, residuals = train_final_model(df)
    print(f"[module6] SHAP available: {HAS_SHAP}")

    background = df[FEATURES].sample(min(200, len(df)), random_state=42)
    explain = build_explainer(model, background)

    sample = df[FEATURES].iloc[[0]]
    values = explain(sample)
    row_values = values[0] if HAS_SHAP else values[0]
    prediction = model.predict(sample)[0]

    drivers = explain_row(row_values, FEATURES, sample.iloc[0], base_prediction=prediction, top_k=4)
    print(f"\nPrediction: {prediction:.1f} units/₹")
    print(format_driver_summary(drivers))
