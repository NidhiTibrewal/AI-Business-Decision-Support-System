"""
Module 3 — Predict Future Demand
==================================
Responsibility: forecast next-week demand per (store, category), with a
PREDICTION INTERVAL attached to every point forecast.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


from xgboost import XGBRegressor

FEATURES = [
    "lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4", "rolling_mean_8",
    "is_promo", "is_holiday", "temperature", "fuel_price", "cpi", "unemployment",
    "week_of_year", "month", "store_type_A", "store_type_B", "store_type_C",
]
TARGET = "weekly_sales"


def _make_tree_model():
    return XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=10)


def walk_forward_splits(weeks: np.ndarray, n_splits: int = 4, test_weeks: int = 4):
    """
    Yield (train_weeks, test_weeks) pairs that always train on the past.
    """
    weeks = np.sort(weeks)
    for i in range(n_splits, 0, -1):
        end = len(weeks) - (i - 1) * test_weeks
        start_test = end - test_weeks
        if start_test <= 0:
            continue
        yield weeks[:start_test], weeks[start_test:end]


def evaluate_walk_forward(df: pd.DataFrame, n_splits: int = 4, test_weeks: int = 4) -> pd.DataFrame:
    """
    Run walk-forward validation for both the tree model and the linear
    baseline, so model choice is justified by evidence, not assumption.
    """
    weeks = df["week"].unique()
    results = []

    for split_id, (train_weeks, test_weeks_arr) in enumerate(
        walk_forward_splits(weeks, n_splits, test_weeks), start=1
    ):
        train = df[df["week"].isin(train_weeks)]
        test = df[df["week"].isin(test_weeks_arr)]
        if train.empty or test.empty:
            continue

        X_train, y_train = train[FEATURES], train[TARGET]
        X_test, y_test = test[FEATURES], test[TARGET]

        linear = LinearRegression().fit(X_train, y_train)
        tree = _make_tree_model().fit(X_train, y_train)

        for name, model in [("linear_baseline", linear), ("tree_model", tree)]:
            preds = model.predict(X_test)
            results.append({
                "split": split_id,
                "model": name,
                "mae": mean_absolute_error(y_test, preds),
                "rmse": mean_squared_error(y_test, preds) ** 0.5,
                "n_test_rows": len(y_test),
            })

    return pd.DataFrame(results)


def train_final_model(df: pd.DataFrame):
    """
    Train the production model on ALL available history. 
    """
    model = _make_tree_model().fit(df[FEATURES], df[TARGET])
    residuals = df[TARGET].values - model.predict(df[FEATURES])
    return model, residuals


def predict_with_interval(model, residuals: np.ndarray, X: pd.DataFrame,
                           confidence: float = 0.90, n_bootstrap: int = 500,
                           seed: int = 42) -> pd.DataFrame:
    """
    Point forecast + a bootstrap prediction interval built from the
    in-sample residual distribution.

    Method: resample residuals with replacement `n_bootstrap` times, add
    each resampled residual to the point forecast, then read off the
    (1-confidence)/2 and 1-(1-confidence)/2 percentiles.
    """
    rng = np.random.default_rng(seed)
    point_forecast = model.predict(X)

    lower_q = (1 - confidence) / 2
    upper_q = 1 - lower_q

    lowers, uppers = [], []
    for pf in point_forecast:
        sampled_resid = rng.choice(residuals, size=n_bootstrap, replace=True)
        sim = pf + sampled_resid
        lowers.append(np.quantile(sim, lower_q))
        uppers.append(np.quantile(sim, upper_q))

    out = X.copy()
    out["forecast"] = point_forecast
    out["lower_bound"] = np.clip(lowers, 0, None)
    out["upper_bound"] = uppers
    out["interval_width"] = out["upper_bound"] - out["lower_bound"]

    relative_width = out["interval_width"] / out["forecast"].replace(0, np.nan)
    out["confidence_score"] = (1 / (1 + relative_width.fillna(relative_width.max()))).clip(0, 1)
    return out


if __name__ == "__main__":
    from module1_data_foundation import build_feature_table

    df = build_feature_table(save=False)

    cv_results = evaluate_walk_forward(df)
    print("\nWalk-forward CV (mean across splits):")
    print(cv_results.groupby("model")[["mae", "rmse"]].mean().round(2))

    model, residuals = train_final_model(df)
    latest_week = df["week"].max()
    latest = df[df["week"] == latest_week]
    forecast = predict_with_interval(model, residuals, latest[FEATURES])
    forecast[["forecast", "lower_bound", "upper_bound", "confidence_score"]] = \
        forecast[["forecast", "lower_bound", "upper_bound", "confidence_score"]].round(2)
    print("\nSample forecasts with intervals (latest week):")
    print(forecast[["forecast", "lower_bound", "upper_bound", "confidence_score"]].head())
