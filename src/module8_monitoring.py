"""
Module 8 — Monitoring & Tracking
===================================
Deliberately small, per the project plan (timeboxed to ~1 day, should not
compete with Module 5 for build time). Two responsibilities, no more:

  1. Log each model run to MLflow (model, RMSE/MAE, params, dataset version).
     Falls back to a local JSON log if MLflow isn't installed, so nothing
     in the pipeline breaks in a restricted environment.
  2. Track rolling MAE between actual vs. predicted demand and flag if it
     crosses a threshold. That is the entire monitoring system on purpose —
     no drift-detection framework, no alerting pipeline.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

LOCAL_LOG_DIR = Path(__file__).resolve().parent.parent / "artifacts"
LOCAL_LOG_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_LOG_FILE = LOCAL_LOG_DIR / "experiment_log.jsonl"


def log_run(model_name: str, params: dict, metrics: dict, dataset_version: str = "v1"):
    """Log one training run. Uses MLflow if available; otherwise appends a
    JSON line locally with the same information, so run history is never
    silently lost.
    """
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "dataset_version": dataset_version,
        "params": params,
        "metrics": metrics,
    }

    if HAS_MLFLOW:
        with mlflow.start_run(run_name=model_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tag("dataset_version", dataset_version)
    else:
        with open(LOCAL_LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

    return record


# ---------------------------------------------------------------------------
# Rolling MAE monitor
# ---------------------------------------------------------------------------
def rolling_mae(actuals: np.ndarray, predictions: np.ndarray, window: int = 4) -> np.ndarray:
    errors = np.abs(np.asarray(actuals) - np.asarray(predictions))
    return np.convolve(errors, np.ones(window) / window, mode="valid")


def check_mae_threshold(actuals: np.ndarray, predictions: np.ndarray,
                         threshold: float, window: int = 4) -> dict:
    """Returns whether the most recent rolling MAE has crossed `threshold`.
    This is intentionally the entire "drift detection" system — a single
    number compared to a single threshold, logged, and surfaced on the
    dashboard's Overview page as a red/green flag.
    """
    rmae = rolling_mae(actuals, predictions, window)
    if len(rmae) == 0:
        return {"status": "insufficient_data", "latest_rolling_mae": None, "flagged": False}

    latest = float(rmae[-1])
    return {
        "status": "ok",
        "latest_rolling_mae": round(latest, 2),
        "threshold": threshold,
        "flagged": latest > threshold,
        "window": window,
    }


if __name__ == "__main__":
    print(f"[module8] MLflow available: {HAS_MLFLOW}")

    record = log_run(
        model_name="xgboost_demand_forecast",
        params={"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05},
        metrics={"mae": 302.75, "rmse": 519.64},
        dataset_version="synthetic_v1",
    )
    print("Logged run:", record if not HAS_MLFLOW else "(sent to MLflow)")

    rng = np.random.default_rng(1)
    actual = rng.normal(1000, 100, 30)
    pred = actual + rng.normal(0, 80, 30)
    result = check_mae_threshold(actual, pred, threshold=90)
    print("\nMonitoring check:", result)
