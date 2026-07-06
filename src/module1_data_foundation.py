"""
Module 1 — Data Foundation
===========================
Responsibility: turn three raw, messy-in-real-life CSVs (sales, stores,
calendar) into a single clean, feature-rich panel that every other module
reads from. Nothing downstream should ever touch a raw file again.
"""

from pathlib import Path
import pandas as pd
import numpy as np

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# Load
def load_raw(raw_dir: Path = RAW_DIR) -> dict:
    """
    Load the three raw tables. 
    """
    sales = pd.read_csv(raw_dir / "sales.csv", parse_dates=["week"])
    stores = pd.read_csv(raw_dir / "stores.csv")
    calendar = pd.read_csv(raw_dir / "calendar.csv", parse_dates=["week"])
    return {"sales": sales, "stores": stores, "calendar": calendar}


# Validate + clean
def validate_and_clean(sales: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data-quality gate.
    """
    before = len(sales)

    sales = sales.drop_duplicates(subset=["store_id", "category_id", "week"])
    sales = sales[sales["weekly_sales"] >= 0].copy()
    sales["weekly_sales"] = sales["weekly_sales"].fillna(0)
    sales["units_sold"] = sales["units_sold"].fillna(0)

    dropped = before - len(sales)
    if dropped:
        print(f"[module1] dropped {dropped} invalid/duplicate rows ({dropped/before:.2%})")

    return sales.sort_values(["store_id", "category_id", "week"]).reset_index(drop=True)


# Merge
def merge_all(tables: dict) -> pd.DataFrame:
    df = tables["sales"].merge(tables["stores"], on="store_id", how="left")
    df = df.merge(tables["calendar"], on="week", how="left")
    return df


# Feature engineering
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Lag / rolling / calendar features. Grouped by (store_id, category_id)
    so history never leaks across series.
    """
    df = df.sort_values(["store_id", "category_id", "week"]).copy()
    grp = df.groupby(["store_id", "category_id"])["weekly_sales"]

    df["lag_1"] = grp.shift(1)
    df["lag_2"] = grp.shift(2)
    df["lag_4"] = grp.shift(4)

    shifted = grp.shift(1) 
    group_keys = [df["store_id"], df["category_id"]]
    df["rolling_mean_4"] = shifted.groupby(group_keys).rolling(4).mean().reset_index(level=[0, 1], drop=True)
    df["rolling_std_4"] = shifted.groupby(group_keys).rolling(4).std().reset_index(level=[0, 1], drop=True)
    df["rolling_mean_8"] = shifted.groupby(group_keys).rolling(8).mean().reset_index(level=[0, 1], drop=True)

    df["week_of_year"] = df["week"].dt.isocalendar().week.astype(int)
    df["month"] = df["week"].dt.month
    df["year"] = df["week"].dt.year

    df["store_type_A"] = (df["store_type"] == "A").astype(int)
    df["store_type_B"] = (df["store_type"] == "B").astype(int)
    df["store_type_C"] = (df["store_type"] == "C").astype(int)

    # Rows without enough history for lag_4 (first ~4 weeks per series) get dropped
    df = df.dropna(
        subset=["lag_1", "lag_2", "lag_4", "rolling_mean_4", "rolling_std_4", "rolling_mean_8"]
    ).reset_index(drop=True)
    return df


# Orchestration
def build_feature_table(save: bool = True) -> pd.DataFrame:
    tables = load_raw()
    sales = validate_and_clean(tables["sales"])
    tables["sales"] = sales
    merged = merge_all(tables)
    featured = engineer_features(merged)

    if save:
        out = PROCESSED_DIR / "feature_table.parquet"
        try:
            featured.to_parquet(out, index=False)
        except Exception:
            featured.to_csv(PROCESSED_DIR / "feature_table.csv", index=False)
    return featured


if __name__ == "__main__":
    ft = build_feature_table()
    print(f"[module1] feature table: {ft.shape[0]:,} rows x {ft.shape[1]} cols")
    print(ft.head(3))
