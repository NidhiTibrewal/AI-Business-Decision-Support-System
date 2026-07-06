"""
generate_synthetic_data.py
===========================
Generates a synthetic weekly retail dataset.

Output files (written to data/raw/):
    sales.csv     - store x category x week panel of weekly_sales, units, promo
    stores.csv    - store metadata (region, size, type)
    calendar.csv  - week -> holiday flag, holiday name, is_super_bowl etc.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_STORES = 20
N_CATEGORIES = 8
N_WEEKS = 156  # 3 years of weekly data
START_DATE = pd.Timestamp("2023-01-06")  

STORE_TYPES = ["A", "B", "C"]  # A = large format, B = medium, C = small
REGIONS = ["North", "South", "East", "West"]


def build_calendar() -> pd.DataFrame:
    """
    Weekly calendar with US-style holiday markers used as model features.
    """
    dates = pd.date_range(START_DATE, periods=N_WEEKS, freq="W-FRI")
    cal = pd.DataFrame({"week": dates})
    cal["is_holiday"] = 0
    cal["holiday_name"] = ""

    # Mark the weeks that contain major US retail holidays (approximate, but
    # consistent year over year so the model can actually learn the pattern).
    for year in cal["week"].dt.year.unique():
        holiday_weeks = {
            "Super Bowl": pd.Timestamp(f"{year}-02-09"),
            "Independence Day": pd.Timestamp(f"{year}-07-04"),
            "Thanksgiving": pd.Timestamp(f"{year}-11-24"),
            "Christmas": pd.Timestamp(f"{year}-12-25"),
        }
        for name, hdate in holiday_weeks.items():
            nearest_idx = (cal["week"] - hdate).abs().idxmin()
            cal.loc[nearest_idx, "is_holiday"] = 1
            cal.loc[nearest_idx, "holiday_name"] = name

    cal["temperature"] = (
        60 + 25 * np.sin(2 * np.pi * (cal["week"].dt.dayofyear / 365.0) - np.pi / 2)
        + RNG.normal(0, 3, len(cal))
    ).round(1)
    cal["fuel_price"] = (3.2 + RNG.normal(0, 0.15, len(cal)).cumsum() * 0.02).round(2)
    cal["cpi"] = (210 + np.arange(len(cal)) * 0.05 + RNG.normal(0, 0.3, len(cal))).round(2)
    cal["unemployment"] = (6.5 - np.arange(len(cal)) * 0.005 + RNG.normal(0, 0.1, len(cal))).round(2)
    return cal


def build_stores() -> pd.DataFrame:
    stores = pd.DataFrame({"store_id": np.arange(1, N_STORES + 1)})
    stores["region"] = RNG.choice(REGIONS, size=N_STORES)
    stores["store_type"] = RNG.choice(STORE_TYPES, size=N_STORES, p=[0.3, 0.45, 0.25])
    size_map = {"A": (150000, 200000), "B": (80000, 140000), "C": (30000, 70000)}
    stores["size_sqft"] = stores["store_type"].apply(
        lambda t: RNG.integers(*size_map[t])
    )
    return stores


def build_sales(calendar: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, store in stores.iterrows():
        # store-level "personality": baseline demand level + volatility
        type_multiplier = {"A": 1.6, "B": 1.0, "C": 0.55}[store["store_type"]]
        store_base = RNG.uniform(800, 1600) * type_multiplier
        store_volatility = RNG.uniform(0.08, 0.35)  
        trend_per_week = RNG.uniform(-0.3, 1.2)

        for cat in range(1, N_CATEGORIES + 1):
            cat_multiplier = RNG.uniform(0.4, 1.4)
            promo_weeks = RNG.choice(
                [0, 1], size=N_WEEKS, p=[0.75, 0.25]
            )  # 25% of weeks have an active promotion

            for i, wk_row in calendar.iterrows():
                seasonal = 1 + 0.25 * np.sin(2 * np.pi * i / 52.0)
                holiday_lift = 1.35 if wk_row["is_holiday"] else 1.0
                promo_lift = 1.22 if promo_weeks[i] else 1.0
                trend = 1 + trend_per_week * i / 100.0
                noise = RNG.normal(1.0, store_volatility)

                weekly_sales = max(
                    0,
                    store_base
                    * cat_multiplier
                    * seasonal
                    * holiday_lift
                    * promo_lift
                    * trend
                    * noise,
                )
                units = int(weekly_sales / RNG.uniform(8, 15))
                rows.append(
                    {
                        "store_id": store["store_id"],
                        "category_id": cat,
                        "week": wk_row["week"],
                        "weekly_sales": round(weekly_sales, 2),
                        "units_sold": units,
                        "is_promo": int(promo_weeks[i]),
                    }
                )
    return pd.DataFrame(rows)


def main():
    calendar = build_calendar()
    stores = build_stores()
    sales = build_sales(calendar, stores)

    calendar.to_csv(OUT_DIR / "calendar.csv", index=False)
    stores.to_csv(OUT_DIR / "stores.csv", index=False)
    sales.to_csv(OUT_DIR / "sales.csv", index=False)

    print(f"Wrote {len(sales):,} sales rows, {len(stores)} stores, {len(calendar)} weeks -> {OUT_DIR}")


if __name__ == "__main__":
    main()
