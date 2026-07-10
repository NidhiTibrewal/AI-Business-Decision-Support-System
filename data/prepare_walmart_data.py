"""
prepare_walmart_data.py
========================
Converts the REAL Walmart Recruiting - Store Sales Forecasting dataset
(train.csv, test.csv, features.csv, stores.csv from Kaggle) into the same
three-file schema the rest of this project already expects
(data/raw/sales.csv, stores.csv, calendar.csv) — so nothing in Modules 1-9
needs to change to run on real data. This is the "swap in the real CSVs"
step promised in the README.

SOURCE FILES (place in data/raw_walmart/, already done for you):
    train.csv     Store, Dept, Date, Weekly_Sales, IsHoliday   (2010-02 -> 2012-10)
    test.csv      Store, Dept, Date, IsHoliday                 (2012-11 -> 2013-07, NO target)
    features.csv  Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday
    stores.csv    Store, Type, Size

IMPORTANT — why test.csv is NOT used for training or benchmarking:
    test.csv has no Weekly_Sales column. It exists so competitors submit
    predictions to Kaggle's servers for scoring; without a Kaggle account
    there is no way to know the true answer for those weeks. All benchmark
    numbers in this project therefore come from WALK-FORWARD VALIDATION
    inside train.csv (holding out the last few weeks of *known* history),
    not from test.csv. This is a more honest number anyway — it is what
    you can actually reproduce and defend without a leaderboard.

MAPPING TO THIS PROJECT'S SCHEMA:
    Store        -> store_id
    Dept         -> category_id
    Date         -> week
    Weekly_Sales -> weekly_sales
    IsHoliday    -> is_holiday
    Type         -> store_type
    Size         -> size_sqft
    MarkDown1-5  -> collapsed into is_promo (1 if any markdown is active that
                    week for that store) — Walmart's markdown events are the
                    closest real-world analog to the synthetic dataset's
                    promo flag.
    units_sold   -> NOT present in the real dataset (Walmart only discloses
                    dollar sales, not unit counts). Left as NaN; Module 4's
                    margin proxy falls back gracefully when this happens
                    (see the comment in module4_segmentation.py).
"""

from pathlib import Path
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).parent / "raw_walmart"
OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    train = pd.read_csv(SRC_DIR / "train.csv", parse_dates=["Date"])
    features = pd.read_csv(SRC_DIR / "features.csv", parse_dates=["Date"])
    stores = pd.read_csv(SRC_DIR / "stores.csv")

    # ---- stores.csv --------------------------------------------------------
    stores_out = stores.rename(columns={"Store": "store_id", "Type": "store_type", "Size": "size_sqft"})
    stores_out["region"] = "NA"  # not disclosed in the real dataset
    stores_out = stores_out[["store_id", "region", "store_type", "size_sqft"]]
    stores_out.to_csv(OUT_DIR / "stores.csv", index=False)

    # ---- calendar.csv (kept at STORE x WEEK grain, not just week, since  --
    # ---- the real dataset's temperature/fuel/CPI/unemployment genuinely --
    # ---- vary by store — collapsing to week-only would throw away signal)-
    markdown_cols = [c for c in features.columns if c.startswith("MarkDown")]
    features["is_promo"] = features[markdown_cols].fillna(0).gt(0).any(axis=1).astype(int)

    calendar_out = features.rename(columns={
        "Store": "store_id", "Date": "week", "Temperature": "temperature",
        "Fuel_Price": "fuel_price", "CPI": "cpi", "Unemployment": "unemployment",
        "IsHoliday": "is_holiday",
    })
    calendar_out["is_holiday"] = calendar_out["is_holiday"].astype(bool).astype(int)
    calendar_out["holiday_name"] = np.where(calendar_out["is_holiday"] == 1, "Holiday", "")
    # CPI/Unemployment have ~7% missing (mostly the most recent weeks) —
    # forward-fill per store, which is standard practice for these two
    # slow-moving macro series in this specific competition.
    calendar_out = calendar_out.sort_values(["store_id", "week"])
    calendar_out[["cpi", "unemployment"]] = (
        calendar_out.groupby("store_id")[["cpi", "unemployment"]].ffill().bfill()
    )
    calendar_out = calendar_out[[
        "store_id", "week", "is_holiday", "holiday_name", "temperature",
        "fuel_price", "cpi", "unemployment", "is_promo",
    ]]
    calendar_out.to_csv(OUT_DIR / "calendar.csv", index=False)

    # ---- sales.csv ----------------------------------------------------------
    sales_out = train.rename(columns={
        "Store": "store_id", "Dept": "category_id", "Date": "week",
        "Weekly_Sales": "weekly_sales",
    })
    # is_holiday and is_promo both come from calendar.csv at merge time
    # (module1 merges on store_id + week) — sales.csv doesn't duplicate them,
    # so there's a single source of truth for calendar-derived flags.
    sales_out["units_sold"] = np.nan  # not disclosed in the real dataset
    sales_out = sales_out[["store_id", "category_id", "week", "weekly_sales", "units_sold"]]
    sales_out.to_csv(OUT_DIR / "sales.csv", index=False)

    print(f"[prepare_walmart_data] wrote real dataset:")
    print(f"  sales.csv    {len(sales_out):,} rows  "
          f"({sales_out.store_id.nunique()} stores x {sales_out.category_id.nunique()} depts, "
          f"{sales_out.week.min().date()} -> {sales_out.week.max().date()})")
    print(f"  stores.csv   {len(stores_out)} rows")
    print(f"  calendar.csv {len(calendar_out):,} rows")


if __name__ == "__main__":
    main()
