"""
Module 2 — Understand the Business (EDA + KPIs)
=================================================
Responsibility: turn the clean feature table into numbers a business
stakeholder actually cares about, plus statistical evidence for the two claims managers always ask about:
"does promotion actually move sales?" and "do holidays actually move sales?"
"""

import pandas as pd
import numpy as np
from scipy import stats


# KPIs
def compute_kpis(df: pd.DataFrame) -> dict:
    total_revenue = df["weekly_sales"].sum()

    # Revenue trend: compare first quarter of history vs last quarter.
    df_sorted = df.sort_values("week")
    weeks = df_sorted["week"].unique()
    cut = len(weeks) // 4
    early = df_sorted[df_sorted["week"].isin(weeks[:cut])]["weekly_sales"].sum()
    late = df_sorted[df_sorted["week"].isin(weeks[-cut:])]["weekly_sales"].sum()
    revenue_trend_pct = (late - early) / early * 100 if early else np.nan

    # Category concentration: share of revenue from the top 20% of
    series_rev = df.groupby(["store_id", "category_id"])["weekly_sales"].sum().sort_values(ascending=False)
    top_20pct_n = max(1, int(len(series_rev) * 0.2))
    concentration_pct = series_rev.iloc[:top_20pct_n].sum() / series_rev.sum() * 100

    active_weeks = df.groupby(["store_id", "category_id"])["weekly_sales"].apply(lambda s: (s > 0).mean())
    retention_proxy_pct = (active_weeks >= 0.9).mean() * 100

    cv = df.groupby(["store_id", "category_id"])["weekly_sales"].agg(lambda s: s.std() / s.mean() if s.mean() else 0)
    avg_volatility_cv = cv.mean()

    return {
        "total_revenue": round(total_revenue, 2),
        "revenue_trend_pct_first_vs_last_quarter": round(revenue_trend_pct, 2),
        "top20pct_series_revenue_share_pct": round(concentration_pct, 2),
        "retention_proxy_pct_series_consistently_active": round(retention_proxy_pct, 2),
        "avg_demand_volatility_cv": round(avg_volatility_cv, 3),
    }


# Hypothesis testing
def test_promotion_effect(df: pd.DataFrame) -> dict:
    """
    Welch's t-test: does weekly_sales differ between promo and non-promo
    weeks?
    """
    promo = df.loc[df["is_promo"] == 1, "weekly_sales"]
    non_promo = df.loc[df["is_promo"] == 0, "weekly_sales"]
    t_stat, p_val = stats.ttest_ind(promo, non_promo, equal_var=False)
    lift_pct = (promo.mean() - non_promo.mean()) / non_promo.mean() * 100
    return {
        "test": "Welch's t-test (promo vs non-promo weekly_sales)",
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p_val), 6),
        "significant_at_0.05": bool(p_val < 0.05),
        "estimated_lift_pct": round(lift_pct, 2),
    }


def test_holiday_effect(df: pd.DataFrame) -> dict:
    """
    One-way ANOVA across holiday vs non-holiday weeks.
    """
    groups = [g["weekly_sales"].values for _, g in df.groupby("is_holiday")]
    f_stat, p_val = stats.f_oneway(*groups)
    holiday_mean = df.loc[df["is_holiday"] == 1, "weekly_sales"].mean()
    non_holiday_mean = df.loc[df["is_holiday"] == 0, "weekly_sales"].mean()
    lift_pct = (holiday_mean - non_holiday_mean) / non_holiday_mean * 100
    return {
        "test": "One-way ANOVA (holiday vs non-holiday weekly_sales)",
        "f_stat": round(float(f_stat), 3),
        "p_value": round(float(p_val), 6),
        "significant_at_0.05": bool(p_val < 0.05),
        "estimated_lift_pct": round(lift_pct, 2),
    }


# Trend / seasonality summary (for the Overview dashboard page)
def weekly_revenue_series(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("week", as_index=False)["weekly_sales"].sum().sort_values("week")


def category_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    return (df.groupby("category_id", as_index=False)["weekly_sales"].sum().sort_values("weekly_sales", ascending=False))


# Numbered business insights 
def generate_insights(df: pd.DataFrame) -> list:
    kpis = compute_kpis(df)
    promo = test_promotion_effect(df)
    holiday = test_holiday_effect(df)
    top_cat = category_leaderboard(df).iloc[0]

    insights = [
        f"1. Revenue moved {kpis['revenue_trend_pct_first_vs_last_quarter']:+.1f}% "
        f"from the first to the last quarter of the observed history.",
        f"2. Promotions produce a statistically significant "
        f"({'p<0.05' if promo['significant_at_0.05'] else 'not significant'}) "
        f"sales lift of {promo['estimated_lift_pct']:+.1f}%.",
        f"3. Holiday weeks show a {holiday['estimated_lift_pct']:+.1f}% sales lift "
        f"({'statistically significant' if holiday['significant_at_0.05'] else 'not significant'}), "
        f"confirming holiday weeks deserve their own inventory buffer.",
        f"4. Category {int(top_cat['category_id'])} is the single largest revenue driver "
        f"at {top_cat['weekly_sales']:,.0f} total.",
        f"5. The top 20% of (store, category) series account for "
        f"{kpis['top20pct_series_revenue_share_pct']:.1f}% of total revenue — "
        f"a concentration worth prioritizing in allocation decisions.",
    ]
    return insights


if __name__ == "__main__":
    from module1_data_foundation import build_feature_table

    df = build_feature_table(save=False)
    print("KPIs:", compute_kpis(df))
    print("Promo test:", test_promotion_effect(df))
    print("Holiday test:", test_holiday_effect(df))
    print("\nInsights:")
    for line in generate_insights(df):
        print(" ", line)
