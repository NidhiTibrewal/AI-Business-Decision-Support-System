"""
Module 4 — Understand Store Behavior (Segmentation)
=====================================================
Responsibility: cluster stores by behavior (not just by store_type metadata)
so Module 5 can reason about groups like "high-volume/low-volatility — safe
for lean inventory buffers" instead of treating every store identically.

Pipeline: engineer per-store behavioral features -> standardize -> PCA (for
visualization + noise reduction) -> KMeans -> attach a human-readable label
to each resulting cluster.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def build_store_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per store, describing its demand behavior rather than its
    metadata (size/type already comes from Module 1's merge, but here we
    care about how the store actually performs).
    """
    agg = df.groupby("store_id").agg(
        avg_weekly_sales=("weekly_sales", "mean"),
        volume_total=("weekly_sales", "sum"),
        volatility_std=("weekly_sales", "std"),
        promo_sensitivity=("is_promo", "mean"),
    ).reset_index()

    agg["volatility_cv"] = agg["volatility_std"] / agg["avg_weekly_sales"]

    margin_proxy = df.groupby("store_id").apply(
        lambda g: (g["weekly_sales"].sum() / g["units_sold"].sum()) if g["units_sold"].sum() else 0,
        include_groups=False,
    ).reset_index(name="margin_proxy")
    agg = agg.merge(margin_proxy, on="store_id")
    return agg


def cluster_stores(store_features: pd.DataFrame, n_clusters: int = 4, seed: int = 42):
    feature_cols = ["avg_weekly_sales", "volatility_cv", "promo_sensitivity", "margin_proxy"]
    X = store_features[feature_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    out = store_features.copy()
    out["cluster"] = labels
    out["pca_1"] = X_pca[:, 0]
    out["pca_2"] = X_pca[:, 1]
    return out, {"scaler": scaler, "pca": pca, "kmeans": kmeans,
                 "explained_variance_ratio": pca.explained_variance_ratio_.round(3).tolist()}


def label_clusters(clustered: pd.DataFrame) -> dict:
    """
    Translate each numeric cluster into a business-readable description
    by comparing its centroid behavior against the overall population
    median on volume and volatility.
    """
    overall_vol_med = clustered["avg_weekly_sales"].median()
    overall_cv_med = clustered["volatility_cv"].median()

    labels = {}
    for cluster_id, g in clustered.groupby("cluster"):
        high_volume = g["avg_weekly_sales"].mean() >= overall_vol_med
        high_volatility = g["volatility_cv"].mean() >= overall_cv_med

        if high_volume and not high_volatility:
            desc = "High-volume / low-volatility — safe for lean inventory buffers."
        elif high_volume and high_volatility:
            desc = "High-volume / high-volatility — keep a larger safety stock despite the strong sales."
        elif not high_volume and not high_volatility:
            desc = "Low-volume / low-volatility — steady, predictable, minimal buffer needed."
        else:
            desc = "Low-volume / high-volatility — highest allocation risk; forecast confidence matters most here."

        labels[int(cluster_id)] = {
            "description": desc,
            "n_stores": int(len(g)),
            "avg_weekly_sales": round(float(g["avg_weekly_sales"].mean()), 1),
            "avg_volatility_cv": round(float(g["volatility_cv"].mean()), 3),
        }
    return labels


if __name__ == "__main__":
    from module1_data_foundation import build_feature_table

    df = build_feature_table(save=False)
    store_feats = build_store_features(df)
    clustered, meta = cluster_stores(store_feats)
    labels = label_clusters(clustered)

    print(f"[module4] PCA explained variance ratio: {meta['explained_variance_ratio']}")
    print("\nCluster labels:")
    for cid, info in labels.items():
        print(f"  Cluster {cid}: {info}")
    print("\nSample store assignments:")
    print(clustered[["store_id", "cluster", "avg_weekly_sales", "volatility_cv"]].head(8))
