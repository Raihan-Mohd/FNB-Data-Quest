"""
EDA analytics — WoE / IV, data quality, default rate summaries.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import streamlit as st

MISSING_LABEL = "__missing__"


@st.cache_data(show_spinner=False)
def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "default_flag",
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE per bin and total IV for a single feature.

    For numeric features with high cardinality: quantile binning.
    For categorical / low-cardinality: categories used as-is.
    Missing values: assigned to own bin.

    Returns:
        woe_table: DataFrame [bin, events, total, non_events, woe, iv, default_rate]
        iv:        total Information Value (float)
    """
    d = df[[feature, target]].copy()
    n_unique = d[feature].nunique()

    is_numeric = pd.api.types.is_numeric_dtype(d[feature])
    use_bins = is_numeric and n_unique > 12

    if use_bins:
        non_null = d[feature].notna()
        d["bin"] = MISSING_LABEL
        try:
            d.loc[non_null, "bin"] = pd.qcut(
                d.loc[non_null, feature], q=bins, duplicates="drop"
            ).astype(str)
        except Exception:
            try:
                d.loc[non_null, "bin"] = pd.cut(
                    d.loc[non_null, feature], bins=bins
                ).astype(str)
            except Exception:
                d.loc[non_null, "bin"] = d.loc[non_null, feature].astype(str)
    else:
        d["bin"] = d[feature].fillna(MISSING_LABEL).astype(str)

    total_ev  = int(d[target].sum())
    total_nev = int((d[target] == 0).sum())

    g = (
        d.groupby("bin", observed=True)[target]
         .agg(["sum", "count"])
         .rename(columns={"sum": "events", "count": "total"})
    )
    g["non_events"] = g["total"] - g["events"]
    g["dist_ev"]    = (g["events"]     / max(total_ev, 1)).clip(lower=1e-4)
    g["dist_nev"]   = (g["non_events"] / max(total_nev, 1)).clip(lower=1e-4)
    g["woe"]        = np.log(g["dist_ev"] / g["dist_nev"])
    g["iv"]         = (g["dist_ev"] - g["dist_nev"]) * g["woe"]
    g["default_rate"] = g["events"] / g["total"]

    iv = float(g["iv"].sum())
    return g.reset_index(), iv


def _classify_iv(iv: float) -> str:
    if pd.isna(iv):      return "error"
    if iv < 0.02:        return "not predictive"
    if iv < 0.10:        return "weak"
    if iv < 0.30:        return "medium"
    if iv < 0.50:        return "strong"
    return "suspicious — check for leakage"


@st.cache_data(show_spinner="Ranking features by IV...")
def rank_features_by_iv(
    df: pd.DataFrame,
    features: list[str],
    target: str = "default_flag",
    bins: int = 10,
) -> pd.DataFrame:
    """Compute IV for every feature and return a sorted summary table."""
    rows = []
    for feat in features:
        if feat not in df.columns:
            rows.append({"feature": feat, "iv": np.nan, "strength": "missing column"})
            continue
        try:
            _, iv = compute_woe_iv(df, feat, target=target, bins=bins)
            rows.append({"feature": feat, "iv": iv, "strength": _classify_iv(iv)})
        except Exception as e:
            rows.append({"feature": feat, "iv": np.nan, "strength": f"error: {e}"})
    return (
        pd.DataFrame(rows)
          .sort_values("iv", ascending=False, na_position="last")
          .reset_index(drop=True)
    )


@st.cache_data(show_spinner=False)
def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        rows.append({
            "column":        col,
            "dtype":         str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct":   float(df[col].isna().mean() * 100),
            "n_unique":      int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


@st.cache_data(show_spinner=False)
def default_rate_by_category(
    df: pd.DataFrame, feature: str, target: str = "default_flag"
) -> pd.DataFrame:
    return (
        df.groupby(feature, dropna=False)
          .agg(count=(target, "size"), default_rate=(target, "mean"))
          .reset_index()
          .rename(columns={feature: "category"})
          .sort_values("default_rate", ascending=False)
    )
