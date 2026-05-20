"""
EDA analytics. Owner: Person A.

Functions here compute WoE/IV tables, distributions, and data-quality summaries.
View modules call these and pass results to components/charts.py for rendering.

Person B notes:
- Each function should be pure (input df, output dataframe/dict).
- Use @st.cache_data on expensive computations.
- Don't render in this module — return data, render in views/.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_data
def compute_woe_iv(
    df: pd.DataFrame,
    feature: str,
    target: str = "default_flag",
    bins: int = 10,
) -> tuple[pd.DataFrame, float]:
    """
    Compute WoE per bin and overall IV for a single feature.

    Returns:
        woe_table: DataFrame with columns ['bin', 'count', 'goods', 'bads', 'woe']
        iv: total Information Value (float)

    TODO (Person A): implement proper handling for:
      - Categorical features (no binning needed)
      - Numeric features (quantile binning or supervised binning)
      - Missing values (treated as own bin)
      - Smoothing for empty bins (add small epsilon)
    """
    raise NotImplementedError("Person A: implement WoE/IV computation.")


@st.cache_data
def data_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return per-column quality summary:
    column, dtype, missing_count, missing_pct, n_unique, sample_values.

    TODO (Person A): extend with outlier flags, suspicious distributions,
    and notes for downstream feature engineering.
    """
    rows = []
    for col in df.columns:
        rows.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_pct": float(df[col].isna().mean() * 100),
            "n_unique": int(df[col].nunique(dropna=True)),
        })
    return pd.DataFrame(rows).sort_values("missing_pct", ascending=False)


@st.cache_data
def default_rate_by_category(
    df: pd.DataFrame, feature: str, target: str = "default_flag"
) -> pd.DataFrame:
    """
    Default rate per category for a categorical feature.

    Returns DataFrame: ['category', 'count', 'default_rate'].
    """
    grouped = df.groupby(feature, dropna=False).agg(
        count=(target, "size"),
        default_rate=(target, "mean"),
    ).reset_index().rename(columns={feature: "category"})
    return grouped.sort_values("default_rate", ascending=False)
