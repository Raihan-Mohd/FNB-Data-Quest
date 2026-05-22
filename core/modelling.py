"""
Modelling pipelines.

Two models per the task spec:

  1. fit_baseline   — minimal feature engineering (median impute + scale + one-hot).
  2. fit_improved   — WoE-engineered logistic regression, IV-filtered features.

Both return a FittedModel dataclass that the views and metrics layer consume.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core.eda import _bin_feature, WOE_EPSILON, MISSING_LABEL


# ── Feature configuration ──────────────────────────────────────────────────
NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_length_years",
    "num_open_accounts",
    "num_delinquencies_2yr",
    "total_revolving_balance",
    "credit_utilisation_pct",
    "months_since_oldest_account",
    "num_hard_inquiries_6mo",
    "loan_amount",
    "interest_rate",
    "dti_ratio",
    "months_since_last_delinquency",
    "pct_accounts_current",
    "months_at_current_address",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
    "email_domain_type",
    "phone_verified",
]

EXCLUDED_FEATURES = [
    "region",            # geographic proxy → disparate-impact risk
    "branch_code_id",    # geographic proxy → disparate-impact risk
    "application_dow",   # no economic rationale
]

TARGET = "default_flag"

# Aggressive binning for the improved model — more bins captures finer
# non-linearity, at modest overfit risk (mitigated by IV filtering).
IMPROVED_N_BINS = 20
IV_FILTER_THRESHOLD = 0.02  # Drop features that fail the conventional "weak" floor.


# ── Helpers ────────────────────────────────────────────────────────────────
def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Use the dataset's pre-built 'set' column to split."""
    return df[df["set"] == "train"].copy(), df[df["set"] == "test"].copy()


@dataclass
class FittedModel:
    """Everything the app and metrics layer need after fitting."""
    name: str
    pipeline: object
    feature_names_in: list[str]
    feature_names_out: list[str]
    y_train: np.ndarray
    y_test: np.ndarray
    y_score_train: np.ndarray
    y_score_test: np.ndarray
    iv_table: Optional[pd.DataFrame] = None  # populated for improved model only


# ══════════════════════════════════════════════════════════════════════════
# BASELINE
# ══════════════════════════════════════════════════════════════════════════
def _build_baseline_pipeline(numeric, categorical) -> Pipeline:
    """Median impute + scale numeric, fill-missing + one-hot categorical, L2 logistic."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_LABEL)),
        ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe,     numeric),
        ("cat", categorical_pipe, categorical),
    ])
    return Pipeline([
        ("prep",  preprocessor),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])


@st.cache_resource(show_spinner="Fitting baseline logistic regression...")
def fit_baseline(df: pd.DataFrame) -> FittedModel:
    train, test = split_train_test(df)
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_train, y_train = train[features], train[TARGET]
    X_test,  y_test  = test[features],  test[TARGET]

    pipe = _build_baseline_pipeline(NUMERIC_FEATURES, CATEGORICAL_FEATURES)
    pipe.fit(X_train, y_train)

    return FittedModel(
        name="baseline",
        pipeline=pipe,
        feature_names_in=features,
        feature_names_out=list(pipe.named_steps["prep"].get_feature_names_out()),
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        y_score_train=pipe.predict_proba(X_train)[:, 1],
        y_score_test=pipe.predict_proba(X_test)[:, 1],
    )


# ══════════════════════════════════════════════════════════════════════════
# IMPROVED MODEL — WoE pipeline
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class WoEEncoder:
    """
    Per-feature WoE encoder. Fit on training data once; apply to any data.

    Attributes:
        feature:        column name
        bin_mapping:    dict mapping bin-label-as-string → WoE value
        is_categorical: True if the feature was binned as categorical
        bin_edges:      numpy edges for numeric features (qcut output, monotonic)
        iv:             total Information Value (fitted on training data)
        default_woe:    fallback WoE used for unseen categories at transform time
    """
    feature: str
    bin_mapping: dict
    is_categorical: bool
    bin_edges: Optional[np.ndarray] = None
    iv: float = 0.0
    default_woe: float = 0.0


def fit_woe_encoder(
    df: pd.DataFrame, feature: str, target: str = TARGET, n_bins: int = IMPROVED_N_BINS
) -> WoEEncoder:
    """
    Fit a WoE encoder on training data using the same binning rules as core.eda.

    Returns a WoEEncoder that can transform any future data via .transform().
    """
    s = df[feature]
    y = df[target]

    is_categorical = (
        not pd.api.types.is_numeric_dtype(s)
        or s.nunique(dropna=True) <= 12
    )

    binned = _bin_feature(s, n_bins=n_bins)

    # Capture bin edges for numeric features so transform() can re-bin new data
    bin_edges = None
    if not is_categorical:
        non_null = s.notna()
        if non_null.sum() > 0:
            try:
                _, edges = pd.qcut(s[non_null], q=n_bins, duplicates="drop", retbins=True)
                bin_edges = edges
            except ValueError:
                bin_edges = None

    # Compute WoE per bin (same logic as compute_woe_iv, kept self-contained)
    df_g = pd.DataFrame({"bin": binned, "y": y.values})
    agg = df_g.groupby("bin", observed=True).agg(
        count=("y", "size"), bads=("y", "sum")
    ).reset_index()
    agg["goods"] = agg["count"] - agg["bads"]

    g_smooth = agg["goods"] + WOE_EPSILON
    b_smooth = agg["bads"]  + WOE_EPSILON
    pct_g = g_smooth / g_smooth.sum()
    pct_b = b_smooth / b_smooth.sum()
    woe   = np.log(pct_g / pct_b)
    iv    = float(((pct_g - pct_b) * woe).sum())

    bin_mapping = dict(zip(agg["bin"].astype(str), woe.values))

    return WoEEncoder(
        feature=feature,
        bin_mapping=bin_mapping,
        is_categorical=is_categorical,
        bin_edges=bin_edges,
        iv=iv,
        default_woe=0.0,  # neutral fallback for unseen categories
    )


def _bin_with_edges(series: pd.Series, edges: np.ndarray) -> pd.Series:
    """Bin a numeric series using pre-fitted edges. Used by transform()."""
    non_null = series.notna()
    out = pd.Series([MISSING_LABEL] * len(series), index=series.index, dtype="object")
    if non_null.sum() == 0 or edges is None:
        return out
    cut = pd.cut(series[non_null], bins=edges, include_lowest=True, duplicates="drop")
    out[non_null] = cut.astype(str)
    return out


def transform_with_woe(series: pd.Series, encoder: WoEEncoder) -> pd.Series:
    """Apply a fitted WoE encoder to a (possibly new) series."""
    if encoder.is_categorical:
        binned = series.astype("object").fillna(MISSING_LABEL).astype(str)
    elif encoder.bin_edges is not None:
        binned = _bin_with_edges(series, encoder.bin_edges)
    else:
        binned = series.astype("object").fillna(MISSING_LABEL).astype(str)

    # Map each bin label → WoE value, with default_woe for unseen bins/categories
    return binned.map(encoder.bin_mapping).fillna(encoder.default_woe).astype(float)


@st.cache_resource(show_spinner="Fitting WoE-engineered logistic regression...")
def fit_improved(df: pd.DataFrame) -> FittedModel:
    """
    Improved model pipeline:
      1. Fit a WoEEncoder per candidate feature on TRAINING data only
      2. Filter out features with IV below IV_FILTER_THRESHOLD
      3. Transform train + test using the fitted encoders
      4. Fit L2 logistic regression on the WoE-encoded matrix
    """
    train, test = split_train_test(df)
    candidate_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    # 1 + 2. Fit encoders on train, retain those that pass IV filter
    encoders = {}
    iv_records = []
    for feat in candidate_features:
        enc = fit_woe_encoder(train, feat, target=TARGET, n_bins=IMPROVED_N_BINS)
        iv_records.append({
            "feature": feat,
            "iv": enc.iv,
            "kept": enc.iv >= IV_FILTER_THRESHOLD,
        })
        if enc.iv >= IV_FILTER_THRESHOLD:
            encoders[feat] = enc

    iv_table = pd.DataFrame(iv_records).sort_values("iv", ascending=False).reset_index(drop=True)

    if not encoders:
        raise RuntimeError("No features passed the IV filter threshold.")

    selected = list(encoders.keys())

    # 3. Transform train and test
    X_train_woe = pd.DataFrame({
        f: transform_with_woe(train[f], encoders[f]) for f in selected
    })
    X_test_woe = pd.DataFrame({
        f: transform_with_woe(test[f], encoders[f]) for f in selected
    })

    # 4. Fit logistic regression
    model = LogisticRegression(max_iter=2000, random_state=42, C=1.0)
    model.fit(X_train_woe, train[TARGET])

    # Wrap in a thin pipeline-like object so callers can use .predict_proba(X)
    class _WoEPipeline:
        def __init__(self, encoders, model):
            self.encoders = encoders
            self.model = model
            self.feature_order = list(encoders.keys())
            self.named_steps = {"model": model}  # for get_coefficients compatibility

        def predict_proba(self, X):
            Xw = pd.DataFrame({
                f: transform_with_woe(X[f], self.encoders[f])
                for f in self.feature_order
            })
            return self.model.predict_proba(Xw)

    pipeline = _WoEPipeline(encoders, model)

    return FittedModel(
        name="improved",
        pipeline=pipeline,
        feature_names_in=selected,
        feature_names_out=selected,
        y_train=train[TARGET].to_numpy(),
        y_test=test[TARGET].to_numpy(),
        y_score_train=pipeline.predict_proba(train[selected])[:, 1],
        y_score_test=pipeline.predict_proba(test[selected])[:, 1],
        iv_table=iv_table,
    )


# ══════════════════════════════════════════════════════════════════════════
# Coefficient inspection
# ══════════════════════════════════════════════════════════════════════════
def get_coefficients(fitted: FittedModel) -> pd.DataFrame:
    """Tidy coefficient table. Works for both baseline and improved models."""
    coef = fitted.pipeline.named_steps["model"].coef_.ravel()
    intercept = float(fitted.pipeline.named_steps["model"].intercept_[0])

    df = pd.DataFrame({
        "feature": fitted.feature_names_out,
        "coefficient": coef,
        "abs_coefficient": np.abs(coef),
        "direction": np.where(coef > 0, "↑ risk", "↓ risk"),
    })
    df = df.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    df.attrs["intercept"] = intercept
    return df


def predict_proba(fitted: FittedModel, X: pd.DataFrame) -> np.ndarray:
    return fitted.pipeline.predict_proba(X)[:, 1]
