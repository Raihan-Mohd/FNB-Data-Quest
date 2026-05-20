"""
Modelling artefacts. Owner: Person A.

Loads / trains the baseline and improved logistic regression models and exposes
prediction and coefficient inspection functions.

Person B notes:
- Models should be fit on train, scored on test.
- Cache fitted models with @st.cache_resource.
- Surface coefficients in a tidy DataFrame so the view can render them.
"""
from __future__ import annotations
import pandas as pd
import numpy as np
import streamlit as st


@st.cache_resource
def fit_baseline_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit the baseline logistic regression. No feature engineering beyond raw inputs.

    TODO (Person A): implement.
    Returns: fitted sklearn model.
    """
    raise NotImplementedError("Person A: implement baseline logistic regression.")


@st.cache_resource
def fit_improved_logistic(X_train: pd.DataFrame, y_train: pd.Series):
    """
    Fit the improved logistic regression using WoE-transformed features.

    TODO (Person A): implement.
    """
    raise NotImplementedError("Person A: implement improved logistic regression.")


def get_coefficients(model, feature_names: list[str]) -> pd.DataFrame:
    """
    Return a tidy coefficient table for display.

    Columns: feature, coefficient, abs_coefficient, direction.
    """
    coef = model.coef_.ravel()
    return (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": coef,
            "abs_coefficient": np.abs(coef),
            "direction": np.where(coef > 0, "↑ risk", "↓ risk"),
        })
        .sort_values("abs_coefficient", ascending=False)
        .reset_index(drop=True)
    )


def predict_proba(model, X: pd.DataFrame) -> np.ndarray:
    """Predicted probability of default."""
    return model.predict_proba(X)[:, 1]
