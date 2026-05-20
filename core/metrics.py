"""
Model evaluation metrics. Owner: Person A.

Pure functions: given y_true and y_score (or y_pred), return metrics or curves.
View modules consume these and render with components/charts.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)


def compute_classification_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> dict:
    """All headline metrics at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_score)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "gini": 2 * auc - 1,
    }


def roc_curve_data(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    """Return FPR, TPR, AUC for plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    return {"fpr": fpr, "tpr": tpr, "auc": auc}


def confusion_matrix_at_threshold(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> np.ndarray:
    """Return the 2x2 confusion matrix at a given threshold."""
    y_pred = (y_score >= threshold).astype(int)
    return confusion_matrix(y_true, y_pred)


def approval_strategy_table(
    y_true: np.ndarray, y_score: np.ndarray, thresholds: np.ndarray = None
) -> pd.DataFrame:
    """
    For each decision threshold, compute approval rate, expected default rate,
    precision, and recall. Used by the business dashboard.
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for t in thresholds:
        approve = y_score < t  # lower predicted default prob => approve
        approval_rate = approve.mean()
        if approve.sum() > 0:
            approved_default_rate = y_true[approve].mean()
        else:
            approved_default_rate = np.nan
        rows.append({
            "threshold": t,
            "approval_rate": approval_rate,
            "approved_default_rate": approved_default_rate,
        })
    return pd.DataFrame(rows)
