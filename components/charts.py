"""
Reusable Plotly chart builders. Keep figures consistent across pages.

Convention: each function returns a plotly.graph_objects.Figure and does NOT call
st.plotly_chart. The caller decides where to render.
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np


# Shared theme tokens — keep visuals coherent across pages
COLOUR_PRIMARY = "#2563EB"
COLOUR_GOOD = "#10B981"
COLOUR_BAD = "#EF4444"
COLOUR_MUTED = "#94A3B8"


def histogram_by_target(df: pd.DataFrame, feature: str, target: str = "default_flag") -> go.Figure:
    """Overlapping histograms of `feature` split by the binary target."""
    fig = px.histogram(
        df,
        x=feature,
        color=target,
        barmode="overlay",
        opacity=0.65,
        nbins=40,
        color_discrete_map={0: COLOUR_GOOD, 1: COLOUR_BAD},
        labels={target: "Default"},
    )
    fig.update_layout(
        title=f"Distribution of {feature} by default status",
        xaxis_title=feature,
        yaxis_title="Count",
        legend_title_text="Default",
    )
    return fig


def woe_bar_chart(woe_table: pd.DataFrame, feature: str) -> go.Figure:
    """
    Bar chart of WoE per bin for a given feature.

    Expected columns in woe_table: ['bin', 'woe', 'count'].
    """
    fig = go.Figure(
        go.Bar(
            x=woe_table["bin"].astype(str),
            y=woe_table["woe"],
            marker_color=[
                COLOUR_GOOD if v > 0 else COLOUR_BAD for v in woe_table["woe"]
            ],
            hovertemplate="Bin: %{x}<br>WoE: %{y:.3f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLOUR_MUTED)
    fig.update_layout(
        title=f"Weight of Evidence by bin — {feature}",
        xaxis_title="Bin",
        yaxis_title="WoE",
    )
    return fig


def roc_curve_chart(
    curves: list[dict],
) -> go.Figure:
    """
    Plot one or more ROC curves on the same axes.

    Each entry in `curves` should be: {"name": str, "fpr": array, "tpr": array, "auc": float}.
    """
    fig = go.Figure()
    for c in curves:
        fig.add_trace(
            go.Scatter(
                x=c["fpr"],
                y=c["tpr"],
                mode="lines",
                name=f"{c['name']} (AUC = {c['auc']:.3f})",
            )
        )
    # Diagonal reference line
    fig.add_shape(
        type="line", x0=0, y0=0, x1=1, y1=1,
        line=dict(color=COLOUR_MUTED, dash="dot"),
    )
    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(range=[0, 1]),
        legend=dict(x=0.55, y=0.05),
    )
    return fig


def confusion_matrix_chart(cm: np.ndarray, labels=("Non-default", "Default")) -> go.Figure:
    """Heatmap of a 2x2 confusion matrix."""
    fig = go.Figure(
        go.Heatmap(
            z=cm,
            x=[f"Pred: {l}" for l in labels],
            y=[f"Actual: {l}" for l in labels],
            text=cm,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=False,
        )
    )
    fig.update_layout(title="Confusion Matrix", height=400)
    return fig
