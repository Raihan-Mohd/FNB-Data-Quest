"""Modelling page — baseline vs improved logistic regression."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pipeline import train_models, ENGINEERING_DECISIONS
from core.metrics import confusion_matrix_at_threshold
from utils.data_loader import load_loan_book

C_BLUE  = "#2563EB"
C_MUTED = "#94A3B8"
C_GREEN = "#10B981"
C_RED   = "#EF4444"

st.title("Modelling")
st.caption("Baseline vs improved logistic regression — every engineering decision justified by EDA.")

df_raw  = load_loan_book()
models  = train_models(df_raw)

auc_i = models["impr_test_auc"]
gini  = 2 * auc_i - 1
pct   = (auc_i - 0.68) / (0.82 - 0.68) * 100

# ── KPI row ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Given Baseline AUC", "0.68",        help="Competition reference model.")
c2.metric("Our Baseline AUC",   f"{models['base_test_auc']:.4f}", help="Raw numeric features, mean impute, scale.")
c3.metric("Our Improved AUC",   f"{auc_i:.4f}", delta=f"+{auc_i - 0.68:.4f} vs given", help="32 engineered features.")
c4.metric("% to LightGBM",      f"{pct:.0f}%",  help="Gap closed vs 0.82 ceiling.")

st.caption(f"Gini: **{gini:.4f}** · Train AUC: **{models['impr_train_auc']:.4f}** · Train-test gap: **{models['impr_train_auc'] - auc_i:+.4f}**")
st.divider()

# ── ROC curve ──────────────────────────────────────────────────────────────
st.subheader("ROC Curve")
fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(
    x=models["fpr"], y=models["tpr"], mode="lines",
    name=f"Improved model (AUC = {auc_i:.4f})",
    line=dict(color=C_BLUE, width=3),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 1], y=[0, 1], mode="lines", name="Random (AUC = 0.50)",
    line=dict(color=C_MUTED, dash="dot"),
))
fig_roc.add_trace(go.Scatter(
    x=[0, 0, 1], y=[0, 1, 1], mode="lines", name="Perfect (AUC = 1.00)",
    line=dict(color=C_GREEN, dash="dash"),
))
fig_roc.update_layout(
    xaxis_title="False Positive Rate",
    yaxis_title="True Positive Rate (Recall)",
    plot_bgcolor="white", paper_bgcolor="white", height=400,
    legend=dict(x=0.55, y=0.05),
)
st.plotly_chart(fig_roc, use_container_width=True)
st.divider()

# ── Threshold → confusion matrix ───────────────────────────────────────────
st.subheader("Decision threshold")
threshold = st.slider(
    "Approve applicants with predicted PD **below** this threshold",
    min_value=0.05, max_value=0.95, value=0.20, step=0.01,
)

probs  = models["probs_test"]
actual = models["y_test"]
approved    = probs < threshold
n_approved  = approved.sum()
n_total     = len(actual)
bad_approved = int((approved & (actual == 1)).sum())
bad_rejected = int((~approved & (actual == 1)).sum())

ka, kb, kc, kd = st.columns(4)
ka.metric("Approved",         f"{n_approved:,}",  f"{n_approved/n_total:.1%} of applicants")
kb.metric("Portfolio DR",     f"{actual[approved].mean()*100:.1f}%", help="Default rate among approved applicants.")
kc.metric("Defaults blocked", f"{bad_rejected:,}", f"{bad_rejected/max(actual.sum(),1):.1%} of all defaults")
kd.metric("Defaults slipping",f"{bad_approved:,}", f"{bad_approved/max(actual.sum(),1):.1%} of all defaults")

cm = confusion_matrix_at_threshold(actual, probs, threshold)
labels = ["Non-default (0)", "Default (1)"]
fig_cm = go.Figure(go.Heatmap(
    z=cm, x=[f"Pred: {l}" for l in labels], y=[f"Actual: {l}" for l in labels],
    text=cm, texttemplate="%{text:,}",
    colorscale="Blues", showscale=False,
))
fig_cm.update_layout(title=f"Confusion matrix at threshold {threshold:.2f}", height=300,
                     plot_bgcolor="white", paper_bgcolor="white")
st.plotly_chart(fig_cm, use_container_width=True)
st.divider()

# ── Coefficients ───────────────────────────────────────────────────────────
st.subheader("Top 20 feature coefficients")
st.caption("Negative = associated with lower default risk. Positive = higher risk.")
coef_df = models["coef_df"].copy()
colours = [C_RED if v > 0 else C_GREEN for v in coef_df["coef"]]
fig_coef = go.Figure(go.Bar(
    x=coef_df["coef"], y=coef_df["feature"],
    orientation="h", marker_color=colours,
    hovertemplate="%{y}: %{x:.4f}<extra></extra>",
))
fig_coef.update_layout(
    xaxis_title="Coefficient", yaxis={"categoryorder": "total ascending"},
    plot_bgcolor="white", paper_bgcolor="white", height=500,
)
st.plotly_chart(fig_coef, use_container_width=True)

st.info(
    "**Ever Delinquent** and **Months Since Last Delinquency** are the two dominant features — "
    "both derived by treating 49.9% missingness as a signal rather than noise.",
)
st.divider()

# ── Engineering decisions ──────────────────────────────────────────────────
st.subheader("Engineering decisions")
eng_df = pd.DataFrame(
    ENGINEERING_DECISIONS,
    columns=["Step", "Justification (from EDA)", "Impact"],
)
st.dataframe(eng_df, use_container_width=True, hide_index=True)
