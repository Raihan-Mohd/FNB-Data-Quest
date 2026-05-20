"""Modelling page. ROC curves, coefficients, confusion matrix at a chosen threshold."""
import streamlit as st
from components.kpi_cards import render_kpi_row

st.title("Modelling")
st.caption("Compare baseline vs improved logistic regression. Inspect coefficients and trade-offs.")

st.info(
    "This page depends on `core/modelling.py` and `core/metrics.py`, which are currently stubs. "
    "Once Person A wires up the models, the placeholders below will be populated.",
    icon="🚧",
)

# Headline KPIs — placeholders until models are wired in
render_kpi_row([
    {"label": "Baseline AUC", "value": "0.68", "help": "Project benchmark."},
    {"label": "Improved AUC", "value": "—", "delta": None, "help": "Will populate once model is fit."},
    {"label": "Gini", "value": "—"},
    {"label": "LightGBM ceiling", "value": "0.82"},
])

st.divider()

# Threshold slider (drives confusion matrix + precision/recall in the wired version)
threshold = st.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help="Applicants with predicted default probability ≥ threshold are flagged as risky.",
)

c1, c2 = st.columns(2)

with c1:
    st.subheader("ROC curves")
    st.write("Baseline vs Improved (placeholder).")
    # Person A: call components.charts.roc_curve_chart with both curves
    st.empty()

with c2:
    st.subheader("Confusion matrix")
    st.write(f"At threshold = {threshold:.2f} (placeholder).")
    # Person A: call core.metrics.confusion_matrix_at_threshold and components.charts.confusion_matrix_chart
    st.empty()

st.divider()

st.subheader("Coefficients")
st.write("Once the improved model is fit, this section will display the tidy coefficient table from `core.modelling.get_coefficients`.")
