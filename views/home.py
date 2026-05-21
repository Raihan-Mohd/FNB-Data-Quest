"""Home / landing page."""
import streamlit as st

st.title("DataQuest 2026 - Interpretable Credit Models")
st.caption("Retail lending analytics, scorecard-style logistic regression, business decision support.")

st.markdown(
    """
This application supports the DataQuest 2026 brief: investigate a simulated loan-book,
identify risk patterns, and build an interpretable logistic-regression credit model.

Use the sidebar to navigate.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Research")
    st.write(
        "Plain-language overview of GLMs vs non-linear models, WoE / IV, "
        "evaluation metrics, and regulatory feature concerns."
    )
with col2:
    st.subheader("Exploration")
    st.write(
        "Univariate and bivariate explorers, plus a data-quality report. "
        "WoE and IV surfaced where useful."
    )
with col3:
    st.subheader("Modelling")
    st.write(
        "Baseline vs improved logistic regression. Coefficient inspection, ROC, "
        "Gini, and a confusion matrix at a user-chosen threshold."
    )

st.divider()

st.subheader("Project benchmarks")
b1, b2, b3 = st.columns(3)
b1.metric("Baseline logistic AUC", "0.68", help="Older model on raw features.")
b2.metric("Reference LightGBM AUC", "0.82", help="Performance ceiling (not for submission).")
b3.metric("Target", "→ 0.82", help="Logistic regression + WoE feature engineering.")

st.info(
    "Final model constraint: must be a **logistic regression**. "
    "Non-linear models such as LightGBM are reference benchmarks only.",
    icon="ℹ️",
)
