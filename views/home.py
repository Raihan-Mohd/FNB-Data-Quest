"""Home / landing page."""
import streamlit as st

st.title("DataQuest 2026 — Interpretable Credit Models")
st.caption("FNB Data Challenge · Retail lending analytics · Logistic regression scorecard")

st.markdown(
    "This application investigates a simulated loan book of **120,960 applications**, "
    "engineers features from EDA findings, and builds an interpretable logistic regression "
    "that closes **78% of the gap** between the given baseline and the LightGBM ceiling."
)

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Research")
    st.write("GLMs vs non-linear models, WoE / IV theory, all six evaluation metrics, and regulatory feature concerns.")
with col2:
    st.subheader("Exploration")
    st.write("Univariate and bivariate explorers with WoE / IV analysis, plus a data quality report.")
with col3:
    st.subheader("Modelling")
    st.write("Baseline vs improved logistic regression. Coefficients, ROC, Gini, confusion matrix.")

st.divider()

b1, b2, b3, b4 = st.columns(4)
b1.metric("Given Baseline AUC",  "0.68",  help="Competition reference.")
b2.metric("Our Improved AUC",    "0.79",  delta="+0.11")
b3.metric("LightGBM ceiling",    "0.82",  help="Non-linear benchmark — for reference only.")
b4.metric("Gini (improved)",     "0.58")

st.info(
    "Final model constraint: **logistic regression only**. "
    "LightGBM is shown as a reference ceiling, not the submission model.",
)
