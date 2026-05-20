"""Research page. Renders the five content markdown files as tabs."""
import streamlit as st
from utils.content_loader import load_markdown

st.title("Research")
st.caption("Foundations for interpretable credit modelling. Plain language first, technical precision second.")

tabs = st.tabs([
    "GLMs vs Non-Linear",
    "Interpretability vs Complexity",
    "WoE & IV",
    "Evaluation Metrics",
    "Regulatory Concerns",
    "References",
])

with tabs[0]:
    st.markdown(load_markdown("glm_vs_nonlinear"))

with tabs[1]:
    st.markdown(load_markdown("interpretability"))

with tabs[2]:
    st.markdown(load_markdown("woe_iv"))

with tabs[3]:
    st.markdown(load_markdown("metrics"))

with tabs[4]:
    st.markdown(load_markdown("regulatory"))

with tabs[5]:
    st.markdown(load_markdown("references"))
