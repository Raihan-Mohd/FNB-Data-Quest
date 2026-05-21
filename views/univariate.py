"""Univariate Explorer. One feature at a time, with target overlay and (eventually) WoE/IV."""
import streamlit as st
from utils.data_loader import load_loan_book, get_feature_columns
from components.charts import histogram_by_target
from core.eda import default_rate_by_category

st.title("Univariate Explorer")
st.caption("Inspect one feature at a time. See how its distribution differs between defaulters and non-defaulters.")

df = load_loan_book()
cols = get_feature_columns()

# Feature picker
left, right = st.columns([1, 3])
with left:
    feature_type = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True)
    if feature_type == "Numeric":
        feature = st.selectbox("Feature", cols["numeric"])
    else:
        feature = st.selectbox("Feature", cols["categorical"])

with right:
    # Distribution chart
    if feature_type == "Numeric":
        st.plotly_chart(histogram_by_target(df, feature), use_container_width=True)
    else:
        rate_table = default_rate_by_category(df, feature)
        st.bar_chart(rate_table.set_index("category")["default_rate"])
        st.caption("Default rate by category.")

st.divider()

# WoE/IV placeholder - Person A will fill in compute_woe_iv in core/eda.py
st.subheader("Weight of Evidence & Information Value")
st.info(
    "WoE/IV computation lives in `core/eda.py::compute_woe_iv` and is currently a stub. "
    "Once Person A implements it, this section will render a WoE bar chart and IV summary.",
    icon="🚧",
)

# Summary statistics
with st.expander("Summary statistics"):
    if feature_type == "Numeric":
        st.write(df[feature].describe())
    else:
        st.write(df[feature].value_counts(dropna=False))
