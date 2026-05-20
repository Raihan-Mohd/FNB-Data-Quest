"""Bivariate Explorer. Two features at once, with target as colour."""
import streamlit as st
import plotly.express as px
from utils.data_loader import load_loan_book, get_feature_columns

st.title("Bivariate Explorer")
st.caption("Compare two features to surface interactions and subgroup effects.")

df = load_loan_book()
cols = get_feature_columns()
all_features = cols["numeric"] + cols["categorical"]

c1, c2, c3 = st.columns(3)
with c1:
    x_feature = st.selectbox("X axis", all_features, index=all_features.index("annual_income"))
with c2:
    y_feature = st.selectbox(
        "Y axis", all_features, index=all_features.index("dti_ratio")
    )
with c3:
    plot_type = st.selectbox("Plot type", ["Scatter", "Box (by target)", "Heatmap"])

st.divider()

# Optional sample to keep plotly snappy on large data
sample_size = min(10_000, len(df))
sample = df.sample(sample_size, random_state=42)

if plot_type == "Scatter":
    fig = px.scatter(
        sample,
        x=x_feature,
        y=y_feature,
        color="default_flag",
        opacity=0.5,
        color_continuous_scale=[[0, "#10B981"], [1, "#EF4444"]],
        title=f"{x_feature} vs {y_feature} (sample of {sample_size:,} rows)",
    )
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Box (by target)":
    fig = px.box(
        df,
        x="default_flag",
        y=y_feature,
        color="default_flag",
        color_discrete_map={0: "#10B981", 1: "#EF4444"},
        title=f"{y_feature} by default status",
    )
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Heatmap":
    # Numeric correlation heatmap (ignores selected x/y for now — Person A can refine)
    numeric_cols = cols["numeric"]
    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation heatmap (numeric features)",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)
