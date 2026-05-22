"""Bivariate Explorer. Two features at once, with target as colour."""
import streamlit as st
import plotly.express as px
from utils.data_loader import load_loan_book, get_feature_columns, col_label

st.title("Bivariate Explorer")
st.caption("Compare two features to surface interactions and subgroup effects.")

df = load_loan_book()
cols = get_feature_columns()
all_features = cols["numeric"] + cols["categorical"]

c1, c2, c3 = st.columns(3)
with c1:
    x_feature = st.selectbox("X axis", all_features, index=all_features.index("annual_income"), format_func=col_label)
with c2:
    y_feature = st.selectbox(
        "Y axis", all_features, index=all_features.index("dti_ratio"), format_func=col_label
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
        labels={"default_flag": "Default Flag", x_feature: col_label(x_feature), y_feature: col_label(y_feature)},
        title=f"{col_label(x_feature)} vs {col_label(y_feature)} (sample of {sample_size:,} rows)",
    )
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Box (by target)":
    fig = px.box(
        df,
        x="default_flag",
        y=y_feature,
        color="default_flag",
        color_discrete_map={0: "#10B981", 1: "#EF4444"},
        labels={"default_flag": "Default Flag", y_feature: col_label(y_feature)},
        title=f"{col_label(y_feature)} by Default Status",
    )
    st.plotly_chart(fig, use_container_width=True)

elif plot_type == "Heatmap":
    numeric_cols = cols["numeric"]
    corr = df[numeric_cols].corr()
    corr.columns = [col_label(c) for c in corr.columns]
    corr.index = [col_label(c) for c in corr.index]
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1,
        title="Correlation Heatmap (Numeric Features)",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)
