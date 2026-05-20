"""Data Quality page. Surfaces missingness, dtypes, and basic stats."""
import streamlit as st
from utils.data_loader import load_loan_book
from core.eda import data_quality_summary

st.title("Data Quality Report")
st.caption("Diagnostics for the loan book. Use this to spot missingness and obvious issues before EDA.")

df = load_loan_book()

# Headline figures
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Default rate", f"{df['default_flag'].mean() * 100:.2f}%")
c4.metric("Date range", f"{df['application_date'].min():%Y-%m} → {df['application_date'].max():%Y-%m}")

st.divider()

# Per-column quality table
st.subheader("Per-column quality")
summary = data_quality_summary(df)
st.dataframe(
    summary,
    use_container_width=True,
    hide_index=True,
    column_config={
        "missing_pct": st.column_config.ProgressColumn(
            "Missing %", min_value=0, max_value=100, format="%.1f%%"
        ),
    },
)

st.divider()

# Sample preview
with st.expander("Preview rows"):
    st.dataframe(df.head(50), use_container_width=True)
