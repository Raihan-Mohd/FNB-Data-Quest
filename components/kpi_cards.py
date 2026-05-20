"""Reusable KPI cards. Headline metrics shown as a row of large numbers."""
import streamlit as st


def render_kpi_row(kpis: list[dict]):
    """
    Render a row of KPI cards.

    Each kpi dict supports: label, value, delta (optional), help (optional).
    Example: {"label": "AUC", "value": "0.82", "delta": "+0.14 vs baseline"}
    """
    cols = st.columns(len(kpis))
    for col, kpi in zip(cols, kpis):
        with col:
            st.metric(
                label=kpi["label"],
                value=kpi["value"],
                delta=kpi.get("delta"),
                help=kpi.get("help"),
            )
