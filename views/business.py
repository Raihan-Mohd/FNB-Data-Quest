"""Business Decision Dashboard. Volume vs risk trade-off, approval policy simulation."""
import streamlit as st

st.title("Business Decision Dashboard")
st.caption("Translate model performance into approval policy. Volume, risk, and trade-offs.")

st.info(
    "This page is scaffolded for the bonus task. It depends on a fitted model and the "
    "`approval_strategy_table` helper in `core/metrics.py`.",
    icon="🚧",
)

# Policy controls
st.subheader("Approval policy")
c1, c2 = st.columns(2)
with c1:
    threshold = st.slider(
        "Approval threshold (predicted default probability)",
        min_value=0.05, max_value=0.95, value=0.20, step=0.01,
        help="Applicants with predicted PD below this are approved.",
    )
with c2:
    loss_given_default = st.slider(
        "Loss given default (LGD)",
        min_value=0.10, max_value=1.00, value=0.45, step=0.05,
        help="Share of loan amount expected to be lost when a default occurs.",
    )

st.divider()

# KPI strip — placeholders
k1, k2, k3, k4 = st.columns(4)
k1.metric("Approval rate", "—")
k2.metric("Approved default rate", "—")
k3.metric("Expected loss / loan", "—")
k4.metric("Volume change vs baseline", "—")

st.divider()

# Two charts side by side
c1, c2 = st.columns(2)
with c1:
    st.subheader("Volume vs Risk")
    st.write("Approval rate (x) vs approved-portfolio default rate (y), curve over thresholds.")
    st.empty()  # Person A: call core.metrics.approval_strategy_table and chart it

with c2:
    st.subheader("Precision vs Recall — business view")
    st.write(
        "**Precision:** of customers we flagged as risky, how many really were? "
        "Affects how many profitable applicants we reject.\n\n"
        "**Recall:** of customers who actually defaulted, how many did we catch? "
        "Affects how much loss we absorbed."
    )
    st.empty()
