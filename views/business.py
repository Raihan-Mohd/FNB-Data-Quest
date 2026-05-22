"""Business Decision Dashboard — approval policy, volume vs risk, expected loss in £."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.pipeline import train_models
from utils.data_loader import load_loan_book

TEAL  = "#28949C"
DARK  = "#082D2F"
LIGHT = "#65CED1"
GREEN = "#10B981"
RED   = "#EF4444"
AMBER = "#F59E0B"
MUTED = "#94A3B8"
WHITE = "#FFFFFF"

st.title("Business Decision Dashboard")
st.caption("Translate model output into an approval policy. Volume, risk, and expected loss in £.")

df_raw   = load_loan_book()
models   = train_models(df_raw)
probs    = models["probs_test"]
actual   = models["y_test"]
n_total  = models["n_test"]
avg_loan = models["avg_loan"]

# ── Controls ───────────────────────────────────────────────────────────────
ctrl1, ctrl2 = st.columns(2)
with ctrl1:
    threshold = st.slider("Approval threshold — approve if predicted PD below this",
        0.05, 0.95, 0.20, 0.01)
with ctrl2:
    lgd = st.slider("Loss given default (LGD)", 0.10, 1.00, 0.45, 0.05,
        help="Share of loan value lost when a default occurs.")

st.divider()

# ── Live KPIs ──────────────────────────────────────────────────────────────
approved     = probs < threshold
n_approved   = approved.sum()
bad_approved = int((approved & (actual == 1)).sum())
bad_rejected = int((~approved & (actual == 1)).sum())
portfolio_dr = actual[approved].mean() if n_approved > 0 else 0.0
exp_loss_tot = bad_approved * lgd * avg_loan
exp_loss_pct = portfolio_dr * lgd * 100

k1, k2, k3, k4 = st.columns(4)
k1.metric("Approved",          f"{n_approved:,}",       f"{n_approved/n_total:.1%} of applicants")
k2.metric("Portfolio DR",      f"{portfolio_dr*100:.1f}%")
k3.metric("Defaults blocked",  f"{bad_rejected:,}",     f"{bad_rejected/max(actual.sum(),1):.1%} of all defaults")
k4.metric("Expected loss (£)", f"£{exp_loss_tot/1e6:.2f}M",
    help=f"Defaults in approved book × LGD × avg loan £{avg_loan:,.0f}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# VOLUME vs RISK vs EXPECTED LOSS  (3 Y-axis options — Plotly toggle)
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Volume vs Risk trade-off")
st.caption("Use the buttons inside the chart to switch the Y-axis metric.")

thresholds  = models["thresholds_curve"]
vol_curve   = models["vol_curve"]
risk_curve  = models["risk_curve"]
avoid_curve = models["avoided_curve"]
exp_loss_curve = risk_curve * lgd * avg_loan * vol_curve * n_total / 1e6  # in £M

fig_vr = go.Figure()
# Trace 0: Approval rate (default visible)
fig_vr.add_trace(go.Scatter(x=thresholds, y=vol_curve*100, mode="lines",
    name="Approval rate (%)", line=dict(color=TEAL, width=2.5), visible=True))
# Trace 1: Portfolio DR (default visible)
fig_vr.add_trace(go.Scatter(x=thresholds, y=risk_curve*100, mode="lines",
    name="Portfolio default rate (%)", line=dict(color=RED, width=2.5), visible=True))
# Trace 2: Defaults blocked (hidden)
fig_vr.add_trace(go.Scatter(x=thresholds, y=avoid_curve*100, mode="lines",
    name="Defaults blocked (%)", line=dict(color=GREEN, width=2.5), visible=False))
# Trace 3: Expected loss £M (hidden)
fig_vr.add_trace(go.Scatter(x=thresholds, y=exp_loss_curve, mode="lines",
    name="Expected loss (£M)", line=dict(color=AMBER, width=2.5), visible=False))

fig_vr.add_vline(x=threshold, line_dash="dot", line_color=DARK,
    annotation_text=f"Current: {threshold:.2f}", annotation_position="top right")

fig_vr.update_layout(
    xaxis_title="Approval threshold",
    yaxis_title="Rate (%)",
    plot_bgcolor=WHITE, paper_bgcolor=WHITE,
    legend=dict(orientation="h", y=1.15), height=400,
    updatemenus=[dict(
        type="buttons", direction="right", x=0.0, y=1.22,
        buttons=[
            dict(label="Volume & Risk",
                 method="update",
                 args=[{"visible": [True, True, False, False]},
                       {"yaxis.title.text": "Rate (%)"}]),
            dict(label="Defaults blocked",
                 method="update",
                 args=[{"visible": [True, False, True, False]},
                       {"yaxis.title.text": "Rate (%)"}]),
            dict(label="Expected loss £",
                 method="update",
                 args=[{"visible": [True, False, False, True]},
                       {"yaxis.title.text": "Expected loss (£M)"}]),
        ]
    )]
)
st.plotly_chart(fig_vr, use_container_width=True)
st.divider()

# ══════════════════════════════════════════════════════════════════════════
# PRECISION vs RECALL
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Precision vs Recall")
col1, col2 = st.columns([3, 2])
with col1:
    fig_pr = go.Figure()
    fig_pr.add_trace(go.Scatter(x=models["rec_arr"], y=models["prec_arr"],
        mode="lines", line=dict(color=TEAL, width=2.5)))
    fig_pr.update_layout(
        xaxis_title="Recall — share of all defaults caught",
        yaxis_title="Precision — of those flagged, share that default",
        plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=350, showlegend=False)
    st.plotly_chart(fig_pr, use_container_width=True)
with col2:
    st.markdown("**Precision:** of applicants we rejected, how many were genuinely risky? Low precision = good customers wrongly rejected.")
    st.markdown("**Recall:** of all actual defaults, how many did we catch? Low recall = bad loans issued.")
    st.markdown("Moving along the curve trades one error type for the other. The threshold slider above drives this.")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# POLICY COMPARISON TABLE — with expected loss in £
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Three policy scenarios")
rows = []
for t, name in [(0.15,"Conservative"),(0.25,"Balanced ✓"),(0.40,"Aggressive")]:
    app  = probs < t
    n    = int(app.sum())
    dr   = float(actual[app].mean()) if n > 0 else 0.0
    blk  = int((~app & (actual==1)).sum())
    slip = int((app & (actual==1)).sum())
    exp_total = slip * lgd * avg_loan
    book_size = n * avg_loan
    rows.append({
        "Policy":            name,
        "Threshold":         t,
        "Approved":          f"{n:,} ({n/n_total:.1%})",
        "Portfolio DR":      f"{dr*100:.1f}%",
        "Defaults blocked":  f"{blk:,} ({blk/max(actual.sum(),1):.1%})",
        "Expected loss (£)": f"£{exp_total/1e6:.1f}M",
        "Book size (£)":     f"£{book_size/1e6:.0f}M",
        "Loss / book":       f"{exp_total/book_size*100:.1f}%",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.info(
    "**Recommended starting point: threshold 0.20–0.25.** "
    "Approves ~73–80% of applicants, holds portfolio DR below 10%, "
    "blocks over half of all defaults. A/B test before full deployment.",
    icon="💡",
)
