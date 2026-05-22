"""Univariate Explorer — distributions, default rates by decile, WoE / IV."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.eda import compute_woe_iv, default_rate_by_category
from components.charts import histogram_by_target, woe_bar_chart
from utils.data_loader import load_loan_book, get_feature_columns

TEAL  = "#28949C"
DARK  = "#082D2F"
GREEN = "#10B981"
RED   = "#EF4444"
MUTED = "#94A3B8"
WHITE = "#FFFFFF"

st.title("Univariate Explorer")
st.caption("Inspect one feature at a time — distribution, default rate by decile, and WoE / IV.")

df   = load_loan_book()
cols = get_feature_columns()

# ── Feature selector ───────────────────────────────────────────────────────
sel_col, ctrl_col = st.columns([1, 3])
with sel_col:
    ftype = st.radio("Feature type", ["Numeric", "Categorical"], horizontal=True)
    if ftype == "Numeric":
        feature = st.selectbox("Feature", cols["numeric"])
        n_deciles = st.select_slider("Deciles", [5, 10, 20], value=10,
            help="Number of equally-sized groups for the default rate chart.")
        woe_bins  = st.slider("WoE bins", 5, 20, 10)
    else:
        feature = st.selectbox("Feature", cols["categorical"])
        woe_bins  = 10

with ctrl_col:
    if ftype == "Numeric":
        st.plotly_chart(histogram_by_target(df, feature), use_container_width=True)
    else:
        rate_table = default_rate_by_category(df, feature)
        fig = go.Figure(go.Bar(
            x=rate_table["category"].astype(str),
            y=rate_table["default_rate"],
            marker_color=TEAL,
            text=rate_table["default_rate"].map(lambda x: f"{x:.1%}"),
            textposition="outside",
        ))
        fig.update_layout(
            title=f"Default rate by {feature}",
            xaxis_title=feature, yaxis_title="Default rate",
            yaxis_tickformat=".1%",
            plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380,
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# DEFAULT RATE BY DECILE — with Y-axis toggle
# ══════════════════════════════════════════════════════════════════════════
if ftype == "Numeric":
    st.subheader(f"Default rate by decile — {feature}")

    y_metric = st.radio("Y axis",
        ["Default rate (%)", "Event count", "Default count"],
        horizontal=True, key="decile_y")

    col = df[feature].copy()
    try:
        df["_decile"] = pd.qcut(col, q=n_deciles, duplicates="drop", labels=False)
    except Exception:
        st.warning("Cannot create deciles for this feature (too few unique values).")
        df["_decile"] = pd.cut(col, bins=n_deciles, duplicates="drop", labels=False)

    dec_agg = df.groupby("_decile", observed=False).agg(
        count=("default_flag", "size"),
        defaults=("default_flag", "sum"),
        default_rate=("default_flag", "mean"),
        midpoint=(feature, "median"),
    ).reset_index()

    dec_agg["decile_label"] = (
        dec_agg["_decile"].astype(int) + 1
    ).astype(str).str.zfill(2)

    y_col_map = {
        "Default rate (%)": ("default_rate", lambda x: f"{x:.1%}", "Default Rate (%)"),
        "Event count":      ("count",        lambda x: f"{x:,}",   "Total Applications"),
        "Default count":    ("defaults",     lambda x: f"{x:,}",   "Number of Defaults"),
    }
    y_col, fmt_fn, y_label = y_col_map[y_metric]
    y_vals = dec_agg[y_col].values
    if y_col == "default_rate":
        y_display = y_vals * 100
        bar_colors = [RED if v > df["default_flag"].mean() else GREEN for v in y_vals]
    else:
        y_display = y_vals
        bar_colors = TEAL

    fig_dec = go.Figure()
    fig_dec.add_trace(go.Bar(
        x=dec_agg["decile_label"],
        y=y_display,
        marker_color=bar_colors,
        text=[fmt_fn(v) for v in y_vals],
        textposition="outside",
        hovertext=[f"Median {feature}: {m:.2f}<br>Count: {c:,}<br>DR: {d:.1%}"
                   for m, c, d in zip(dec_agg["midpoint"], dec_agg["count"], dec_agg["default_rate"])],
        hoverinfo="text",
    ))
    if y_col == "default_rate":
        fig_dec.add_hline(y=df["default_flag"].mean()*100, line_dash="dot",
            line_color=MUTED, annotation_text=f"Population DR {df['default_flag'].mean()*100:.1f}%",
            annotation_position="right")

    fig_dec.update_layout(
        xaxis_title=f"Decile (1 = lowest {feature}  →  {n_deciles} = highest)",
        yaxis_title=y_label,
        plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=380,
    )
    st.plotly_chart(fig_dec, use_container_width=True)
    df.drop(columns=["_decile"], inplace=True)
    st.divider()

# ══════════════════════════════════════════════════════════════════════════
# WoE / IV section
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Weight of Evidence & Information Value")

def classify_iv(v):
    if v < 0.02: return "🔴 Not predictive"
    if v < 0.10: return "🟡 Weak"
    if v < 0.30: return "🟢 Medium"
    if v < 0.50: return "🟢 Strong"
    return "⚠️ Suspicious — check for leakage"

try:
    woe_table, iv = compute_woe_iv(df, feature, bins=woe_bins)
    col_iv, col_chart = st.columns([1, 3])
    with col_iv:
        st.metric("Information Value", f"{iv:.4f}")
        st.caption(classify_iv(iv))
    with col_chart:
        fig_woe = woe_bar_chart(woe_table, feature)
        st.plotly_chart(fig_woe, use_container_width=True)
    with st.expander("WoE table detail"):
        disp = [c for c in ["bin","total","events","non_events","woe","iv","default_rate"] if c in woe_table.columns]
        st.dataframe(
            woe_table[disp].rename(columns={"events":"bads","non_events":"goods"})
              .style.format({"woe":"{:.4f}","iv":"{:.4f}","default_rate":"{:.2%}"}),
            use_container_width=True,
        )
except Exception as e:
    st.warning(f"Could not compute WoE for {feature}: {e}")

st.divider()
with st.expander("Summary statistics"):
    if ftype == "Numeric":
        st.write(df[feature].describe())
    else:
        st.write(df[feature].value_counts(dropna=False))
