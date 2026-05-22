"""Data Quality page — missingness, delinquency signal callout, column diagnostics."""
import plotly.graph_objects as go
import streamlit as st

from core.eda import data_quality_summary
from utils.data_loader import load_loan_book

TEAL  = "#28949C"
DARK  = "#082D2F"
LIGHT = "#65CED1"
RED   = "#EF4444"
GREEN = "#10B981"
MUTED = "#94A3B8"
WHITE = "#FFFFFF"

st.title("Data Quality Report")
st.caption("Diagnostics for the loan book — missingness, dtypes, and the key EDA signal hidden in the data.")

df = load_loan_book()

# ── Headline KPIs ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", len(df.columns))
c3.metric("Default rate", f"{df['default_flag'].mean()*100:.1f}%")
c4.metric("Date range",
    f"{df['application_date'].min()[:7]} → {df['application_date'].max()[:7]}")

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# DELINQUENCY CALLOUT — the key EDA signal
# ══════════════════════════════════════════════════════════════════════════
missing_mask      = df["months_since_last_delinquency"].isna()
missing_pct       = missing_mask.mean() * 100
never_delinq_dr   = df[missing_mask]["default_flag"].mean() * 100
ever_delinq_dr    = df[~missing_mask]["default_flag"].mean() * 100
gap               = ever_delinq_dr - never_delinq_dr

st.subheader("Key EDA signal — missingness as a predictor")
col_a, col_b = st.columns([2, 3])

with col_a:
    st.markdown(
        f"""
<div style="background:{DARK};border-radius:8px;padding:18px 22px;margin-bottom:12px">
  <div style="color:{LIGHT};font-size:11px;font-weight:600;letter-spacing:1.5px;margin-bottom:6px">
    MONTHS SINCE LAST DELINQUENCY
  </div>
  <div style="color:white;font-size:13px;margin-bottom:14px">
    <b style="color:{LIGHT}">{missing_pct:.1f}%</b> of values are missing —
    meaning the applicant has <b>never been delinquent</b>.
  </div>
  <div style="display:flex;gap:16px">
    <div style="background:#1F4E4F;border-radius:6px;padding:10px 14px;flex:1">
      <div style="color:{GREEN};font-size:22px;font-weight:700">{never_delinq_dr:.1f}%</div>
      <div style="color:#aaa;font-size:10px">Never delinquent<br>(missing value)</div>
    </div>
    <div style="background:#1F4E4F;border-radius:6px;padding:10px 14px;flex:1">
      <div style="color:{RED};font-size:22px;font-weight:700">{ever_delinq_dr:.1f}%</div>
      <div style="color:#aaa;font-size:10px">Previously delinquent<br>(non-missing)</div>
    </div>
  </div>
  <div style="color:{LIGHT};font-size:11px;margin-top:10px">
    ↑ {gap:.1f}pp gap — the single most impactful EDA finding
  </div>
</div>
""", unsafe_allow_html=True)

with col_b:
    fig_delinq = go.Figure(go.Bar(
        x=["Never delinquent\n(missing = never)", "Previously delinquent\n(non-missing)"],
        y=[never_delinq_dr, ever_delinq_dr],
        marker_color=[GREEN, RED],
        text=[f"{never_delinq_dr:.1f}%", f"{ever_delinq_dr:.1f}%"],
        textposition="outside",
        width=0.45,
    ))
    fig_delinq.add_hline(y=df["default_flag"].mean()*100, line_dash="dot",
        line_color=MUTED, annotation_text=f"Population DR {df['default_flag'].mean()*100:.1f}%",
        annotation_position="right")
    fig_delinq.update_layout(
        title="Default rate by delinquency history",
        yaxis_title="Default Rate (%)", yaxis=dict(range=[0, 30]),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE, height=320,
        showlegend=False,
    )
    st.plotly_chart(fig_delinq, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# MISSINGNESS BAR CHART — interactive sort + filter
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Missingness by column")
summary = data_quality_summary(df)
missing_only = summary[summary["missing_pct"] > 0].copy()

ctrl1, ctrl2 = st.columns([2, 2])
with ctrl1:
    sort_by = st.radio("Sort by", ["Missing % (desc)", "Missing % (asc)", "Alphabetical"],
        horizontal=True, key="miss_sort")
with ctrl2:
    min_pct = st.slider("Minimum missing %", 0.0, 100.0, 0.0, 0.5, key="miss_filter")

filtered = missing_only[missing_only["missing_pct"] >= min_pct].copy()
if sort_by == "Missing % (desc)":
    filtered = filtered.sort_values("missing_pct", ascending=True)   # ascending=True for horizontal bar (bottom = highest)
elif sort_by == "Missing % (asc)":
    filtered = filtered.sort_values("missing_pct", ascending=False)
else:
    filtered = filtered.sort_values("column", ascending=False)

if filtered.empty:
    st.info("No columns meet the filter criteria.")
else:
    bar_colors = [RED if p > 20 else TEAL if p > 5 else MUTED for p in filtered["missing_pct"]]
    fig_miss = go.Figure(go.Bar(
        x=filtered["missing_pct"],
        y=filtered["column"],
        orientation="h",
        marker_color=bar_colors,
        text=filtered["missing_pct"].map(lambda x: f"{x:.1f}%"),
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}% missing<extra></extra>",
    ))
    fig_miss.update_layout(
        xaxis_title="Missing (%)", xaxis=dict(range=[0, 105]),
        plot_bgcolor=WHITE, paper_bgcolor=WHITE,
        height=max(280, len(filtered) * 32),
        margin=dict(l=180),
    )
    st.plotly_chart(fig_miss, use_container_width=True)

st.divider()

# ══════════════════════════════════════════════════════════════════════════
# FULL QUALITY TABLE
# ══════════════════════════════════════════════════════════════════════════
st.subheader("Per-column quality")
st.dataframe(
    summary, use_container_width=True, hide_index=True,
    column_config={
        "missing_pct": st.column_config.ProgressColumn(
            "Missing %", min_value=0, max_value=100, format="%.1f%%")
    },
)

with st.expander("Preview rows"):
    st.dataframe(df.head(50), use_container_width=True)
