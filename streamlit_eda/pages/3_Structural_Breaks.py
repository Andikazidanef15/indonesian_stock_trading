import streamlit as st
import numpy as np

from src.sidebar import render_sidebar
from src.utils import load_ohlcv
from src.stats import calculate_daily_price_metrics, calculate_cusum_test
from src.visualization import visualize_cusum_chart, visualize_returns_with_breaks


ticker, period = render_sidebar()

st.title(":material/timeline: Structural Break Analysis")
st.caption(f"Analysing **{ticker}** over **{period}**")

st.write(
    """
    The **CUSUM (Cumulative Sum) Test** detects structural breaks in time series data by analysing 
    cumulative recursive residuals. Significant deviations from expected stability suggest a regime 
    shift in the return-generating process.
    """
)

# ── Data ───────────────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

close = calculate_daily_price_metrics(df)["close"]

# ── CUSUM calculation ──────────────────────────────────────────────────────────
with st.spinner("Running CUSUM test…"):
    breaks_detected, cusum_df = calculate_cusum_test(close)

# ── Charts ─────────────────────────────────────────────────────────────────────
st.subheader("CUSUM Statistic vs Critical Boundaries")
st.altair_chart(visualize_cusum_chart(cusum_df), use_container_width=True)

st.subheader("Log Returns with Detected Break Periods")
st.altair_chart(visualize_returns_with_breaks(breaks_detected, cusum_df), use_container_width=True)

# ── Test result metrics ────────────────────────────────────────────────────────
st.subheader("Test Results")

max_cusum    = np.max(np.abs(cusum_df["CUSUM"]))
max_boundary = np.max(cusum_df["Upper Bound"])
break_flag   = max_cusum > max_boundary

col1, col2, col3 = st.columns(3)
col1.metric("Max |CUSUM| Statistic", f"{max_cusum:.4f}")
col2.metric("5% Critical Boundary",  f"{max_boundary:.4f}")
col3.metric("Result", "Break Detected" if break_flag else "No Break")

if len(breaks_detected) > 0:
    first = cusum_df["Log Returns"].index[breaks_detected[0]].strftime("%Y-%m-%d")
    last  = cusum_df["Log Returns"].index[breaks_detected[-1]].strftime("%Y-%m-%d")
    st.info(
        f"**{len(breaks_detected)} period(s)** exceeded critical bounds.  \n"
        f"First break: **{first}** — Last break: **{last}**"
    )

# ── Interpretation ─────────────────────────────────────────────────────────────
st.subheader("Interpretation")

if break_flag:
    st.warning(
        """
        **Structural break detected** — The return-generating process shows evidence of structural change.

        - Model parameters estimated on historical data may not be stable
        - Consider using adaptive/rolling estimation windows
        - Regime-switching models may be appropriate
        """
    )
else:
    st.success(
        """
        **No structural break detected** — The return-generating process appears stable over the sample period.

        - Historical parameters may be used for forecasting with more confidence
        - Single-regime models are likely appropriate
        """
    )
