import streamlit as st
import pandas as pd
import numpy as np

from src.sidebar import render_sidebar
from src.utils import load_ohlcv
from src.stats import (
    calculate_daily_price_metrics,
    calculate_return_distribution_metrics,
    calculate_normality_test,
    calculate_rolling_skewness_kurtosis,
)
from src.visualization import (
    visualize_log_return_distribution,
    visualize_qq_plot,
    visualize_rolling_skewness_kurtosis,
)


ticker, period = render_sidebar()

st.title(":material/bar_chart: Return Distribution Analysis")
st.caption(f"Analysing **{ticker}** over **{period}**")

# ── Data ───────────────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

log_returns = calculate_daily_price_metrics(df)["log_returns"]

# ── Summary statistics ─────────────────────────────────────────────────────────
st.subheader("Summary Statistics")

m = calculate_return_distribution_metrics(log_returns)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mean Daily Return", f"{m['mu']:.5f}")
c2.metric("Std Dev",           f"{m['std']:.5f}")
c3.metric("Skewness",          f"{m['sk']:.3f}")
c4.metric("Excess Kurtosis",   f"{m['ku']:.3f}")

# ── Distribution charts ────────────────────────────────────────────────────────
st.subheader("Distribution Charts")
col_hist, col_qq = st.columns(2)

with col_hist:
    st.altair_chart(
        visualize_log_return_distribution(log_returns, m["mu"], m["std"]),
        use_container_width=True,
    )

with col_qq:
    st.altair_chart(visualize_qq_plot(log_returns), use_container_width=True)

# ── Normality tests ────────────────────────────────────────────────────────────
st.subheader("Normality Tests")
st.caption("H₀: log returns are normally distributed")

test_df = calculate_normality_test(log_returns, m["mu"], m["std"])
st.dataframe(test_df, hide_index=True, use_container_width=True)

any_reject = any(float(p) < 0.05 for p in test_df["p-value"])
if any_reject:
    st.warning(
        "**Fat tails detected.** Returns deviate significantly from normality. "
        "Standard deviation alone understates tail risk — consider strategies robust to large moves."
    )
else:
    st.success("Returns appear consistent with normality over this period.")

st.divider()

# ── Rolling skewness & kurtosis ────────────────────────────────────────────────
roll_window = st.slider(
    "Rolling window (days)",
    min_value=20, max_value=120, value=60, step=10,
    help="Number of trading days used for the rolling skewness and kurtosis calculation.",
)

st.subheader(f"Rolling {roll_window}-Day Skewness & Kurtosis")

roll_df = calculate_rolling_skewness_kurtosis(log_returns, roll_window)
st.altair_chart(
    visualize_rolling_skewness_kurtosis(roll_df, roll_window),
    use_container_width=True,
)

mean_sk = roll_df["Skewness"].mean()
mean_ku = roll_df["Kurtosis"].mean()

ic1, ic2 = st.columns(2)
if mean_sk > 0.1:
    ic1.info(f"**Skewness ({mean_sk:.3f}):** Right-skewed on average — more frequent small losses, occasional large gains.")
elif mean_sk < -0.1:
    ic1.warning(f"**Skewness ({mean_sk:.3f}):** Left-skewed on average — risk of large losses (unfavourable tail risk).")
else:
    ic1.info(f"**Skewness ({mean_sk:.3f}):** Approximately symmetric — no strong directional bias in tail behaviour.")

if mean_ku > 1:
    ic2.warning(
        f"**Kurtosis ({mean_ku:.3f}):** Fat tails (leptokurtic) — extreme moves occur more often than "
        "a normal distribution predicts. Standard risk models may underestimate tail risk."
    )
else:
    ic2.info(f"**Kurtosis ({mean_ku:.3f}):** Near-normal tails — standard deviation is a reasonable risk proxy.")
