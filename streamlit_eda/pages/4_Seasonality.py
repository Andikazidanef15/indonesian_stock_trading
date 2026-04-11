import streamlit as st
import pandas as pd
import numpy as np

from src.sidebar import render_sidebar
from src.utils import load_ohlcv, get_earnings_dates
from src.stats import (
    calculate_daily_price_metrics,
    calculate_aggregated_log_returns,
    calculate_pre_post_earnings_behavior,
)
from src.visualization import (
    visualize_agg_returns_boxplot,
    visualize_pre_post_earnings_behavior,
)


ticker, period = render_sidebar()

st.title(":material/calendar_month: Seasonality Analysis")
st.caption(f"Analysing **{ticker}** over **{period}**")

st.write(
    """
    Examines whether returns exhibit systematic patterns by **day of week**, **calendar month**, 
    or around **earnings announcements**.
    """
)

# ── Data ───────────────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

log_returns = calculate_daily_price_metrics(df)["log_returns"]

# ── Day-of-week & Monthly seasonality ─────────────────────────────────────────
st.subheader("Day-of-Week & Monthly Effects")

day_labels   = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

dow_returns,   dow_grouped   = calculate_aggregated_log_returns(log_returns, "dayofweek")
month_returns, month_grouped = calculate_aggregated_log_returns(log_returns, "month")

col1, col2 = st.columns(2)

with col1:
    st.altair_chart(
        visualize_agg_returns_boxplot(
            dow_returns,
            key_column="DayOfWeek",
            label_column="DayLabel",
            labels=day_labels,
        ),
        use_container_width=True,
    )
    dow_mean_df = pd.DataFrame({
        "Day":             [day_labels[d] for d in range(5)],
        "Mean Log Return": [dow_grouped.mean().get(d, np.nan) for d in range(5)],
    })
    st.caption("Mean log return by day of week")
    st.table(dow_mean_df)

with col2:
    st.altair_chart(
        visualize_agg_returns_boxplot(
            month_returns,
            key_column="Month",
            label_column="MonthLabel",
            labels=month_labels,
            key_offset=1,
        ),
        use_container_width=True,
    )
    month_mean_df = pd.DataFrame({
        "Month":           [month_labels[m - 1] for m in range(1, 13)],
        "Mean Log Return": [month_grouped.mean().get(m, np.nan) for m in range(1, 13)],
    })
    st.caption("Mean log return by calendar month")
    st.table(month_mean_df)

st.divider()

# ── Pre/Post Earnings Behavior ─────────────────────────────────────────────────
st.subheader("Pre/Post Earnings Behavior")
st.write(
    """
    Measures average log returns in the **±10 trading days** window around earnings releases 
    to identify pre-earnings drift, earnings-day reactions, and post-earnings announcement drift (PEAD).
    """
)

with st.spinner("Fetching earnings dates…"):
    earning_dates = get_earnings_dates(df, ticker)

N_DAYS = 10
returns_around_earnings = calculate_pre_post_earnings_behavior(df, log_returns, earning_dates, N_DAYS)

if not returns_around_earnings:
    st.warning("No earnings date overlaps found in the dataset for this period.")
    st.stop()

returns_matrix = np.array(returns_around_earnings)
mean_returns   = np.nanmean(returns_matrix, axis=0)
days           = np.arange(-N_DAYS, N_DAYS + 1)

earnings_df = pd.DataFrame({
    "Days from Earnings": days,
    "Mean Log Return":    mean_returns,
})

st.altair_chart(visualize_pre_post_earnings_behavior(earnings_df), use_container_width=True)

mean_returns_df = pd.DataFrame({
    "Day":             [f"{d:+}" for d in days],
    "Mean Log Return": mean_returns,
})
with st.expander("Show daily mean log return table"):
    st.table(mean_returns_df)

# ── Earnings interpretation ────────────────────────────────────────────────────
st.subheader("Interpretation")

pre_mask  = (days >= -5) & (days < 0)
post_mask = (days > 0)  & (days <= 5)
pre_avg   = np.nanmean(mean_returns[pre_mask])
post_avg  = np.nanmean(mean_returns[post_mask])
day0_ret  = mean_returns[days == 0][0]
pre_cum   = np.nansum(mean_returns[pre_mask])
post_cum  = np.nansum(mean_returns[post_mask])

ic1, ic2, ic3 = st.columns(3)

with ic1:
    st.metric("Pre-earnings avg (d -5 to -1)", f"{pre_avg*100:.3f}%")
    if pre_avg > 0.001:
        st.success("Pre-earnings drift detected — stock tends to rise before announcements.")
    elif pre_avg < -0.001:
        st.warning("Unusual pre-earnings selling pressure.")
    else:
        st.info("No significant pre-earnings drift.")

with ic2:
    st.metric("Earnings day return (d 0)", f"{day0_ret*100:.3f}%")
    if abs(day0_ret) > 0.01:
        st.warning("High volatility on earnings day (|return| > 1%).")
    else:
        st.info("Moderate reaction on earnings day.")

with ic3:
    st.metric("Post-earnings avg (d +1 to +5)", f"{post_avg*100:.3f}%")
    if post_avg > 0.001:
        st.success("Post-earnings announcement drift (PEAD) detected.")
    elif post_avg < -0.001:
        st.warning("Post-earnings reversal or negative drift.")
    else:
        st.info("No significant post-earnings drift.")

# Trading implications
implications = []
if pre_avg > 0.001:
    implications.append("Consider entering long positions ~5 days before expected earnings.")
if abs(day0_ret) > 0.01:
    implications.append("Earnings day shows high volatility — options strategies may be valuable.")
if post_avg > 0.001:
    implications.append("Momentum continuation after positive earnings may be tradeable.")
elif post_avg < -0.001:
    implications.append("Consider taking profits before or on earnings day.")

if implications:
    st.markdown("**Trading Implications:**")
    for imp in implications:
        st.info(f"• {imp}")

st.warning(
    "This analysis uses estimated quarterly earnings dates. "
    "Actual dates should be verified before making trading decisions."
)
