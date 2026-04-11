import streamlit as st
import numpy as np

from src.sidebar import render_sidebar
from src.utils import load_ohlcv
from src.visualization import visualize_price_chart
from src.stats import calculate_daily_price_metrics

ticker, period = render_sidebar()

st.title(":material/home: Daily Stock")
st.caption("Select a stock and time horizon in the sidebar, then explore each analysis page.")

st.divider()

try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

st.subheader(f"{ticker} — Daily Price & Volume")
st.altair_chart(visualize_price_chart(df), use_container_width=True)

metrics = calculate_daily_price_metrics(df)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",         f"{metrics['current_price']:,.2f}")
c2.metric("Period Return",         f"{metrics['period_return']:.1f}%")
c3.metric("Annualised Volatility", f"{metrics['annual_vol']:.1f}%")
c4.metric("Max Drawdown",          f"{metrics['max_dd']:.1f}%")
