import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.sidebar import render_sidebar
from src.utils import load_ohlcv
from src.stats import calculate_daily_price_metrics, calculate_arch_lm_test, calculate_garch_volatility_test

ticker, period = render_sidebar()
st.title(":material/waves: Volatility Structure Analysis")
st.caption(f"Analysing **{ticker}** over **{period}**")
st.markdown(
    """
    The **Volatility Structure Analysis** is used to analyse the volatility structure of a stock. 
    Volatility is defined as the standard deviation of the returns of a stock.
    """
)
st.divider()
st.subheader("ARCH-LM Test")
st.markdown(
    """
    The **ARCH-LM Test** is used to detect conditional heteroskedasticity in the returns of a stock.
    If the p-value is less than 0.05, then there is a significant ARCH effect detected (conditional volatility is present),
    GARCH modeling is appropriate.
    """
)

# ── Data ───────────────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

log_returns = calculate_daily_price_metrics(df)["log_returns"]

# ── ARCH-LM Test ──────────────────────────────────────────────────────────────
arch_test_lags = st.slider(
    "ARCH-LM Test lags",
    min_value=1, max_value=24, value=12, step=1,
    help="Number of lags to use for the ARCH-LM test.",
)
with st.spinner("Running ARCH-LM test..."):
    arch_lm_result = calculate_arch_lm_test(log_returns, lags=arch_test_lags)

st.subheader("ARCH-LM Test Results (lags={})".format(arch_test_lags))
st.dataframe(arch_lm_result, hide_index=True, use_container_width=True)

if float(arch_lm_result["p-value"][0]) < 0.05:
    st.warning("Significant ARCH effect detected (conditional volatility is present). GARCH modeling is appropriate.")
else:
    st.success("No significant ARCH effect detected (conditional volatility not strongly present).")


# --- Visualize Volatility Structure (GARCH) ---
st.divider()
st.subheader("GARCH(1,1) Conditional Volatility")
st.markdown(
    """
    The **GARCH(1,1)** model is a popular model for volatility clustering.
    It is a type of GARCH model that is used to model the volatility of a time series.
    """
)

# Fit GARCH(1,1) model to log returns
with st.spinner("Fitting GARCH(1,1) model..."):
    vol_df, garch_res = calculate_garch_volatility_test(log_returns)

vol_chart = alt.Chart(vol_df).mark_line(color='steelblue').encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('Volatility:Q', title='Annualized Volatility'),
    tooltip=[
        alt.Tooltip('Date:T', title='Date'),
        alt.Tooltip('Volatility:Q', title='Volatility', format='.4f')
    ]
).properties(
    title=f'Estimated Conditional Volatility (GARCH(1,1)) for {ticker} (Annualized)',
    height=400
)

st.altair_chart(vol_chart, use_container_width=True)

# --- GARCH Model Parameters ---
st.subheader("GARCH(1,1) Model Parameters")

# Extract key parameters
omega = garch_res.params['omega']
alpha = garch_res.params['alpha[1]']
beta = garch_res.params['beta[1]']
persistence = alpha + beta

col1, col2, col3, col4 = st.columns(4)
col1.metric("ω (omega)", f"{omega:.6f}", help="Baseline variance")
col2.metric("α (alpha)", f"{alpha:.4f}", help="Shock impact coefficient")
col3.metric("β (beta)", f"{beta:.4f}", help="Volatility persistence coefficient")
col4.metric("α + β", f"{persistence:.4f}", help="Volatility persistence")

# --- GARCH Model Interpretation Summary ---
st.subheader("GARCH(1,1) Model Interpretation")

# Persistence interpretation
if persistence >= 0.99:
    st.warning("**Extremely high persistence:** Volatility shocks decay very slowly (near unit root).")
elif persistence >= 0.95:
    st.warning("**High persistence:** Volatility clusters strongly; shocks take a long time to dissipate.")
elif persistence >= 0.85:
    st.info("**Moderate persistence:** Volatility mean-reverts at a reasonable pace.")
else:
    st.success("**Low persistence:** Volatility shocks dissipate quickly.")

# Alpha interpretation
if alpha > 0.1:
    st.info("**High α:** Market is reactive to recent shocks (news impact is significant).")
else:
    st.info("**Moderate/low α:** Market absorbs shocks without dramatic volatility spikes.")

# Beta interpretation
if beta > 0.8:
    st.info("**High β:** Past volatility strongly predicts future volatility.")

# Half-life of volatility shocks
if persistence < 1:
    half_life = np.log(0.5) / np.log(persistence)
    st.metric("Volatility Shock Half-life", f"~{half_life:.1f} trading days")

st.subheader("Practical Implications")
st.markdown(f"""
- **GARCH effects** confirm volatility clustering in **{ticker}** returns.
- Risk models should account for **time-varying volatility**.
- VaR/CVaR estimates should use **conditional** (not unconditional) volatility.
""")

with st.expander("View Full Model Summary"):
    st.html(garch_res.summary().as_html())

