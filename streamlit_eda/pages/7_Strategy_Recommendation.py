import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from src.sidebar import render_sidebar
from src.utils import load_ohlcv
from src.stats import (
    calculate_daily_price_metrics,
    calculate_acf_ljungbox,
    calculate_hurst_exponent,
    calculate_adf_stationarity,
    calculate_variance_ratio,
    calculate_half_life_mr,
    calculate_rolling_regime,
    calculate_strategy_scorecard,
)
from src.visualization import (
    visualize_acf_chart,
    visualize_hurst_rs_chart,
    visualize_variance_ratio_chart,
    visualize_half_life_spread_chart,
    visualize_rolling_regime_chart,
)

ticker, period = render_sidebar()
st.title(":material/recommend: Strategy Recommendation")
st.caption(f"Analysing **{ticker}** over **{period}**")
st.markdown(
    """
    A comprehensive statistical analysis to determine which trading strategy best fits the
    stock's return dynamics. Each test maps a property to a strategy signal:

    | Property | Test | Trend Following | Mean Reversion | Random Walk |
    |---|---|---|---|---|
    | Autocorrelation | ACF / Ljung-Box | Positive at short lags | Negative at short lags | ~Zero |
    | Hurst Exponent | R/S Analysis | H > 0.5 | H < 0.5 | H ≈ 0.5 |
    | Stationarity | ADF Test | Non-stationary prices | Stationary prices | Unit root |
    | Variance Ratio | Lo-MacKinlay | VR > 1 (momentum) | VR < 1 (reversal) | VR ≈ 1 |
    | Half-life | OLS on lagged spread | Long / ∞ | Short (days–weeks) | ∞ |
    """
)
st.divider()

# ── Data ───────────────────────────────────────────────────────────────────────
try:
    df = load_ohlcv(ticker, period)
except Exception as e:
    st.error(str(e))
    st.stop()

metrics = calculate_daily_price_metrics(df)
log_returns = metrics["log_returns"]
close = metrics["close"]

# ── 1. Autocorrelation Analysis ────────────────────────────────────────────────
st.subheader("1. Autocorrelation Analysis")
st.markdown(
    """
    Autocorrelation measures how today's return correlates with past returns:
    - **Positive autocorrelation** → returns persist → **trend following** works
    - **Negative autocorrelation** → returns reverse → **mean reversion** works
    - **No autocorrelation** → random walk → neither works reliably

    The red dashed lines mark the 95% confidence interval for the null hypothesis (ACF = 0).
    """
)

acf_values, ci_bound, lb_df = calculate_acf_ljungbox(log_returns, n_lags=40)
st.altair_chart(visualize_acf_chart(acf_values, ci_bound, n_lags=40), use_container_width=True)

st.markdown("**Ljung-Box Test** (H₀: no autocorrelation up to lag k)")
st.dataframe(lb_df, hide_index=True, use_container_width=True)

avg_acf_1_5 = float(np.mean(acf_values[1:6]))
lb_significant = bool((lb_df["p-value"] < 0.05).any())
if lb_significant:
    if avg_acf_1_5 > 0:
        st.info(
            f"Significant autocorrelation detected. "
            f"Avg ACF(1-5) = **{avg_acf_1_5:+.4f}** → **Positive** (Trend Following signal)"
        )
    else:
        st.info(
            f"Significant autocorrelation detected. "
            f"Avg ACF(1-5) = **{avg_acf_1_5:+.4f}** → **Negative** (Mean Reversion signal)"
        )
else:
    st.success("No significant autocorrelation → Returns behave like a random walk.")

st.divider()

# ── 2. Hurst Exponent ─────────────────────────────────────────────────────────
st.subheader("2. Hurst Exponent (R/S Analysis)")
st.markdown(
    r"""
    The Hurst exponent $H$ classifies the memory of a time series:
    - **H > 0.5** → Persistent (trending) → **Trend Following**
    - **H = 0.5** → Random walk → No exploitable pattern
    - **H < 0.5** → Anti-persistent (mean-reverting) → **Mean Reversion**

    Computed via Rescaled Range (R/S) analysis across multiple window sizes.
    """
)

with st.spinner("Computing Hurst exponent..."):
    H, log_w, log_rs = calculate_hurst_exponent(log_returns)

st.altair_chart(visualize_hurst_rs_chart(log_w, log_rs, H), use_container_width=True)

col1, col2 = st.columns(2)
col1.metric("Hurst Exponent (H)", f"{H:.4f}")
col2.metric(
    "Classification",
    "Trending (H > 0.5)" if H > 0.55 else "Mean Reverting (H < 0.5)" if H < 0.45 else "Random Walk (H ≈ 0.5)",
)

if H > 0.55:
    st.info(f"H = {H:.4f} > 0.5 → **Persistent (Trending)** — Trend Following strategies are suitable.")
elif H < 0.45:
    st.info(f"H = {H:.4f} < 0.5 → **Anti-Persistent (Mean Reverting)** — Mean Reversion strategies are suitable.")
else:
    st.warning(f"H = {H:.4f} ≈ 0.5 → **Random Walk** — Neither strategy has a strong statistical edge.")

st.divider()

# ── 3. Stationarity (ADF Test) ────────────────────────────────────────────────
st.subheader("3. Stationarity — Augmented Dickey-Fuller Test")
st.markdown(
    """
    The ADF test checks for a **unit root** (non-stationary series):
    - **Prices non-stationary + Returns stationary** → Normal behavior, need differencing
    - **Prices stationary** → Mean-reverting price → supports **Mean Reversion** directly on price level
    """
)

adf_df, adf_price_stationary, adf_price_pvalue = calculate_adf_stationarity(close, log_returns)
st.dataframe(
    adf_df[["ADF Statistic", "p-value", "5% Critical", "Stationary"]].reset_index(),
    hide_index=True,
    use_container_width=True,
)

if not adf_price_stationary:
    st.info(
        "Prices are **non-stationary**, returns are stationary — typical unit-root (random walk with drift) behavior. "
        "Mean reversion on raw price is not supported; use returns or price–SMA spreads instead."
    )
else:
    st.success(
        "Prices are **stationary** — supports Mean Reversion strategies directly on the price level."
    )

st.divider()

# ── 4. Variance Ratio Test ────────────────────────────────────────────────────
st.subheader("4. Variance Ratio Test (Lo-MacKinlay)")
st.markdown(
    r"""
    $$VR(k) = \frac{\text{Var}(r_t^{(k)})}{k \cdot \text{Var}(r_t)}$$

    - **VR > 1** → Positive serial correlation → **Trend Following** (returns persist)
    - **VR = 1** → Random walk (variance scales linearly with horizon)
    - **VR < 1** → Negative serial correlation → **Mean Reversion** (returns reverse)
    """
)

vr_df = calculate_variance_ratio(log_returns)
st.altair_chart(visualize_variance_ratio_chart(vr_df), use_container_width=True)
st.dataframe(vr_df, hide_index=True, use_container_width=True)

avg_vr = float(vr_df["VR(k)"].mean())
if avg_vr > 1.05:
    st.info(f"Average VR = **{avg_vr:.4f}** > 1 → **Trend Following** signal (positive serial correlation).")
elif avg_vr < 0.95:
    st.info(f"Average VR = **{avg_vr:.4f}** < 1 → **Mean Reversion** signal (negative serial correlation).")
else:
    st.success(f"Average VR = **{avg_vr:.4f}** ≈ 1 → Consistent with a **random walk**.")

st.divider()

# ── 5. Half-Life of Mean Reversion ────────────────────────────────────────────
st.subheader("5. Half-Life of Mean Reversion")
st.markdown(
    r"""
    If the stock mean-reverts, how fast? We fit an Ornstein-Uhlenbeck (OU) process to the
    price–SMA spread:

    $$\Delta y_t = \theta \cdot y_{t-1} + \text{intercept} + \varepsilon_t$$

    Half-life $= -\ln(2) / \ln(1 + \theta)$ where $\theta < 0$ for mean reversion.
    A short half-life (< 20 days) makes mean-reversion strategies practical.
    """
)

hl_rows = []
for w in [20, 50, 100]:
    spread = (close - close.rolling(w).mean()).dropna()
    hl, theta = calculate_half_life_mr(spread)
    hl_rows.append({
        "SMA Window": w,
        "θ (theta)": round(theta, 6),
        "Half-Life (days)": f"{hl:.1f}" if not np.isinf(hl) else "∞",
    })

st.dataframe(pd.DataFrame(hl_rows), hide_index=True, use_container_width=True)

hl_50, _ = calculate_half_life_mr((close - close.rolling(50).mean()).dropna())
st.altair_chart(
    visualize_half_life_spread_chart(close, sma_window=50, half_life=hl_50),
    use_container_width=True,
)

if not np.isinf(hl_50) and hl_50 < 20:
    st.info(
        f"Half-life (SMA50) = **{hl_50:.1f} days** → **Fast mean reversion** — "
        "Bollinger Band / RSI mean-reversion strategies are viable."
    )
elif not np.isinf(hl_50) and hl_50 < 60:
    st.warning(
        f"Half-life (SMA50) = **{hl_50:.1f} days** → **Moderate mean reversion** — "
        "mean reversion is possible but requires longer holding periods."
    )
else:
    hl_str = f"{hl_50:.1f}" if not np.isinf(hl_50) else "∞"
    st.warning(
        f"Half-life (SMA50) = **{hl_str} days** → **Slow / no mean reversion** — "
        "mean reversion on the price–SMA spread is impractical."
    )

st.divider()

# ── 6. Rolling Regime Analysis ────────────────────────────────────────────────
st.subheader("6. Rolling Regime Analysis")
st.markdown(
    """
    Markets shift between regimes — a stock can trend for months, then mean-revert.
    Rolling Hurst exponents and Variance Ratios reveal **when** the stock trends vs. reverts,
    using a sliding window (up to 252 trading days, adapted to available data).
    """
)


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _cached_rolling_regime(t: str, p: str):
    _df = load_ohlcv(t, p)
    _lr = calculate_daily_price_metrics(_df)["log_returns"]
    return calculate_rolling_regime(_lr, window=252)


with st.spinner("Computing rolling regime (this may take a moment)..."):
    roll_hurst, roll_vr, pct_trending, current_hurst, eff_window = _cached_rolling_regime(ticker, period)

if roll_hurst.dropna().empty:
    st.warning(
        f"Insufficient data for rolling regime analysis — the selected period is too short. "
        f"Select **2 Years** or longer to see meaningful rolling results."
    )
else:
    st.altair_chart(
        visualize_rolling_regime_chart(roll_hurst, roll_vr, window=eff_window),
        use_container_width=True,
    )

col1, col2, col3 = st.columns(3)
col1.metric("Trending periods (H > 0.5)", f"{pct_trending:.1f}%")
col2.metric("Mean-reverting periods (H < 0.5)", f"{100 - pct_trending:.1f}%")
col3.metric("Current Hurst", f"{current_hurst:.4f}" if not np.isnan(current_hurst) else "N/A")

if pct_trending > 60:
    st.info(
        f"Stock spends **{pct_trending:.1f}%** of its time in trending regimes "
        "→ **Trend Following** is the primary strategy."
    )
elif pct_trending < 40:
    st.info(
        f"Stock spends **{100 - pct_trending:.1f}%** of its time in mean-reverting regimes "
        "→ **Mean Reversion** is the primary strategy."
    )
else:
    st.warning(
        "Stock alternates fairly equally between regimes — consider a **regime-switching** approach."
    )

st.divider()

# ── 7. Strategy Selection Scorecard ───────────────────────────────────────────
st.subheader("7. Strategy Selection Summary")
st.markdown(
    "All evidence combined into a single scorecard to determine the optimal trading strategy."
)

scorecard, trend_votes, revert_votes, neutral_votes = calculate_strategy_scorecard(
    acf_values=acf_values,
    lb_df=lb_df,
    H=H,
    adf_price_stationary=adf_price_stationary,
    adf_price_pvalue=adf_price_pvalue,
    vr_df=vr_df,
    hl_50=hl_50,
    roll_hurst=roll_hurst,
)


def _color_verdict(val: str) -> str:
    if "Trend" in val:
        return "color: #FF5722; font-weight: bold"
    if "Reversion" in val or "Revert" in val:
        return "color: #2196F3; font-weight: bold"
    return ""


st.dataframe(
    scorecard.style.map(_color_verdict, subset=["Verdict"]),
    hide_index=True,
    use_container_width=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Trend Following votes", trend_votes)
c2.metric("Mean Reversion votes", revert_votes)
c3.metric("Neutral / Other", neutral_votes)

st.subheader("Recommendation")

if trend_votes > revert_votes and trend_votes >= 3:
    st.success(
        f"""**TREND FOLLOWING** is recommended for **{ticker}**.

Suggested strategies:
1. EMA Crossover (short/long EMA)
2. Breakout (Donchian Channel / ATR-based)
3. MACD Signal Line Crossover
4. Momentum (ROC / RSI trend)"""
    )
elif revert_votes > trend_votes and revert_votes >= 3:
    st.success(
        f"""**MEAN REVERSION** is recommended for **{ticker}**.

Suggested strategies:
1. Bollinger Bands (sell at upper band, buy at lower band)
2. RSI Overbought/Oversold (RSI > 70 / < 30)
3. Z-score on price–SMA spread
4. Pairs trading with a correlated asset"""
    )
else:
    st.warning(
        f"""The evidence is **mixed** for **{ticker}** — the stock alternates between regimes.

Suggested approaches:
1. **Regime-Switching:** Use rolling Hurst to toggle between trend-following (H > 0.5) and mean-reversion (H < 0.5)
2. **Volatility-Based:** Trend-follow in high-vol, mean-revert in low-vol
3. **ML Classification:** Train a model to predict the current regime, then apply the appropriate strategy
4. **Ensemble:** Combine both strategy types with adaptive weighting"""
    )
