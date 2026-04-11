# Indonesian Stock EDA Dashboard

A multi-page Streamlit app for exploratory data analysis of Indonesian (IDX) stocks. Select any stock and time horizon from the sidebar, then navigate through seven analysis pages to understand price dynamics, return behaviour, and optimal trading strategies.

The format is cloned from the following Streamlit Dashboard example: https://demo-stockpeers.streamlit.app/?ref=streamlit-io-gallery-favorites

## Try it on your machine

1. Start a virtual environment and install dependencies (requires `uv`):

   ```sh
   uv venv
   .venv/Scripts/activate      # Windows
   # .venv/bin/activate        # macOS / Linux
   uv sync
   ```

2. Run the app from the `streamlit_eda/` directory:

   ```sh
   uv run streamlit run streamlit_app.py
   ```

---

## Sidebar Controls

| Control | Description |
|---|---|
| **Stock ticker** | Predefined IDX stocks: BBCA, BBRI, BMRI, TLKM, ASII, ANTM, PGAS, PGEO |
| **Custom ticker** | Override with any Yahoo Finance symbol (e.g. `AAPL`, `NVDA`) |
| **Time horizon** | 1 Month · 3 Months · 6 Months · 1 Year · 2 Years · 5 Years |

---

## Pages

### 1. Daily Stock
Home page showing the OHLCV candlestick & volume chart alongside four summary metrics:
- Current price, period return, annualised volatility, and maximum drawdown.

### 2. Return Distribution
Analyses the statistical properties of log returns:
- Histogram vs. fitted Normal PDF and Q-Q plot side-by-side
- Normality tests: Jarque-Bera, Kolmogorov-Smirnov, Shapiro-Wilk
- Interactive rolling skewness & kurtosis chart (adjustable window)

### 3. Structural Breaks
Detects regime shifts in the return-generating process using the **CUSUM test**:
- CUSUM statistic vs. 5% critical boundaries chart
- Log returns with detected break periods highlighted
- Summary metrics and interpretation (stable vs. structural break)

### 4. Seasonality
Examines systematic calendar patterns and earnings-driven behaviour:
- **Day-of-week effect** — boxplot and mean return table (Mon–Fri)
- **Monthly effect** — boxplot and mean return table (Jan–Dec)
- **Pre/Post Earnings** — average log return in the ±10 trading-day window around earnings releases, with PEAD detection

### 5. Volatility Structure
Models time-varying volatility using industry-standard approaches:
- **ARCH-LM Test** — detects conditional heteroskedasticity (adjustable lags)
- **GARCH(1,1)** — conditional volatility chart (annualised), model parameters (ω, α, β), persistence analysis, and volatility shock half-life

### 6. External Correlation
Measures how the stock co-moves with external market factors:
- **Preset comparison tickers**: IHSG, USD/IDR, S&P 500, VIX, Gold, Crude Oil (WTI), and major IDX stocks
- **Custom tickers**: add any Yahoo Finance symbol
- Static correlation heatmap and rolling correlation chart (adjustable window)
- **IHSG Decoupling Analysis** — compares correlation during market crash periods vs. normal periods

### 7. Strategy Recommendation
A comprehensive statistical framework that maps return dynamics to a trading strategy. Each test contributes a vote to a final scorecard:

| Test | Trend Following | Mean Reversion | Random Walk |
|---|---|---|---|
| ACF / Ljung-Box | Positive autocorrelation | Negative autocorrelation | ~Zero |
| Hurst Exponent (R/S) | H > 0.5 | H < 0.5 | H ≈ 0.5 |
| ADF Stationarity | Non-stationary prices | Stationary prices | Unit root |
| Variance Ratio (Lo-MacKinlay) | VR > 1 | VR < 1 | VR ≈ 1 |
| Half-Life (OU process) | Long / ∞ | Short (days–weeks) | ∞ |
| Rolling Regime (Hurst + VR) | >60% trending | >60% reverting | Mixed |

The page concludes with a **Strategy Scorecard** and a recommendation (Trend Following, Mean Reversion, or Regime-Switching) with concrete strategy suggestions.

---

## Tech Stack

| Library | Purpose |
|---|---|
| `streamlit` | App framework and UI |
| `yfinance` | OHLCV and earnings data from Yahoo Finance |
| `altair` | Interactive charts |
| `pandas` / `numpy` | Data wrangling and numerical computation |
| `statsmodels` | ACF, Ljung-Box, ADF, CUSUM, OLS |
| `arch` | ARCH-LM test and GARCH(1,1) fitting |
| `scipy` | Normality tests, variance ratio z-stats |
