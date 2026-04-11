import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

from .utils import to_utc_naive

def calculate_daily_price_metrics(df: pd.DataFrame):
    close = df["Close"]
    log_returns = np.log(close).diff().dropna()
    return {
        "close": close,
        "log_returns": log_returns,
        "current_price": close.iloc[-1],
        "period_return": (close.iloc[-1] / close.iloc[0] - 1) * 100,
        "annual_vol": log_returns.std() * np.sqrt(252) * 100,
        "max_dd": ((close / close.cummax()) - 1).min() * 100,
    }

def calculate_aggregated_log_returns(log_returns:pd.Series, period:str='dayofweek'):
    dow_returns = log_returns.copy()
    dow_returns = pd.DataFrame(dow_returns)
    dow_returns.columns = ['Log Return']
    if period == 'dayofweek':
        dow_returns['DayOfWeek'] = dow_returns.index.dayofweek
        dow_grouped = dow_returns.groupby('DayOfWeek')['Log Return']
    elif period == 'month':
        dow_returns['Month'] = dow_returns.index.month
        dow_grouped = dow_returns.groupby('Month')['Log Return']
    else:
        raise ValueError(f"Invalid period: {period}")
    return dow_returns, dow_grouped

def calculate_return_distribution_metrics(log_returns: pd.Series):
    return {
        "mu": log_returns.mean(),
        "std": log_returns.std(),
        "sk": log_returns.skew(),
        "ku": log_returns.kurt()
    }

def calculate_normality_test(log_returns: pd.Series, mu:float, std:float, ):
    jb_stat,      jb_p      = stats.jarque_bera(log_returns)
    ks_stat,      ks_p      = stats.kstest(log_returns, "norm", args=(mu, std))
    shapiro_stat, shapiro_p = stats.shapiro(log_returns.values[:5000])   # Shapiro capped at 5 000

    def verdict(p: float, alpha: float = 0.05) -> str:
        return "✗ Non-normal" if p < alpha else "✓ Normal"
    
    test_df = pd.DataFrame({
        "Test":            ["Jarque-Bera", "Kolmogorov-Smirnov", "Shapiro-Wilk"],
        "Statistic":       [f"{jb_stat:.4f}",     f"{ks_stat:.4f}",     f"{shapiro_stat:.4f}"],
        "p-value":         [f"{jb_p:.4e}",        f"{ks_p:.4e}",        f"{shapiro_p:.4e}"],
        "Verdict (α=0.05)":[verdict(jb_p),        verdict(ks_p),        verdict(shapiro_p)],
    })
    
    return test_df

def calculate_rolling_skewness_kurtosis(log_returns: pd.Series, roll_window: int):
    roll_df = pd.DataFrame({
        "Date":     log_returns.index,
        "Skewness": log_returns.rolling(roll_window).skew().values,
        "Kurtosis": log_returns.rolling(roll_window).kurt().values,
    })
    roll_df["Date"] = pd.to_datetime(roll_df["Date"]).dt.tz_localize(None)
    roll_df = roll_df.dropna()
    return roll_df

def calculate_cusum_test(close: pd.Series):
    log_returns_struct = np.log(close).diff().dropna()

    # For CUSUM test, we regress returns on a constant (testing for mean shift)
    X = sm.add_constant(np.arange(len(log_returns_struct)))
    y = log_returns_struct.values

    # Fit OLS model
    model = sm.OLS(y, X).fit()
    residuals = model.resid

    # Compute CUSUM statistics
    n = len(residuals)
    sigma = np.std(residuals, ddof=1)
    cusum = np.cumsum(residuals) / (sigma * np.sqrt(n))
    t = np.arange(1, n + 1) / n

    # Simplified boundary (commonly used)
    simple_upper = 0.948 + 2 * 0.948 * t
    simple_lower = -0.948 - 2 * 0.948 * t

    # Detect breaks: where CUSUM exceeds boundaries
    breaks_detected = np.where((cusum > simple_upper) | (cusum < simple_lower))[0]

    # Create CUSUM dataframe for Altair
    cusum_df = pd.DataFrame({
        'Date': log_returns_struct.index,
        'CUSUM': cusum,
        'Upper Bound': simple_upper,
        'Lower Bound': simple_lower,
        'Log Returns': log_returns_struct.values
    })

    return breaks_detected, cusum_df

def calculate_pre_post_earnings_behavior(df: pd.DataFrame, log_returns: pd.Series, earnings_dates: pd.Series, n_days: int):
    returns_around_earnings = []
    for event_date in earnings_dates:
        event_date = to_utc_naive(event_date)  # ensure consistent

        try:
            closest_idx = df.index.get_indexer([event_date], method='nearest')[0]
            if closest_idx < 0:
                continue
            idx = closest_idx
        except (KeyError, IndexError):
            continue

        expected_len = 2 * n_days + 1
        padded_window = np.full(expected_len, np.nan)

        data_start = max(0, idx - n_days)
        data_end   = min(len(log_returns), idx + n_days + 1)
        padded_start = data_start - (idx - n_days)

        window = log_returns.iloc[data_start:data_end].values
        if len(window) > 0:
            padded_window[padded_start:padded_start + len(window)] = window

        returns_around_earnings.append(padded_window)
    
    return returns_around_earnings

def calculate_arch_lm_test(log_returns: pd.Series, lags: int):
    arch_lm_result = het_arch(log_returns, nlags=lags)
    arch_lm_df = pd.DataFrame({
        "Test": ["ARCH-LM", "F-test"],
        "Statistic": [f"{arch_lm_result[0]:.4f}", f"{arch_lm_result[2]:.4f}"],
        "p-value": [f"{arch_lm_result[1]:.4g}", f"{arch_lm_result[3]:.4g}"],
    })
    return arch_lm_df

def calculate_garch_volatility_test(log_returns: pd.Series, p: int = 1, q: int = 1):
    garch_model = arch_model(log_returns * 100, vol='Garch', p=p, q=q, mean='Constant', rescale=False)
    garch_res = garch_model.fit(disp='off')

    # Extract conditional volatility (annualized)
    cond_vol = garch_res.conditional_volatility * np.sqrt(252) / 100  # daily sigma, annualized

    # Create DataFrame for Altair
    vol_df = pd.DataFrame({
        "Date": log_returns.index,
        "Volatility": cond_vol.values
    })

    return vol_df, garch_res


# ── Strategy Selection EDA ─────────────────────────────────────────────────────

def calculate_acf_ljungbox(log_returns: pd.Series, n_lags: int = 40):
    """Compute ACF values with CI bound and Ljung-Box test results."""
    acf_values, confint = acf(log_returns, nlags=n_lags, alpha=0.05)
    # CI bound ≈ 1.96 / sqrt(n), derived from the confint returned by statsmodels
    ci_bound = float((confint[1:, 1] - acf_values[1:]).mean())

    lb_test = acorr_ljungbox(log_returns, lags=[1, 5, 10, 20], return_df=True)
    lb_test.index.name = "Lag"
    lb_df = lb_test.reset_index()
    lb_df.columns = ["Lag", "LB Statistic", "p-value"]
    lb_df["Significant"] = lb_df["p-value"] < 0.05
    return acf_values, ci_bound, lb_df


def calculate_hurst_exponent(series: pd.Series, min_window: int = 10, n_points: int = 20):
    """Compute Hurst exponent via Rescaled Range (R/S) analysis."""
    values = np.array(series.dropna())
    n = len(values)
    max_window = n // 2

    windows = np.unique(
        np.logspace(np.log10(min_window), np.log10(max_window), n_points).astype(int)
    )

    log_windows, log_rs = [], []
    for w in windows:
        rs_vals = []
        for start in range(0, n - w + 1, w):
            sub = values[start : start + w]
            mean = sub.mean()
            dev = np.cumsum(sub - mean)
            R = dev.max() - dev.min()
            S = sub.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            log_windows.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_vals)))

    log_windows = np.array(log_windows)
    log_rs = np.array(log_rs)
    H = float(np.polyfit(log_windows, log_rs, 1)[0])
    return H, log_windows, log_rs


def _fast_hurst(values: np.ndarray) -> float:
    """Faster Hurst approximation using fewer windows (for rolling use)."""
    n = len(values)
    max_w = n // 2
    windows = np.unique(np.logspace(np.log10(10), np.log10(max_w), 8).astype(int))
    log_w, log_rs = [], []
    for w in windows:
        rs_vals = []
        for start in range(0, n - w + 1, w):
            sub = values[start : start + w]
            mean = sub.mean()
            dev = np.cumsum(sub - mean)
            R = dev.max() - dev.min()
            S = sub.std(ddof=1)
            if S > 0:
                rs_vals.append(R / S)
        if rs_vals:
            log_w.append(np.log(w))
            log_rs.append(np.log(np.mean(rs_vals)))
    if len(log_w) < 2:
        return 0.5
    return float(np.polyfit(log_w, log_rs, 1)[0])


def calculate_adf_stationarity(close: pd.Series, log_returns: pd.Series):
    """Run ADF test on price and returns. Returns (df, price_is_stationary, price_pvalue)."""
    def _run(series, name):
        result = adfuller(series.dropna(), autolag="AIC")
        return {
            "Series": name,
            "ADF Statistic": result[0],
            "p-value": result[1],
            "5% Critical": result[4]["5%"],
            "Stationary": bool(result[1] < 0.05),
        }

    adf_price = _run(close, "Close Price")
    adf_returns = _run(log_returns, "Log Returns")
    adf_df = pd.DataFrame([adf_price, adf_returns]).set_index("Series")
    return adf_df, adf_price["Stationary"], adf_price["p-value"]


def calculate_variance_ratio(log_returns: pd.Series, lags: list = None):
    """Lo-MacKinlay Variance Ratio test."""
    if lags is None:
        lags = [2, 5, 10, 20, 30]
    series = log_returns.dropna().values
    n = len(series)
    var_1 = np.var(series, ddof=1)
    results = []
    for k in lags:
        agg = np.array([series[i : i + k].sum() for i in range(n - k + 1)])
        var_k = np.var(agg, ddof=1)
        vr = var_k / (k * var_1)
        z_stat = (vr - 1) / np.sqrt(2 * (k - 1) / (n * k))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        results.append({
            "Lag (k)": k,
            "VR(k)": round(vr, 4),
            "z-stat": round(z_stat, 4),
            "p-value": round(p_value, 4),
            "Signal": "Trend ↑" if vr > 1 else "Revert ↓",
        })
    return pd.DataFrame(results)


def calculate_half_life_mr(series: pd.Series):
    """Compute half-life of mean reversion via OU process OLS.
    Returns (half_life_days, theta). half_life is inf when no mean reversion.
    """
    series = series.dropna()
    y_diff = series.diff().dropna()
    y_lag = series.shift(1).dropna()
    common = y_diff.index.intersection(y_lag.index)
    y_diff, y_lag = y_diff.loc[common], y_lag.loc[common]
    X = sm.add_constant(y_lag)
    model = sm.OLS(y_diff, X).fit()
    theta = float(model.params.iloc[1])
    if theta >= 0 or theta <= -1:
        return np.inf, theta
    return float(-np.log(2) / np.log(1 + theta)), theta


def calculate_rolling_regime(log_returns: pd.Series, window: int = 252):
    """Rolling Hurst exponent and VR(10).

    Returns (roll_hurst, roll_vr, pct_trending, current_hurst, effective_window).
    The effective_window may be smaller than *window* when available data is short.
    """
    values = log_returns.dropna().values
    index = log_returns.dropna().index
    n = len(values)
    k = 10  # VR lag

    # Adapt window so the rolling loop produces meaningful data even for short
    # periods.  Require at least 30 days so _fast_hurst has enough observations.
    effective_window = min(window, n // 2)
    if effective_window < 30:
        roll_hurst = pd.Series(np.full(n, np.nan), index=index)
        roll_vr = pd.Series(np.full(n, np.nan), index=index)
        return roll_hurst, roll_vr, 50.0, np.nan, effective_window

    hurst_vals = np.full(n, np.nan)
    vr_vals = np.full(n, np.nan)

    for i in range(effective_window, n + 1):
        sub = values[i - effective_window : i]
        # Rolling VR(10)
        var_1 = np.var(sub, ddof=1)
        if var_1 > 0:
            agg = np.array([sub[j : j + k].sum() for j in range(len(sub) - k + 1)])
            vr_vals[i - 1] = np.var(agg, ddof=1) / (k * var_1)
        # Rolling Hurst (fast approximation)
        try:
            hurst_vals[i - 1] = _fast_hurst(sub)
        except Exception:
            pass

    roll_hurst = pd.Series(hurst_vals, index=index)
    roll_vr = pd.Series(vr_vals, index=index)
    valid_hurst = roll_hurst.dropna()
    pct_trending = float((valid_hurst > 0.5).sum() / len(valid_hurst) * 100) if len(valid_hurst) > 0 else 50.0
    current_hurst = float(valid_hurst.iloc[-1]) if len(valid_hurst) > 0 else np.nan
    return roll_hurst, roll_vr, pct_trending, current_hurst, effective_window


def calculate_strategy_scorecard(
    acf_values: np.ndarray,
    lb_df: pd.DataFrame,
    H: float,
    adf_price_stationary: bool,
    adf_price_pvalue: float,
    vr_df: pd.DataFrame,
    hl_50: float,
    roll_hurst: pd.Series,
):
    """Build strategy evidence scorecard. Returns (scorecard_df, trend_votes, revert_votes, neutral_votes)."""
    evidence = []

    avg_acf_short = float(np.mean(acf_values[1:6]))
    if avg_acf_short > 0.03:
        evidence.append(("Autocorrelation (lag 1-5)", "Trend Following", f"Avg ACF = {avg_acf_short:+.4f}"))
    elif avg_acf_short < -0.03:
        evidence.append(("Autocorrelation (lag 1-5)", "Mean Reversion", f"Avg ACF = {avg_acf_short:+.4f}"))
    else:
        evidence.append(("Autocorrelation (lag 1-5)", "Neutral", f"Avg ACF = {avg_acf_short:+.4f}"))

    lb_significant = bool((lb_df["p-value"] < 0.05).any())
    evidence.append((
        "Ljung-Box Test",
        "Predictable" if lb_significant else "Random Walk",
        f"Significant at 5%: {lb_significant}",
    ))

    if H > 0.55:
        evidence.append(("Hurst Exponent", "Trend Following", f"H = {H:.4f}"))
    elif H < 0.45:
        evidence.append(("Hurst Exponent", "Mean Reversion", f"H = {H:.4f}"))
    else:
        evidence.append(("Hurst Exponent", "Neutral", f"H = {H:.4f}"))

    if adf_price_stationary:
        evidence.append(("ADF (Price)", "Mean Reversion", f"p = {adf_price_pvalue:.4f}"))
    else:
        evidence.append(("ADF (Price)", "Non-Stationary (typical)", f"p = {adf_price_pvalue:.4f}"))

    avg_vr = float(vr_df["VR(k)"].mean())
    if avg_vr > 1.05:
        evidence.append(("Variance Ratio (avg)", "Trend Following", f"Avg VR = {avg_vr:.4f}"))
    elif avg_vr < 0.95:
        evidence.append(("Variance Ratio (avg)", "Mean Reversion", f"Avg VR = {avg_vr:.4f}"))
    else:
        evidence.append(("Variance Ratio (avg)", "Neutral", f"Avg VR = {avg_vr:.4f}"))

    if np.isinf(hl_50) or hl_50 >= 60:
        hl_str = "∞" if np.isinf(hl_50) else f"{hl_50:.1f}"
        evidence.append(("Half-Life (SMA50)", "No Mean Reversion", f"{hl_str} days"))
    elif hl_50 < 20:
        evidence.append(("Half-Life (SMA50)", "Mean Reversion", f"{hl_50:.1f} days"))
    else:
        evidence.append(("Half-Life (SMA50)", "Weak Mean Reversion", f"{hl_50:.1f} days"))

    valid_hurst = roll_hurst.dropna()
    if len(valid_hurst) > 0:
        pct_trending = float((valid_hurst > 0.5).sum() / len(valid_hurst) * 100)
        if pct_trending > 60:
            evidence.append(("Rolling Regime", "Trend Following", f"{pct_trending:.1f}% trending"))
        elif pct_trending < 40:
            evidence.append(("Rolling Regime", "Mean Reversion", f"{100 - pct_trending:.1f}% reverting"))
        else:
            evidence.append(("Rolling Regime", "Mixed", f"{pct_trending:.1f}% trending"))
    else:
        evidence.append(("Rolling Regime", "Insufficient Data", "N/A"))

    scorecard = pd.DataFrame(evidence, columns=["Test", "Verdict", "Value"])
    trend_votes = sum(1 for _, v, _ in evidence if "Trend" in v)
    revert_votes = sum(1 for _, v, _ in evidence if "Reversion" in v or "Revert" in v)
    neutral_votes = len(evidence) - trend_votes - revert_votes
    return scorecard, trend_votes, revert_votes, neutral_votes