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