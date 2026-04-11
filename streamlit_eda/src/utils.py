import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import statsmodels.api as sm

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching data…", ttl="6h")
def load_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(period=period)
    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the ticker symbol.")
    return df

def load_external_returns(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    df = yf.Ticker(ticker).history(start=start, end=end)
    if df is None or df.empty:
        raise ValueError(f"No data returned for '{ticker}'. Check the ticker symbol.")

    # Handle MultiIndex columns from yfinance (newer versions return MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df_close = df['Close'].iloc[:, 0] if df['Close'].ndim > 1 else df['Close']
    else:
        df_close = df['Close']

    # Ensure the series are 1D and have proper index
    df_close = pd.Series(df_close.values.flatten(), index=df_close.index) if len(df_close) > 0 else pd.Series(dtype=float)
    df_returns = np.log(df_close).diff().dropna() if len(df_close) > 0 else pd.Series(dtype=float)

    return df_returns

def combine_returns(tickers: list[str], returns_list: list[pd.Series]) -> pd.DataFrame:
    """
    Combine multiple return series into a single DataFrame.
    
    Args:
        tickers: List of ticker names corresponding to each return series
        returns_list: List of pd.Series containing log returns for each ticker
    
    Returns:
        DataFrame with aligned returns for all tickers
    """
    if len(tickers) != len(returns_list):
        raise ValueError("Number of tickers must match number of return series")
    
    processed_series = []
    
    for ticker, returns in zip(tickers, returns_list):
        # Ensure returns is a 1D Series
        if isinstance(returns, pd.DataFrame):
            returns = returns.iloc[:, 0]
        returns = pd.Series(returns.values.flatten(), index=returns.index, name=ticker)
        
        # Convert timezone-aware indices to timezone-naive for proper alignment
        if len(returns) > 0 and returns.index.tz is not None:
            returns.index = returns.index.tz_localize(None)
        
        # Normalize index to date only (remove time component) for proper alignment
        if len(returns) > 0:
            returns.index = pd.to_datetime(returns.index).normalize()
        
        processed_series.append(returns)

    # Use outer join first to see all data, then fill or handle missing
    combined_returns = pd.concat(processed_series, axis=1, join='outer')
    combined_returns.columns = tickers

    # Now drop rows with any NaN
    combined_returns = combined_returns.dropna()

    # Remove any remaining NaN or infinite values
    combined_returns = combined_returns.replace([np.inf, -np.inf], np.nan).dropna()

    return combined_returns

def to_utc_naive(dt):
    dt = pd.to_datetime(dt)
    if dt.tzinfo is not None:
        dt = dt.tz_convert('UTC').tz_localize(None)
    return dt

def get_earnings_dates(df:pd.DataFrame, ticker: str):
    calendar = yf.Ticker(ticker).calendar
    earnings_dates = []

    # Extract earnings dates from the calendar data
    if 'Earnings Date' in calendar and calendar['Earnings Date']:
        for ed in calendar['Earnings Date']:
            earnings_dates.append(to_utc_naive(ed))

    # Normalize df index to tz-naive UTC as well
    if df.index.tzinfo is not None:
        df.index = df.index.tz_convert('UTC').tz_localize(None)

    df_index_min = df.index.min()

    # Estimate past quarterly earnings
    if earnings_dates:
        latest_earnings = earnings_dates[0]
        for i in range(1, 20):
            past_date = to_utc_naive(latest_earnings - pd.DateOffset(months=3*i))
            if past_date >= df_index_min:
                earnings_dates.append(past_date)

    # Now safe to convert — all are tz-naive
    earnings_dates = pd.to_datetime(earnings_dates)

    return earnings_dates