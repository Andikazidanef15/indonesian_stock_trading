# train.py — EDIT THIS FILE to experiment with new strategy ideas.
#
# Rules:
#   - Modify get_signal() and the strategy functions below.
#   - Do NOT modify prepare.py — it is the fixed evaluation oracle.
#   - Do NOT add new package imports beyond what is already imported here.
#   - The goal: maximise oos_sharpe on the held-out test set.
#
# Run:  python train.py > run.log 2>&1
# Read: grep "^oos_sharpe:\|^max_drawdown:" run.log

import pandas as pd
import numpy as np
import talib

from prepare import backtest_strategy, load_data, load_ihsg_data, print_results

# ── Strategy functions ────────────────────────────────────────────────────────
# Modify these or add new ones. They are the equivalent of model architecture
# in the ML autoresearch framework.

def trend_following_signal(df, window=20):
    """Donchian Channel Breakout. Long on new N-day high, exit on new N-day low."""
    high_roll = df["High"].rolling(window).max()
    low_roll  = df["Low"].rolling(window).min()
    signal = np.zeros(len(df), dtype=int)

    long_entries = df["Close"] > high_roll.shift(1)
    exits        = df["Close"] < low_roll.shift(1)

    in_trade = False
    for i in range(len(df)):
        if not in_trade and long_entries.iloc[i]:
            in_trade = True
        elif in_trade and exits.iloc[i]:
            in_trade = False
        signal[i] = 1 if in_trade else 0

    return pd.Series(signal, index=df.index)


def mean_reversion_signal(df, bb_period=20, bb_std=2.0, entry_z=-1.0, exit_z=0.0):
    """Bollinger Band Z-Score. Long when oversold (Z < entry_z), exit at mean (Z > exit_z)."""
    close = df["Close"]
    bb_upper, bb_mid, bb_lower = talib.BBANDS(
        close, timeperiod=bb_period, nbdevup=bb_std, nbdevdn=bb_std
    )
    bb_width = (bb_upper - bb_lower) / 2
    zscore = (close - bb_mid) / (bb_width + 1e-10)

    signal = pd.Series(0, index=df.index)
    in_pos = False
    for i in range(1, len(signal)):
        if not in_pos:
            if zscore.iloc[i] < entry_z:
                in_pos = True
                signal.iloc[i] = 1
        else:
            signal.iloc[i] = 1
            if zscore.iloc[i] > exit_z:
                in_pos = False
                signal.iloc[i] = 0

    return signal


def volatility_position_scale(df, lookback=20, target_ann_vol=0.15):
    """Scale exposure down when realized vol exceeds target (vol targeting)."""
    r = np.log(df["Close"]).diff()
    realized_daily = r.rolling(lookback).std()
    target_daily   = target_ann_vol / np.sqrt(252)
    scale = (target_daily / (realized_daily + 1e-8)).clip(upper=1.0)
    return scale.fillna(1.0)


def regime_switching_strategy(
    df,
    window=20,
    bb_period=20,
    bb_std=2.0,
    entry_z=-1.0,
    exit_z=0.0,
    atr_fast=10,
    atr_slow=60,
    vol_ratio_threshold=1.0,
    sma_trend=200,
    ihsg_close=None,
    ihsg_sma_period=50,
    use_ihsg_filter=True,
    use_vol_target_sizing=True,
    target_ann_vol=0.15,
    vol_scale_lookback=20,
):
    """
    Regime-switching: ATR vol-ratio + SMA200 selects trend vs mean-reversion leg.
    Optional IHSG market filter and volatility targeting.
    """
    tf_sig = trend_following_signal(df, window)
    mr_sig = mean_reversion_signal(df, bb_period, bb_std, entry_z, exit_z)

    high, low, close = df["High"], df["Low"], df["Close"]
    atr_f    = talib.ATR(high, low, close, timeperiod=atr_fast)
    atr_s    = talib.ATR(high, low, close, timeperiod=atr_slow)
    vol_ratio = atr_f / (atr_s + 1e-10)
    sma_long  = talib.SMA(close, timeperiod=sma_trend)

    trend_flag = ((vol_ratio < vol_ratio_threshold) & (close > sma_long)).astype(float)

    tf_eff = tf_sig.astype(float)
    if use_ihsg_filter and ihsg_close is not None:
        ihsg_ma    = pd.Series(talib.SMA(ihsg_close.values, timeperiod=ihsg_sma_period), index=df.index)
        market_bull = (ihsg_close >= ihsg_ma).astype(float).fillna(1.0)
        tf_eff = tf_eff * market_bull

    combined = trend_flag * tf_eff + (1.0 - trend_flag) * mr_sig.astype(float)
    combined = combined.clip(0.0, 1.0)

    nan_mask = vol_ratio.isna() | sma_long.isna()
    combined = combined.where(~nan_mask, tf_eff)

    if use_vol_target_sizing:
        scale    = volatility_position_scale(df, vol_scale_lookback, target_ann_vol)
        combined = (combined * scale).clip(0.0, 1.0)

    return combined

# ── Entry point — modify this to experiment ───────────────────────────────────

def get_signal(train_data, test_data, ihsg_close):
    """
    Return the test-period position signal [0, 1].

    This is the function the autoresearch loop modifies each iteration.
    Parameters are fitted on train_data; signal is applied to test_data.

    Ideas to try:
      - Different regime detectors (Hurst, VIX-equivalent, spread)
      - Different entry/exit rules (RSI filter, volume confirmation)
      - Parameter search on train_data with Optuna/Hyperopt
      - Ensemble of multiple signals
    """
    # Align IHSG to test index
    ihsg_test = ihsg_close.reindex(test_data.index).ffill().bfill()

    signal = regime_switching_strategy(
        test_data,
        window=20,
        bb_period=20,
        bb_std=2.0,
        entry_z=-1.0,
        exit_z=0.0,
        atr_fast=10,
        atr_slow=60,
        vol_ratio_threshold=1.0,
        sma_trend=200,
        ihsg_close=ihsg_test,
        ihsg_sma_period=50,
        use_ihsg_filter=True,
        use_vol_target_sizing=True,
        target_ann_vol=0.15,
        vol_scale_lookback=20,
    )
    return signal

# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    TICKER = "BBCA.JK"
    slug   = TICKER.replace(".", "_")

    train_data = None
    test_data  = None
    ihsg_close = None

    try:
        train_data, test_data = load_data(
            f"data/train_{slug}.csv",
            f"data/test_{slug}.csv",
        )
        ihsg_close = load_ihsg_data("data/ihsg_close.csv")
    except FileNotFoundError:
        print("Data not found. Run: python prepare.py")
        raise

    signal  = get_signal(train_data, test_data, ihsg_close)
    results = backtest_strategy(signal, test_data, transaction_cost=0.003)
    print_results(results)
