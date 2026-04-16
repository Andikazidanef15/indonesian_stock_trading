# prepare.py — FIXED. Do not modify.
# Contains: data loading, IHSG alignment, train/test split, backtest evaluation
# harness, and standardized output format.
#
# Run once to populate data/:
#   python prepare.py

import pandas as pd
import numpy as np
import yfinance as yf

# ── Data loading ─────────────────────────────────────────────────────────────

def load_data(train_path: str, test_path: str):
    train_data = pd.read_csv(train_path, index_col=0, parse_dates=True)
    test_data = pd.read_csv(test_path, index_col=0, parse_dates=True)
    return train_data, test_data

def load_ihsg_data(ihsg_path: str):
    return pd.read_csv(ihsg_path, index_col=0, parse_dates=True).squeeze()

def load_ihsg_close_aligned(index, buffer_days=400):
    """Download ^JKSE and align to the stock's trading calendar (forward-fill)."""
    start = (pd.Timestamp(index.min()) - pd.Timedelta(days=buffer_days)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(index.max()) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    ih = yf.download("^JKSE", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(ih.columns, pd.MultiIndex):
        ih.columns = ih.columns.get_level_values(-1)
    ih = ih.squeeze().reindex(index).ffill().bfill()
    return ih

def split_data(data, test_size=0.2):
    train_size = int(len(data) * (1 - test_size))
    return data[:train_size], data[train_size:]

# ── Evaluation harness (fixed — do not modify) ────────────────────────────────

def backtest_strategy(signal, df, transaction_cost=0.003):
    """
    Backtest long/flat/fractional exposure on log returns.
    This is the fixed evaluation oracle. Do not modify.

    Parameters
    ----------
    signal : pd.Series  position size in [0, 1]
    df     : pd.DataFrame  must contain 'Close'
    transaction_cost : float  applied to |Δposition| per day
    """
    signal = signal.reindex(df.index).fillna(0.0)
    log_ret = np.log(df["Close"]).diff()
    strat_ret = signal.shift(1) * log_ret

    turnover = signal.diff().abs()
    costs = turnover * transaction_cost
    strat_ret_net = (strat_ret - costs).dropna()
    equity = strat_ret_net.cumsum()

    n_days = len(strat_ret_net)
    total_ret = equity.iloc[-1] if not equity.empty else 0.0
    ann_ret = total_ret / (n_days / 252) if n_days > 0 else 0.0
    ann_vol = strat_ret_net.std() * np.sqrt(252)
    sharpe = (
        strat_ret_net.mean() / strat_ret_net.std() * np.sqrt(252)
        if strat_ret_net.std() > 0 else 0.0
    )

    drawdown = equity - equity.cummax()
    max_dd = drawdown.min() if not drawdown.empty else 0.0

    winning = strat_ret_net[strat_ret_net > 0]
    losing  = strat_ret_net[strat_ret_net < 0]
    win_rate = len(winning) / (len(winning) + len(losing)) if (len(winning) + len(losing)) > 0 else 0.0

    gross_profit = winning.sum()
    gross_loss   = losing.abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    n_trades = int((turnover > 1e-6).sum())
    exposure = (signal > 1e-6).mean()

    return {
        "total_return":   total_ret,
        "ann_return":     ann_ret,
        "ann_volatility": ann_vol,
        "sharpe_ratio":   sharpe,
        "max_drawdown":   max_dd,
        "win_rate":       win_rate,
        "profit_factor":  profit_factor,
        "n_trades":       n_trades,
        "avg_exposure":   float(signal.mean()),
        "days_in_market": float(exposure),
        "equity":         equity,
        "returns":        strat_ret_net,
    }

def print_results(results: dict):
    """
    Print results in standardized format.
    Log file can be grepped with: grep "^oos_sharpe:" run.log
    """
    pf = results["profit_factor"]
    pf_str = f"{pf:.6f}" if pf != float("inf") else "inf"
    print("---")
    print(f"oos_sharpe:       {results['sharpe_ratio']:.6f}")
    print(f"max_drawdown:     {results['max_drawdown']:.6f}")
    print(f"ann_return:       {results['ann_return']:.6f}")
    print(f"ann_volatility:   {results['ann_volatility']:.6f}")
    print(f"win_rate:         {results['win_rate']:.6f}")
    print(f"profit_factor:    {pf_str}")
    print(f"n_trades:         {results['n_trades']}")
    print(f"avg_exposure:     {results['avg_exposure']:.6f}")

# ── Populate data/ (run once) ─────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)

    TICKER     = "BBCA.JK"
    START_DATE = "2010-01-01"

    print(f"Downloading {TICKER} from {START_DATE}...")
    raw = yf.Ticker(TICKER).history(start=START_DATE)
    train, test = split_data(raw)

    slug = TICKER.replace(".", "_")
    train.to_csv(f"data/train_{slug}.csv", index=True)
    test.to_csv(f"data/test_{slug}.csv",   index=True)
    print(f"Saved data/train_{slug}.csv ({len(train)} rows)")
    print(f"Saved data/test_{slug}.csv  ({len(test)} rows)")

    print("Downloading ^JKSE (IHSG)...")
    ihsg = load_ihsg_close_aligned(raw.index)
    ihsg.to_csv("data/ihsg_close.csv", index=True)
    print(f"Saved data/ihsg_close.csv ({len(ihsg)} rows)")
    print("Data ready. Run: python train.py")
