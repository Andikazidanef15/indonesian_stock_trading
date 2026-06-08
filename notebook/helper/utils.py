import numpy as np

def backtest_strategy(signal, df, transaction_cost=0.003, label="Strategy", risk_free=0.0):
    """
    Backtest long / flat / fractional exposure on log returns.

    Parameters
    ----------
    signal : pd.Series — position size in [0, 1] (0/1 binary or fractional vol targeting)
    df : pd.DataFrame with 'Close'
    transaction_cost : float — cost applied to |Δposition| each day (0.003 ≈ 0.3% per unit change)
    label : str
    risk_free : float — annualised risk-free rate used for Sharpe/Sortino (default 0.0)
    """
    signal = signal.reindex(df.index).fillna(0.0)
    log_ret = np.log(df["Close"]).diff()
    strat_ret = signal.shift(1) * log_ret

    turnover = signal.diff().abs()
    costs = turnover * transaction_cost
    strat_ret_net = strat_ret - costs

    strat_ret_net = strat_ret_net.dropna()
    equity = strat_ret_net.cumsum()

    n_days = len(strat_ret_net)
    total_ret = equity.iloc[-1] if not equity.empty else 0
    ann_ret = total_ret / (n_days / 252) if n_days > 0 else 0
    ann_vol = strat_ret_net.std() * np.sqrt(252)

    rf_daily = risk_free / 252
    excess = strat_ret_net - rf_daily
    sharpe = (
        (excess.mean() / strat_ret_net.std()) * np.sqrt(252)
        if strat_ret_net.std() > 0
        else 0
    )

    # Sortino — penalises downside volatility only
    downside = strat_ret_net[strat_ret_net < rf_daily]
    downside_vol = downside.std() * np.sqrt(252)
    sortino = (ann_ret - risk_free) / downside_vol if downside_vol > 0 else np.inf

    # Drawdown series
    cum_max = equity.cummax()
    drawdown = equity - cum_max
    max_dd = drawdown.min() if not drawdown.empty else 0

    # Calmar — annualised return per unit of max drawdown
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.inf

    # Max drawdown duration (peak-to-recovery, in trading days)
    dd_duration = 0
    if not drawdown.empty:
        in_dd = (drawdown < 0)
        # find contiguous stretches under water
        streak = in_dd * (in_dd.groupby((~in_dd).cumsum()).cumcount() + 1)
        dd_duration = int(streak.max())

    winning = strat_ret_net[strat_ret_net > 0]
    losing  = strat_ret_net[strat_ret_net < 0]
    win_rate = (
        len(winning) / (len(winning) + len(losing))
        if (len(winning) + len(losing)) > 0
        else 0
    )

    # Payoff ratio — average win size vs average loss size
    avg_win  = winning.mean() if len(winning) > 0 else 0
    avg_loss = losing.abs().mean() if len(losing) > 0 else 0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf

    n_trades = int((turnover > 1e-6).sum())
    gross_profit = winning.sum()
    gross_loss   = losing.abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Tail-risk metrics (95 % and 99 % confidence)
    var_95  = float(np.percentile(strat_ret_net, 5))   # daily VaR (log return)
    var_99  = float(np.percentile(strat_ret_net, 1))
    cvar_95 = float(strat_ret_net[strat_ret_net <= var_95].mean())  # Expected Shortfall
    cvar_99 = float(strat_ret_net[strat_ret_net <= var_99].mean())

    # Distribution shape
    skewness = float(strat_ret_net.skew())
    kurt     = float(strat_ret_net.kurt())   # excess kurtosis

    # Best / worst single day
    best_day  = float(strat_ret_net.max())
    worst_day = float(strat_ret_net.min())

    exposure = (signal > 1e-6).mean()

    return {
        "label": label,
        "equity": equity,
        "returns": strat_ret_net,
        # ── Return ───────────────────────────────────────────
        "total_return":   total_ret,
        "ann_return":     ann_ret,
        "ann_volatility": ann_vol,
        # ── Risk-adjusted ─────────────────────────────────────
        "sharpe_ratio":   sharpe,
        "sortino_ratio":  sortino,
        "calmar_ratio":   calmar,
        # ── Drawdown ──────────────────────────────────────────
        "max_drawdown":   max_dd,
        "max_dd_duration": dd_duration,     # trading days under water
        # ── Trade statistics ──────────────────────────────────
        "win_rate":       win_rate,
        "payoff_ratio":   payoff_ratio,     # avg win / avg loss
        "profit_factor":  profit_factor,
        "n_trades":       n_trades,
        # ── Tail risk ─────────────────────────────────────────
        "var_95":         var_95,
        "cvar_95":        cvar_95,          # Expected Shortfall @ 95 %
        "var_99":         var_99,
        "cvar_99":        cvar_99,          # Expected Shortfall @ 99 %
        # ── Distribution ──────────────────────────────────────
        "skewness":       skewness,
        "kurtosis":       kurt,
        "best_day":       best_day,
        "worst_day":      worst_day,
        # ── Exposure ──────────────────────────────────────────
        "days_in_market": float(exposure),
        "avg_exposure":   float(signal.mean()),
    }