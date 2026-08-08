"""
Risk and performance metrics for strategy equity curves.

All formulas use daily returns unless noted.  Annualization assumes
`trading_days_per_year` (default 252).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass
class PerformanceReport:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    n_trades: int
    avg_trade_return: float

    def to_dict(self) -> dict:
        return asdict(self)


def daily_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    """
    Maximum peak-to-trough decline of the equity curve.

    MDD = min_t (E_t / max_{s<=t} E_s - 1)   (negative number)
    """
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """
    Annualized Sharpe:

        SR = sqrt(N) * mean(r - r_f_daily) / std(r)

    where r_f_daily = (1 + r_f_annual)^(1/N) - 1.
    """
    if returns.empty or returns.std(ddof=1) == 0:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_daily
    return float(np.sqrt(periods_per_year) * excess.mean() / excess.std(ddof=1))


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> float:
    """Like Sharpe, but denominator is downside deviation (std of negative excess)."""
    if returns.empty:
        return 0.0
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = returns - rf_daily
    downside = excess[excess < 0]
    if downside.empty or downside.std(ddof=1) == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * excess.mean() / downside.std(ddof=1))


def count_trades(signal: pd.Series) -> int:
    """Count entries: each transition from 0 -> ±1 or flip ±1 -> ∓1."""
    s = signal.fillna(0).astype(int)
    # Count entries (flat -> ±1) and direction flips (±1 -> ∓1).
    entries = ((s.shift(1).fillna(0) == 0) & (s != 0)).sum()
    flips = ((s.shift(1).fillna(0) * s) < 0).sum()
    return int(entries + flips)


def summarize_performance(
    equity: pd.Series,
    signal: pd.Series | None = None,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
) -> PerformanceReport:
    rets = daily_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0) if len(equity) > 1 else 0.0
    ann_ret = float((1.0 + total_return) ** (periods_per_year / max(len(rets), 1)) - 1.0)
    ann_vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if len(rets) else 0.0
    mdd = max_drawdown(equity)
    sharpe = sharpe_ratio(rets, risk_free_rate, periods_per_year)
    sortino = sortino_ratio(rets, risk_free_rate, periods_per_year)
    calmar = float(ann_ret / abs(mdd)) if mdd < 0 else 0.0

    if signal is not None:
        n_trades = count_trades(signal)
        # Approximate per-trade return using returns on days with non-zero position.
        active = rets.loc[signal.reindex(rets.index).fillna(0).astype(int) != 0]
        win_rate = float((active > 0).mean()) if len(active) else 0.0
        avg_trade = float(active.mean()) if len(active) else 0.0
    else:
        n_trades = 0
        win_rate = float((rets > 0).mean()) if len(rets) else 0.0
        avg_trade = float(rets.mean()) if len(rets) else 0.0

    return PerformanceReport(
        total_return=total_return,
        annualized_return=ann_ret,
        annualized_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate,
        n_trades=n_trades,
        avg_trade_return=avg_trade,
    )
