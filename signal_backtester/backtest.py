"""
Minimal, honest backtest engine: lagged execution (no look-ahead), optional
transaction costs, and the metrics that actually matter for a resume bullet
or an interview answer -- not just "it worked."

Usage as a library:

    from data import load_prices
    from signals import zscore_mean_reversion
    from backtest import run_backtest

    close = load_prices("AAPL", start="2020-01-01")["Close"]
    position = zscore_mean_reversion(close, window=20, entry_z=1.5, exit_z=0.3)
    result = run_backtest(close, position, cost_bps=5.0)
    result.summary()
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    strategy_returns: pd.Series
    baseline_returns: pd.Series
    trades: int
    hit_rate: float
    total_return: float
    baseline_total_return: float
    sharpe: float
    baseline_sharpe: float
    max_drawdown: float
    baseline_max_drawdown: float

    def summary(self) -> str:
        lines = [
            f"Trades:                 {self.trades}",
            f"Hit rate (per trade):   {self.hit_rate:.1%}",
            f"Total return:           {self.total_return:+.1%}   (buy-and-hold: {self.baseline_total_return:+.1%})",
            f"Sharpe (rf=2%):         {self.sharpe:+.2f}   (buy-and-hold: {self.baseline_sharpe:+.2f})",
            f"Max drawdown:           {self.max_drawdown:.1%}   (buy-and-hold: {self.baseline_max_drawdown:.1%})",
        ]
        return "\n".join(lines)


def _sharpe(daily_returns: pd.Series, rf_annual: float = 0.02) -> float:
    daily_returns = daily_returns.dropna()
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    rf_daily = rf_annual / 252
    excess = daily_returns - rf_daily
    return float(np.sqrt(252) * excess.mean() / daily_returns.std())


def _max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def _count_trades(position: pd.Series) -> int:
    changes = position.diff().fillna(0)
    entries = ((position != 0) & (changes != 0)).sum()
    return int(entries)


def _hit_rate(position: pd.Series, daily_returns: pd.Series) -> float:
    """Fraction of *trades* (contiguous non-flat position segments) that were
    net profitable, not just fraction of profitable days."""
    trade_id = (position.diff().fillna(0) != 0).cumsum()
    trade_returns = []
    for _, group in daily_returns.groupby(trade_id):
        if group.empty:
            continue
        trade_returns.append((1 + group).prod() - 1)
    trade_returns = [r for r in trade_returns if r != 0]
    if not trade_returns:
        return 0.0
    wins = sum(1 for r in trade_returns if r > 0)
    return wins / len(trade_returns)


def run_backtest(
    close: pd.Series,
    position: pd.Series,
    cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    rf_annual: float = 0.02,
) -> BacktestResult:
    """Backtest a position series against a price series.

    Positions are lagged by one day (signal computed on day t's close only
    earns the t -> t+1 return), matching the look-ahead-bias guard used in
    the stat-arb engine. Costs are charged on turnover (position changes).
    """
    close = close.dropna()
    position = position.reindex(close.index).fillna(0)

    daily_ret = close.pct_change().fillna(0)
    lagged_position = position.shift(1).fillna(0)

    turnover = position.diff().abs().fillna(0)
    cost_per_change = (cost_bps + slippage_bps) / 10_000.0
    costs = turnover * cost_per_change

    strategy_returns = lagged_position * daily_ret - costs
    baseline_returns = daily_ret

    equity_curve = (1 + strategy_returns).cumprod()
    baseline_equity = (1 + baseline_returns).cumprod()

    return BacktestResult(
        equity_curve=equity_curve,
        strategy_returns=strategy_returns,
        baseline_returns=baseline_returns,
        trades=_count_trades(position),
        hit_rate=_hit_rate(position, strategy_returns),
        total_return=float(equity_curve.iloc[-1] - 1) if len(equity_curve) else 0.0,
        baseline_total_return=float(baseline_equity.iloc[-1] - 1) if len(baseline_equity) else 0.0,
        sharpe=_sharpe(strategy_returns, rf_annual),
        baseline_sharpe=_sharpe(baseline_returns, rf_annual),
        max_drawdown=_max_drawdown(equity_curve),
        baseline_max_drawdown=_max_drawdown(baseline_equity),
    )
