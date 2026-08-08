"""
Vectorized-ish pairs backtester with transaction costs and slippage.

Position economics
------------------
Signal s_t ∈ {-1, 0, +1} is the desired spread direction decided on day t using
information available through the close of t.  To avoid look-ahead in execution,
we apply the position on day t+1's returns (shift by 1).

Dollar PnL for a unit of capital allocated to the spread:
    r_spread,t ≈ s_{t-1} * (r_a,t - β * r_b,t)

More carefully with notional weights:
- Long spread (+1):  +w on A,  -w * β_notional on B
- We normalize so gross exposure ≈ position_size * capital.

Costs
-----
Each time the position changes, charge:
    cost = turnover * (transaction_cost_bps + slippage_bps) / 1e4

plus a simple daily short-borrow charge on the short leg.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.backtest.metrics import PerformanceReport, summarize_performance
from src.signals.zscore import SignalConfig, generate_signals

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    position_size: float = 0.1
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    borrow_cost_bps_annual: float = 50.0
    max_holding_days: int | None = 60
    risk_free_rate: float = 0.02
    trading_days_per_year: int = 252


def _apply_time_stop(signal: pd.Series, max_holding_days: int | None) -> pd.Series:
    if max_holding_days is None:
        return signal
    out = signal.copy().astype(int)
    hold = 0
    prev = 0
    vals = out.to_numpy().copy()
    for i, s in enumerate(vals):
        if s == 0:
            hold = 0
            prev = 0
            continue
        if s == prev:
            hold += 1
        else:
            hold = 1
            prev = s
        if hold > max_holding_days:
            vals[i] = 0
            hold = 0
            prev = 0
    out[:] = vals
    return out


def run_pairs_backtest(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | None = None,
    intercept: float = 0.0,
    signal_cfg: SignalConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
) -> tuple[pd.DataFrame, PerformanceReport]:
    """
    Simulate a single pair strategy.

    Returns
    -------
    detail : DataFrame with prices, signal, returns, equity, costs
    report : PerformanceReport summary
    """
    signal_cfg = signal_cfg or SignalConfig()
    bt_cfg = bt_cfg or BacktestConfig()

    sig = generate_signals(price_a, price_b, hedge_ratio, intercept, signal_cfg)
    sig["signal"] = _apply_time_stop(sig["signal"], bt_cfg.max_holding_days)

    aligned = pd.concat(
        {
            "price_a": price_a,
            "price_b": price_b,
            "spread": sig["spread"],
            "zscore": sig["zscore"],
            "signal": sig["signal"],
        },
        axis=1,
        join="inner",
    ).dropna(subset=["price_a", "price_b"])

    # Simple returns of each leg
    r_a = aligned["price_a"].pct_change()
    r_b = aligned["price_b"].pct_change()
    beta = float(sig["hedge_ratio"].iloc[-1])

    # Position known at close t is held over t -> t+1, so shift signal by 1.
    pos = aligned["signal"].shift(1).fillna(0.0)

    # Spread return approximating long A / short β B (beta on price units ≈
    # dollar hedge when prices are comparable; for research we use return space
    # with unit weights scaled by position_size).
    # Dollar-neutral-ish: weight_a = 1, weight_b = -beta * (A/B) roughly;
    # for undergrad clarity we use: r = pos * (r_a - r_b) as a first cut when
    # beta≈1, and generalize to pos * (r_a - beta_ret * r_b).
    # Convert price-beta to a return-space hedge using last prices ratio:
    # shares_b = beta (from price regression P_a ~ beta P_b), so dollar_b / dollar_a
    # ≈ (beta * P_b) / P_a.  Using contemporaneous prices:
    dollar_ratio = (beta * aligned["price_b"]) / aligned["price_a"]
    # Avoid lookahead: use prior day's dollar ratio for today's return.
    hedge_w = dollar_ratio.shift(1).fillna(dollar_ratio.iloc[0])

    gross = bt_cfg.position_size
    # Gross exposure: |w_a| + |w_b| = gross  with w_b = -sign * hedge_w * w_a ...
    # Simpler research convention: allocate gross/2 to each leg notionally.
    raw_spread_ret = r_a - hedge_w * r_b
    strategy_ret = pos * raw_spread_ret * gross

    # Transaction costs on position changes (turnover in units of `pos`).
    turnover = pos.diff().abs().fillna(pos.abs())
    cost_rate = (bt_cfg.transaction_cost_bps + bt_cfg.slippage_bps) / 1e4
    trade_costs = turnover * cost_rate * gross

    # Borrow cost on short leg: charge when positioned.
    borrow_daily = (bt_cfg.borrow_cost_bps_annual / 1e4) / bt_cfg.trading_days_per_year
    borrow_costs = (pos.abs() > 0).astype(float) * borrow_daily * (gross / 2.0)

    net_ret = strategy_ret - trade_costs - borrow_costs
    net_ret = net_ret.fillna(0.0)

    equity = bt_cfg.initial_capital * (1.0 + net_ret).cumprod()

    detail = aligned.copy()
    detail["pos"] = pos
    detail["ret_gross"] = strategy_ret
    detail["trade_costs"] = trade_costs
    detail["borrow_costs"] = borrow_costs
    detail["ret_net"] = net_ret
    detail["equity"] = equity
    detail["hedge_ratio"] = beta

    report = summarize_performance(
        equity,
        signal=aligned["signal"],
        risk_free_rate=bt_cfg.risk_free_rate,
        periods_per_year=bt_cfg.trading_days_per_year,
    )
    return detail, report


def configs_from_yaml(cfg: dict[str, Any]) -> tuple[SignalConfig, BacktestConfig]:
    s = cfg.get("signals", {})
    b = cfg.get("backtest", {})
    m = cfg.get("metrics", {})
    signal_cfg = SignalConfig(
        zscore_window=int(s.get("zscore_window", 60)),
        entry_z=float(s.get("entry_z", 2.0)),
        exit_z=float(s.get("exit_z", 0.5)),
        stop_z=float(s.get("stop_z", 4.0)),
    )
    bt_cfg = BacktestConfig(
        initial_capital=float(b.get("initial_capital", 100_000.0)),
        position_size=float(b.get("position_size", 0.1)),
        transaction_cost_bps=float(b.get("transaction_cost_bps", 5.0)),
        slippage_bps=float(b.get("slippage_bps", 2.0)),
        borrow_cost_bps_annual=float(b.get("borrow_cost_bps_annual", 50.0)),
        max_holding_days=b.get("max_holding_days", 60),
        risk_free_rate=float(m.get("risk_free_rate", 0.02)),
        trading_days_per_year=int(m.get("trading_days_per_year", 252)),
    )
    return signal_cfg, bt_cfg


def save_backtest_result(
    db_path: Path,
    pair_id: str,
    report: PerformanceReport,
    params: dict[str, Any],
) -> str:
    run_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO backtest_results (
                run_id, pair_id, sharpe, sortino, max_drawdown, total_return,
                calmar, win_rate, n_trades, params, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                pair_id,
                report.sharpe,
                report.sortino,
                report.max_drawdown,
                report.total_return,
                report.calmar,
                report.win_rate,
                report.n_trades,
                json.dumps(params),
                now,
            ),
        )
    return run_id
