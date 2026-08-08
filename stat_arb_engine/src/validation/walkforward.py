"""
Walk-forward validation and Monte Carlo permutation significance tests.

Why walk-forward?
-----------------
A single in-sample fit of β and signal thresholds can overfit noise.  Walk-forward
estimation:
  1. Estimate hedge ratio (and optionally signal params) on a train window
  2. Trade only on the subsequent test window with those frozen parameters
  3. Roll the windows forward and stitch out-of-sample equity segments

Why Monte Carlo permutations?
-----------------------------
Even a positive OOS Sharpe can be luck.  A permutation test shuffles (or
circularly shifts) returns under a null of "no timing skill" and asks how often
a random strategy looks as good as yours.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest.engine import BacktestConfig, run_pairs_backtest
from src.backtest.metrics import PerformanceReport, sharpe_ratio, summarize_performance
from src.screening.cointegration import ols_hedge_ratio
from src.signals.zscore import SignalConfig

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    train_days: int = 756
    test_days: int = 126
    step_days: int = 63


@dataclass
class WalkForwardResult:
    oos_equity: pd.Series
    oos_detail: pd.DataFrame
    window_reports: list[dict[str, Any]]
    aggregate: PerformanceReport


def _window_bounds(
    n: int, train: int, test: int, step: int
) -> list[tuple[int, int, int, int]]:
    """Return list of (train_start, train_end, test_start, test_end) integer ilocs."""
    bounds = []
    start = 0
    while True:
        train_start = start
        train_end = start + train
        test_start = train_end
        test_end = test_start + test
        if test_end > n:
            break
        bounds.append((train_start, train_end, test_start, test_end))
        start += step
    return bounds


def walk_forward_pair(
    price_a: pd.Series,
    price_b: pd.Series,
    signal_cfg: SignalConfig | None = None,
    bt_cfg: BacktestConfig | None = None,
    wf_cfg: WalkForwardConfig | None = None,
) -> WalkForwardResult:
    signal_cfg = signal_cfg or SignalConfig()
    bt_cfg = bt_cfg or BacktestConfig()
    wf_cfg = wf_cfg or WalkForwardConfig()

    aligned = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
    aligned.columns = ["a", "b"]
    n = len(aligned)
    bounds = _window_bounds(n, wf_cfg.train_days, wf_cfg.test_days, wf_cfg.step_days)
    if not bounds:
        raise ValueError(
            f"Not enough data for walk-forward: n={n}, "
            f"need train+test={wf_cfg.train_days + wf_cfg.test_days}"
        )

    logger.info("Walk-forward: %d windows over %d days", len(bounds), n)
    oos_chunks: list[pd.DataFrame] = []
    window_reports: list[dict[str, Any]] = []

    for i, (tr0, tr1, te0, te1) in enumerate(bounds, start=1):
        train = aligned.iloc[tr0:tr1]
        # Include a short warm-up from end of train so z-score window is defined
        # at the start of the test window (no future data used for β).
        warm = max(signal_cfg.zscore_window, 5)
        test_with_warm = aligned.iloc[max(tr1 - warm, tr0) : te1]

        beta, alpha = ols_hedge_ratio(train["a"].to_numpy(), train["b"].to_numpy())
        detail, report = run_pairs_backtest(
            test_with_warm["a"],
            test_with_warm["b"],
            hedge_ratio=beta,
            intercept=alpha,
            signal_cfg=signal_cfg,
            bt_cfg=bt_cfg,
        )
        # Keep only pure OOS dates
        test_index = aligned.index[te0:te1]
        detail_oos = detail.loc[detail.index.isin(test_index)].copy()
        if detail_oos.empty:
            continue

        # Re-base equity segment to chain continuously later
        oos_chunks.append(detail_oos)
        window_reports.append(
            {
                "window": i,
                "train_start": str(train.index[0].date()),
                "train_end": str(train.index[-1].date()),
                "test_start": str(test_index[0].date()),
                "test_end": str(test_index[-1].date()),
                "hedge_ratio": beta,
                "sharpe": report.sharpe,
                "total_return": report.total_return,
                "max_drawdown": report.max_drawdown,
                "n_trades": report.n_trades,
            }
        )

    if not oos_chunks:
        raise RuntimeError("Walk-forward produced no OOS segments")

    oos_detail = pd.concat(oos_chunks).sort_index()
    oos_detail = oos_detail[~oos_detail.index.duplicated(keep="first")]

    # Chain net returns into one OOS equity curve
    net = oos_detail["ret_net"].fillna(0.0)
    oos_equity = bt_cfg.initial_capital * (1.0 + net).cumprod()
    oos_detail["equity_oos"] = oos_equity
    aggregate = summarize_performance(
        oos_equity,
        signal=oos_detail["signal"],
        risk_free_rate=bt_cfg.risk_free_rate,
        periods_per_year=bt_cfg.trading_days_per_year,
    )
    return WalkForwardResult(
        oos_equity=oos_equity,
        oos_detail=oos_detail,
        window_reports=window_reports,
        aggregate=aggregate,
    )


def monte_carlo_permutation_test(
    position: pd.Series,
    spread_returns: pd.Series,
    n_permutations: int = 1000,
    position_size: float = 0.1,
    transaction_cost_bps: float = 5.0,
    slippage_bps: float = 2.0,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Timing-skill permutation test (correct null for Sharpe).

    Important subtlety
    ------------------
    You cannot test Sharpe by shuffling the *strategy's net returns*: mean and
    std are invariant to order, so every permutation would give the same Sharpe.

    Correct null: the market path (spread returns) is fixed; only the *timing*
    of positions is random. Shuffle the position series, rebuild PnL
    (including transaction costs from the shuffled turnover), and ask how often
    a random timing rule matches your observed Sharpe.

    H0: positions have no timing skill relative to the spread returns.
    """
    aligned = pd.concat(
        {"pos": position, "r": spread_returns}, axis=1, join="inner"
    ).dropna()
    if len(aligned) < 30:
        raise ValueError("Need at least 30 observations for a meaningful permutation test")

    pos = aligned["pos"].to_numpy(dtype=float)
    r = aligned["r"].to_numpy(dtype=float)
    cost_rate = (transaction_cost_bps + slippage_bps) / 1e4

    def _sharpe_from_pos(p: np.ndarray) -> float:
        gross = p * r * position_size
        turnover = np.abs(np.diff(p, prepend=p[0]))
        costs = turnover * cost_rate * position_size
        net = gross - costs
        return sharpe_ratio(
            pd.Series(net),
            risk_free_rate=risk_free_rate,
            periods_per_year=periods_per_year,
        )

    observed = _sharpe_from_pos(pos)
    rng = np.random.default_rng(random_seed)
    null_sharpes = np.empty(n_permutations, dtype=float)
    for i in range(n_permutations):
        # Circular shift preserves run-length structure better than iid shuffle;
        # we still iid-permute here as a simple, transparent null.
        null_sharpes[i] = _sharpe_from_pos(rng.permutation(pos))

    p_value = float(np.mean(null_sharpes >= observed))
    return {
        "observed_sharpe": float(observed),
        "null_mean_sharpe": float(null_sharpes.mean()),
        "null_p05": float(np.quantile(null_sharpes, 0.05)),
        "null_p95": float(np.quantile(null_sharpes, 0.95)),
        "p_value": p_value,
        "n_permutations": n_permutations,
        "significant_at_5pct": p_value < 0.05,
        "null_type": "shuffle_positions_vs_fixed_spread_returns",
    }


def wf_config_from_yaml(cfg: dict[str, Any]) -> WalkForwardConfig:
    v = cfg.get("validation", {})
    return WalkForwardConfig(
        train_days=int(v.get("train_days", 756)),
        test_days=int(v.get("test_days", 126)),
        step_days=int(v.get("step_days", 63)),
    )
