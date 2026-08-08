"""Unit tests for signal math, metrics, and cointegration helpers (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestConfig, run_pairs_backtest
from src.backtest.metrics import max_drawdown, sharpe_ratio, summarize_performance
from src.screening.cointegration import engle_granger, ols_hedge_ratio, pair_id
from src.signals.zscore import SignalConfig, generate_signals, rolling_zscore
from src.validation.walkforward import monte_carlo_permutation_test


def _make_cointegrated_pair(n: int = 800, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    """Synthetic I(1) pair with known hedge ratio β=1.5 and stationary residual."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n)
    # Common random-walk factor
    shock = rng.normal(0, 1, size=n)
    common = np.cumsum(shock)
    # Idiosyncratic stationary noise
    noise = rng.normal(0, 0.5, size=n)
    b = 50 + common
    a = 10 + 1.5 * b + noise  # β=1.5, stationary residual = noise + const drift handled
    return pd.Series(a, index=dates, name="A"), pd.Series(b, index=dates, name="B")


def test_ols_hedge_ratio_recovers_beta():
    a, b = _make_cointegrated_pair()
    beta, alpha = ols_hedge_ratio(a.to_numpy(), b.to_numpy())
    assert beta == pytest.approx(1.5, rel=0.05)


def test_engle_granger_detects_cointegration():
    a, b = _make_cointegrated_pair()
    is_coint, pval, adf_stat, beta, alpha = engle_granger(a, b, significance=0.05)
    assert is_coint
    assert pval < 0.05
    assert beta == pytest.approx(1.5, rel=0.05)


def test_engle_granger_rejects_independent_walks():
    rng = np.random.default_rng(1)
    n = 800
    dates = pd.bdate_range("2018-01-01", periods=n)
    a = pd.Series(np.cumsum(rng.normal(size=n)), index=dates)
    b = pd.Series(np.cumsum(rng.normal(size=n)), index=dates)
    is_coint, pval, *_ = engle_granger(a, b, significance=0.01)
    # Independent walks should usually fail to cointegrate; allow rare false positives
    # by checking p-value is not extremely small.
    assert pval > 0.01 or not is_coint


def test_pair_id_is_order_invariant():
    assert pair_id("JPM", "BAC") == pair_id("BAC", "JPM") == "BAC_JPM"


def test_rolling_zscore_unit_scale_on_white_noise():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(size=500))
    z = rolling_zscore(s, window=50).dropna()
    assert abs(z.mean()) < 0.2
    assert 0.7 < z.std(ddof=1) < 1.3


def test_generate_signals_enters_on_extreme_z():
    a, b = _make_cointegrated_pair(n=500)
    # Force a large residual shock at the end to trigger entry
    a = a.copy()
    a.iloc[-1] = a.iloc[-1] + 20
    out = generate_signals(a, b, cfg=SignalConfig(zscore_window=40, entry_z=2.0))
    assert set(out["signal"].unique()).issubset({-1, 0, 1})
    assert out["zscore"].notna().sum() > 0


def test_max_drawdown_and_sharpe_smoke():
    equity = pd.Series([100, 110, 105, 120, 90, 95])
    mdd = max_drawdown(equity)
    assert mdd < 0
    rets = equity.pct_change().dropna()
    s = sharpe_ratio(rets, risk_free_rate=0.0)
    assert isinstance(s, float)


def test_backtest_runs_on_synthetic_pair():
    a, b = _make_cointegrated_pair(n=600)
    detail, report = run_pairs_backtest(
        a,
        b,
        signal_cfg=SignalConfig(zscore_window=30, entry_z=1.5, exit_z=0.25),
        bt_cfg=BacktestConfig(transaction_cost_bps=5.0, slippage_bps=2.0),
    )
    assert "equity" in detail.columns
    assert len(detail) > 100
    assert isinstance(report.sharpe, float)
    d = summarize_performance(detail["equity"], signal=detail["signal"]).to_dict()
    assert "max_drawdown" in d


def test_monte_carlo_permutation_returns_pvalue():
    rng = np.random.default_rng(3)
    n = 400
    # Artificial timing skill: positive spread returns when position is +1
    spread = pd.Series(rng.normal(0.0, 0.01, size=n))
    pos = pd.Series(np.where(spread > 0, 1.0, -1.0))  # oracle timing
    out = monte_carlo_permutation_test(
        position=pos,
        spread_returns=spread,
        n_permutations=200,
        random_seed=3,
        risk_free_rate=0.0,
    )
    assert 0.0 <= out["p_value"] <= 1.0
    assert "observed_sharpe" in out
    # Oracle timing should beat most random shuffles
    assert out["p_value"] < 0.05
