"""Leakage and feature construction tests. No network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features import BASELINE_SCALE, TARGET, build_ticker_features, forward_rv, trailing_rv
from leakage import run_leakage_suite
from walkforward import iter_windows


def test_leakage_suite():
    run_leakage_suite()


def test_forward_rv_excludes_today():
    sq = pd.Series([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 49.0], dtype=float)
    got = forward_rv(sq, horizon=5)
    # At index 0: sqrt(4+9+16+25+36) = sqrt(90)
    assert got.iloc[0] == pytest.approx(np.sqrt(90.0))
    assert np.isnan(got.iloc[-1])


def test_trailing_rv_includes_today():
    sq = pd.Series([1.0, 4.0, 9.0, 16.0], dtype=float)
    got = trailing_rv(sq, 2)
    assert got.iloc[1] == pytest.approx(np.sqrt(5.0))
    assert np.isnan(got.iloc[0])


def test_target_starts_at_t_plus_one():
    n = 60
    rng = np.random.default_rng(0)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    dates = pd.bdate_range("2016-01-04", periods=n)
    df = pd.DataFrame(
        {
            "ticker": "X",
            "date": dates,
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.integers(1_000_000, 3_000_000, n).astype(float),
        }
    )
    feat = build_ticker_features(df)
    assert len(feat) > 10
    r = np.log(close).diff()
    t = feat["date"].iloc[10]
    loc = int(dates.get_loc(t))
    expected = float(np.sqrt((r.iloc[loc + 1 : loc + 6] ** 2).sum()))
    assert feat.loc[feat["date"] == t, TARGET].iloc[0] == pytest.approx(expected)


def test_baseline_is_scaled_rv22():
    n = 60
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(n, 0.001))))
    df = pd.DataFrame(
        {
            "ticker": "X",
            "date": pd.bdate_range("2016-01-04", periods=n),
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000.0,
        }
    )
    feat = build_ticker_features(df)
    assert np.allclose(feat["baseline"], feat["rv_22d"] * BASELINE_SCALE)


def test_purge_leaves_five_sessions_between_train_labels_and_test_features():
    dates = pd.bdate_range("2015-01-02", periods=1000)
    windows = list(iter_windows(dates, train_days=756, test_days=126, step_days=126, purge_days=5))
    assert windows
    w0 = windows[0]
    train = w0["train_dates"]
    test = w0["test_dates"]
    full_train_end = dates[755]
    assert train[-1] == dates[750]
    assert test[0] == dates[756]
    assert full_train_end < test[0]
    # Last train label uses the next 5 sessions: dates[751]..dates[755], none of which are test features.
    label_dates = set(dates[751:756])
    assert label_dates.isdisjoint(set(test))
    assert len(windows) >= 1
    # OOS blocks do not overlap
    for a, b in zip(windows, windows[1:]):
        assert a["test_dates"][-1] < b["test_dates"][0]
