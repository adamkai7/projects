"""
Leakage guards.

These checks are the point of the project. If a feature at t moves when you
scramble prices after t, the study is invalid.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features import ALL_FEATURES, TARGET, build_ticker_features


def _toy_ohlcv(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0005, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[close[0]], close[:-1] * np.exp(rng.normal(0, 0.004, size=n - 1))])
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0, 0.01, size=n))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0, 0.01, size=n))
    volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)
    dates = pd.bdate_range("2015-01-02", periods=n)
    return pd.DataFrame(
        {
            "ticker": "TOY",
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def features_at_t_ignore_future(atol: float = 1e-12) -> None:
    """Scramble close/high/low/volume after t; features at t must not change."""
    base = _toy_ohlcv()
    feat = build_ticker_features(base)
    t_idx = 40
    t = feat.loc[t_idx, "date"]
    scrambled = base.copy()
    future = scrambled["date"] > t
    scrambled.loc[future, ["open", "high", "low", "close", "volume"]] = np.random.default_rng(99).uniform(
        50, 150, size=(int(future.sum()), 5)
    )
    feat2 = build_ticker_features(scrambled)
    row1 = feat.loc[feat["date"] == t, ALL_FEATURES].iloc[0]
    row2 = feat2.loc[feat2["date"] == t, ALL_FEATURES].iloc[0]
    if not np.allclose(row1.to_numpy(dtype=float), row2.to_numpy(dtype=float), atol=atol, equal_nan=True):
        delta = (row1 - row2).abs()
        raise AssertionError(f"Feature leakage at {t.date()}: {delta[delta > atol].to_dict()}")


def target_uses_next_five_only(atol: float = 1e-12) -> None:
    """Target at t must change if r_{t+1..t+5} change, and not if prices after t+5 change."""
    base = _toy_ohlcv()
    feat = build_ticker_features(base)
    t = feat.loc[40, "date"]
    y0 = float(feat.loc[feat["date"] == t, TARGET].iloc[0])

    after = base.copy()
    mask = after["date"] > t + pd.tseries.offsets.BDay(5)
    after.loc[mask, "close"] *= 3.0
    y_after = float(build_ticker_features(after).loc[lambda d: d["date"] == t, TARGET].iloc[0])
    if abs(y0 - y_after) > atol:
        raise AssertionError("Target at t moved when prices after t+5 were scrambled")

    inside = base.copy()
    # Next session after t.
    nxt = inside.loc[inside["date"] > t, "date"].min()
    inside.loc[inside["date"] == nxt, "close"] *= 1.15
    y_inside = float(build_ticker_features(inside).loc[lambda d: d["date"] == t, TARGET].iloc[0])
    if abs(y0 - y_inside) <= atol:
        raise AssertionError("Target at t did not move when r_{t+1} changed")


def rv1d_is_abs_return_today() -> None:
    """rv_1d at t is |r_t|, not |r_{t+1}|."""
    base = _toy_ohlcv()
    feat = build_ticker_features(base)
    aligned = feat.copy()
    if not np.allclose(aligned["rv_1d"], aligned["r"].abs(), atol=1e-12):
        raise AssertionError("rv_1d is not |r_t|")


def run_leakage_suite() -> None:
    features_at_t_ignore_future()
    target_uses_next_five_only()
    rv1d_is_abs_return_today()
