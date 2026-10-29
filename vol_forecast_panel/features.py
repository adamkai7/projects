"""
Target and features known at date-t close.

Target
------
RV_{t, t+5} = sqrt(sum_{i=1..5} r_{t+i}^2)
where r is the close-to-close log return. The label starts at t+1, so close-to-close
RV_1d computed at t (which uses r_t) is a valid feature.

Features (all known at t)
-------------------------
- rv_1d, rv_5d, rv_22d  HAR-style trailing realized vol, including r_t
- volume_z              (volume_t - 22d mean) / 22d std
- log_range             log(high_t / low_t)
- overnight_gap         log(open_t / close_{t-1})
- month_end             1 if any of the next 5 trading days is a month-end (calendar)

No interpolation. Rows with missing inputs are dropped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HAR_FEATURES = ["rv_1d", "rv_5d", "rv_22d"]
ALL_FEATURES = HAR_FEATURES + ["volume_z", "log_range", "overnight_gap", "month_end"]
TARGET = "rv_fwd_5d"
BASELINE_SCALE = float(np.sqrt(5.0 / 22.0))


def _log_returns(close: pd.Series) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    return np.log(close).diff()


def trailing_rv(sq_ret: pd.Series, window: int) -> pd.Series:
    """sqrt(sum of last `window` squared log returns, including today)."""
    return sq_ret.rolling(window, min_periods=window).sum().pow(0.5)


def forward_rv(sq_ret: pd.Series, horizon: int = 5) -> pd.Series:
    """sqrt(sum of the *next* `horizon` squared log returns), excluding today."""
    parts = [sq_ret.shift(-i) for i in range(1, horizon + 1)]
    return pd.concat(parts, axis=1).sum(axis=1, min_count=horizon).pow(0.5)


def month_end_in_next_n(dates: pd.Series, n: int = 5) -> pd.Series:
    """Calendar dummy: 1 if any of the next n sessions is that month's last trading day."""
    d = pd.to_datetime(dates)
    s = pd.Series(d.to_numpy(), index=d.index)
    last = s.groupby([s.dt.year, s.dt.month]).transform("max")
    is_month_end = s.eq(last).astype(int)
    future = sum(is_month_end.shift(-i).fillna(0) for i in range(1, n + 1))
    return (future > 0).astype(int)


def build_ticker_features(df: pd.DataFrame, horizon: int = 5, volume_z_window: int = 22) -> pd.DataFrame:
    """One ticker's OHLCV -> one row per date with features and target."""
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    close = pd.to_numeric(out["close"], errors="coerce")
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    opn = pd.to_numeric(out["open"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")

    r = _log_returns(close)
    sq = r.pow(2)

    out["r"] = r
    out["rv_1d"] = trailing_rv(sq, 1)
    out["rv_5d"] = trailing_rv(sq, 5)
    out["rv_22d"] = trailing_rv(sq, 22)
    out[TARGET] = forward_rv(sq, horizon=horizon)
    out["baseline"] = out["rv_22d"] * BASELINE_SCALE

    vol_mean = volume.rolling(volume_z_window, min_periods=volume_z_window).mean()
    vol_std = volume.rolling(volume_z_window, min_periods=volume_z_window).std(ddof=0)
    z = np.where(vol_std > 0, (volume - vol_mean) / vol_std, 0.0)
    out["volume_z"] = np.where(vol_mean.notna(), z, np.nan)

    safe_hl = (high > 0) & (low > 0) & (high >= low)
    out["log_range"] = np.where(safe_hl, np.log(high / low), np.nan)
    prev_close = close.shift(1)
    out["overnight_gap"] = np.where((opn > 0) & (prev_close > 0), np.log(opn / prev_close), np.nan)
    out["month_end"] = month_end_in_next_n(out["date"], n=horizon)

    keep = ["ticker", "date", TARGET, "baseline", "r", *ALL_FEATURES]
    out = out[keep]
    return out.dropna(subset=[TARGET, "baseline", *ALL_FEATURES]).reset_index(drop=True)


def build_panel(frames: dict[str, pd.DataFrame], horizon: int = 5, volume_z_window: int = 22) -> pd.DataFrame:
    pieces = [
        build_ticker_features(df, horizon=horizon, volume_z_window=volume_z_window)
        for df in frames.values()
    ]
    if not pieces:
        raise ValueError("No tickers to build features from")
    panel = pd.concat(pieces, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"])
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)
