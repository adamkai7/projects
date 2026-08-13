"""
Signal generators matching the *style* of common ThinkScript studies
(z-score mean-reversion, regression-based trend fitting, momentum).

Every function takes a price Series (Close, indexed by date) and returns
a position Series aligned to the same index:
    +1 = long, 0 = flat, -1 = short (only for strategies that support shorting)

These are intentionally simple, readable reference implementations, not
production strategies. The point is to give you a real backtest harness you
can drop your own exact entry/exit rules into (see `custom_signal_template`
at the bottom) and get real metrics instead of an eyeball/educated guess.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def zscore_mean_reversion(
    close: pd.Series,
    window: int = 20,
    entry_z: float = 1.5,
    exit_z: float = 0.3,
    long_only: bool = True,
) -> pd.Series:
    """Rolling z-score mean-reversion signal.

    Mirrors the logic behind studies like MeanRev1_L / ZScorePopDrop /
    ZScore_LSTRATEGY: compute how far price has strayed from its own rolling
    mean (in units of rolling std), and fade the move.

    Enter long when z <= -entry_z (price unusually low vs. its own recent
    mean). Exit when z rises back above -exit_z. Symmetric short side is
    optional (long_only=True reproduces the common "_L" retail variant).
    """
    roll_mean = close.rolling(window).mean()
    roll_std = close.rolling(window).std()
    z = (close - roll_mean) / roll_std

    in_long = False
    in_short = False

    z_vals = z.to_numpy()
    pos_vals = np.zeros(len(z_vals), dtype=float)

    for i in range(len(z_vals)):
        zi = z_vals[i]
        if np.isnan(zi):
            pos_vals[i] = 0
            continue

        if in_long:
            if zi >= -exit_z:
                in_long = False
            else:
                pos_vals[i] = 1
                continue

        if in_short:
            if zi <= exit_z:
                in_short = False
            else:
                pos_vals[i] = -1
                continue

        if zi <= -entry_z:
            in_long = True
            pos_vals[i] = 1
        elif (not long_only) and zi >= entry_z:
            in_short = True
            pos_vals[i] = -1
        else:
            pos_vals[i] = 0

    return pd.Series(pos_vals, index=close.index)


def regression_trend_momentum(
    close: pd.Series,
    window: int = 20,
    slope_threshold: float = 0.0,
) -> pd.Series:
    """Rolling linear-regression slope trend filter.

    Mirrors "regression-based trend fitting" studies (e.g. ModR2TrendSTRATEGY,
    SwingRegressionSTUDY): fit a simple OLS line to the last `window` closes
    (in log space so slope is a growth rate), go long while the fitted slope
    is positive (above `slope_threshold`), flat otherwise.
    """
    log_close = np.log(close)
    x = np.arange(window)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def slope(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        return ((x - x_mean) * (y - y_mean)).sum() / x_var

    slopes = log_close.rolling(window).apply(slope, raw=True)
    position = (slopes > slope_threshold).astype(float)
    position[slopes.isna()] = 0
    return position


def nday_momentum(
    close: pd.Series,
    lookback: int = 10,
    threshold: float = 0.0,
) -> pd.Series:
    """Simple N-day return momentum (matches MultiReturnTrend / ReturnMA style
    studies): go long if the trailing `lookback`-day return exceeds
    `threshold`.
    """
    ret = close.pct_change(lookback)
    position = (ret > threshold).astype(float)
    position[ret.isna()] = 0
    return position


def custom_signal_template(close: pd.Series) -> pd.Series:
    """Copy this function and translate your exact ThinkScript entry/exit
    logic here. `close` is a pandas Series of daily closes indexed by date.
    Return a Series of the same index with values in {-1, 0, 1}.

    Example skeleton:

        indicator = close.rolling(14).mean()          # replace with your calc
        long_condition = close > indicator             # your entry rule
        exit_condition = close < indicator              # your exit rule

        position = pd.Series(0.0, index=close.index)
        in_pos = False
        for i in range(len(close)):
            if in_pos and exit_condition.iloc[i]:
                in_pos = False
            elif not in_pos and long_condition.iloc[i]:
                in_pos = True
            position.iloc[i] = 1.0 if in_pos else 0.0
        return position
    """
    raise NotImplementedError("Translate your ThinkScript rule here.")
