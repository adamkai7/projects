"""
Z-score mean-reversion signals on a cointegrated spread.

Spread definition
-----------------
Given hedge ratio β from Engle-Granger (P_a ≈ α + β P_b):

    S_t = P_a,t - β P_b,t

(We often omit α in the traded spread; it shifts the level but not differences.
Here we keep the residual form for consistency with the cointegration regression.)

Z-score
-------
    z_t = (S_t - μ_t) / σ_t

where μ_t, σ_t are rolling mean/std over a window W.  Large |z| means the spread
is unusually far from its recent mean — the classic mean-reversion entry cue.

Signal encoding
---------------
    +1 : long the spread  (long A, short β units of B)  when z < -entry
    -1 : short the spread (short A, long β units of B)  when z > +entry
     0 : flat

Exits: |z| < exit_z, or |z| > stop_z (divergence stop), handled in backtest
via position state machine on top of these raw triggers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.screening.cointegration import ols_hedge_ratio


@dataclass
class SignalConfig:
    zscore_window: int = 60
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 4.0


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | None = None,
    intercept: float = 0.0,
) -> tuple[pd.Series, float, float]:
    """
    Align prices and form spread S = A - β B - α.

    If hedge_ratio is None, estimate β, α by OLS on the full sample.
    """
    aligned = pd.concat([price_a, price_b], axis=1, join="inner").dropna()
    aligned.columns = ["a", "b"]
    if hedge_ratio is None:
        hedge_ratio, intercept = ols_hedge_ratio(
            aligned["a"].to_numpy(), aligned["b"].to_numpy()
        )
    spread = aligned["a"] - float(hedge_ratio) * aligned["b"] - float(intercept)
    spread.name = "spread"
    return spread, float(hedge_ratio), float(intercept)


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    mu = spread.rolling(window=window, min_periods=window).mean()
    sigma = spread.rolling(window=window, min_periods=window).std(ddof=1)
    z = (spread - mu) / sigma
    # Guard against zero variance windows.
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "zscore"
    return z


def generate_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float | None = None,
    intercept: float = 0.0,
    cfg: SignalConfig | None = None,
) -> pd.DataFrame:
    """
    Build a frame with spread, zscore, and a discrete signal in {-1, 0, +1}.

    The signal here is the *desired position* under a simple state machine:
    - Enter long spread when z crosses below -entry
    - Enter short spread when z crosses above +entry
    - Exit to flat when |z| falls below exit, or when |z| exceeds stop
    """
    cfg = cfg or SignalConfig()
    spread, beta, alpha = compute_spread(price_a, price_b, hedge_ratio, intercept)
    z = rolling_zscore(spread, cfg.zscore_window)

    position = np.zeros(len(z), dtype=int)
    current = 0
    z_arr = z.to_numpy()

    for i in range(len(z_arr)):
        zi = z_arr[i]
        if np.isnan(zi):
            position[i] = 0
            current = 0
            continue

        if current == 0:
            if zi <= -cfg.entry_z:
                current = 1
            elif zi >= cfg.entry_z:
                current = -1
        elif current == 1:
            # Long spread: exit on mean reversion or adverse blowout (more negative)
            if abs(zi) <= cfg.exit_z or zi <= -cfg.stop_z:
                current = 0
        elif current == -1:
            if abs(zi) <= cfg.exit_z or zi >= cfg.stop_z:
                current = 0

        position[i] = current

    out = pd.DataFrame(
        {
            "spread": spread.to_numpy(),
            "zscore": z_arr,
            "signal": position,
            "hedge_ratio": beta,
            "intercept": alpha,
        },
        index=spread.index,
    )
    return out
