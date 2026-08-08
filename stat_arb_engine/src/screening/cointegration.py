"""
Cointegration screening (Engle-Granger two-step procedure).

Math sketch (rising undergrad level)
------------------------------------
Two price series P_a,t and P_b,t are *cointegrated* if each is non-stationary
(typically I(1), like a random walk) but a linear combination

    e_t = P_a,t - β P_b,t

is stationary (I(0)).  That residual e_t is the *spread*.  Mean-reversion of
the spread is what pairs trading tries to monetize.

Engle–Granger steps
1. OLS hedge ratio:  P_a = α + β P_b + ε
2. ADF unit-root test on residuals ε̂.  Low p-value ⇒ reject "unit root"
   ⇒ evidence of cointegration.

Correlation is NOT the same thing: highly correlated I(1) series need not
cointegrate (they can wander apart permanently).  We still use correlation as
a cheap pre-filter so we do not run O(n²) ADF tests on hopeless pairs.
"""

from __future__ import annotations

import itertools
import logging
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from src.pipeline.ingest import load_price_panel
from src.utils.universe import tickers_from_config

logger = logging.getLogger(__name__)


@dataclass
class PairResult:
    pair_id: str
    ticker_a: str
    ticker_b: str
    hedge_ratio: float
    intercept: float
    cointegration_pvalue: float
    adf_stat: float
    correlation: float
    test_start: str
    test_end: str
    n_obs: int

    def to_row(self) -> dict[str, Any]:
        return asdict(self)


def pair_id(a: str, b: str) -> str:
    x, y = sorted([a.upper(), b.upper()])
    return f"{x}_{y}"


def ols_hedge_ratio(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """
    Fit y = α + β x by OLS (closed form).

    Returns (beta, alpha).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_design = np.column_stack([x, np.ones(len(x))])
    beta_hat, *_ = np.linalg.lstsq(x_design, y, rcond=None)
    return float(beta_hat[0]), float(beta_hat[1])


def engle_granger(
    y: pd.Series,
    x: pd.Series,
    significance: float = 0.05,
) -> tuple[bool, float, float, float, float]:
    """
    Run Engle-Granger on aligned series y, x.

    Returns
    -------
    is_coint, pvalue, adf_stat, hedge_ratio, intercept
    """
    aligned = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(aligned) < 50:
        return False, 1.0, np.nan, np.nan, np.nan

    y_arr = aligned.iloc[:, 0].to_numpy()
    x_arr = aligned.iloc[:, 1].to_numpy()
    beta, alpha = ols_hedge_ratio(y_arr, x_arr)
    resid = y_arr - (alpha + beta * x_arr)

    # statsmodels.coint implements Engle-Granger; we also keep ADF on residuals
    # for an interpretable test statistic on the resume / write-up.
    _tstat, pvalue, _crit = coint(y_arr, x_arr, trend="c", autolag="AIC")
    adf_stat, adf_p, *_ = adfuller(resid, maxlag=None, autolag="AIC")

    # Prefer the dedicated cointegration p-value; ADF on residuals is reported
    # as a diagnostic (they are closely related but not identical).
    is_coint = bool(pvalue < significance)
    return is_coint, float(pvalue), float(adf_stat), beta, alpha


def candidate_pairs(
    tickers: list[str],
    sectors: dict[str, str] | None = None,
    same_sector_only: bool = False,
) -> list[tuple[str, str]]:
    tickers = sorted({t.upper() for t in tickers})
    pairs = list(itertools.combinations(tickers, 2))
    if same_sector_only and sectors:
        pairs = [(a, b) for a, b in pairs if sectors.get(a) == sectors.get(b)]
    return pairs


def screen_pairs(
    prices: pd.DataFrame,
    pairs: Iterable[tuple[str, str]],
    significance: float = 0.05,
    min_correlation: float = 0.5,
    lookback_days: int | None = 756,
    min_obs: int = 252,
) -> list[PairResult]:
    """
    Screen an iterable of pairs on a (possibly sparse) price panel.

    Each pair is aligned independently (inner join of its two series) so that a
    short-history ticker cannot shrink the sample for unrelated pairs.
    """
    results: list[PairResult] = []
    pair_list = list(pairs)
    logger.info("Screening %d candidate pairs (pairwise date alignment)", len(pair_list))

    for i, (a, b) in enumerate(pair_list, start=1):
        if a not in prices.columns or b not in prices.columns:
            continue

        aligned = prices[[a, b]].dropna()
        if lookback_days is not None and len(aligned) > lookback_days:
            aligned = aligned.iloc[-lookback_days:]
        if len(aligned) < min_obs:
            continue

        ya = aligned[a]
        xb = aligned[b]
        corr = float(ya.corr(xb))
        if abs(corr) < min_correlation:
            continue

        # Canonical alphabetical order for reproducible pair_id / hedge ratio.
        a_c, b_c = sorted([a, b])
        is_coint, pval, adf_stat, beta, alpha = engle_granger(
            aligned[a_c], aligned[b_c], significance
        )
        if not is_coint:
            continue

        corr = float(aligned[a_c].corr(aligned[b_c]))
        results.append(
            PairResult(
                pair_id=pair_id(a_c, b_c),
                ticker_a=a_c,
                ticker_b=b_c,
                hedge_ratio=beta,
                intercept=alpha,
                cointegration_pvalue=pval,
                adf_stat=adf_stat,
                correlation=corr,
                test_start=str(aligned.index.min().date()),
                test_end=str(aligned.index.max().date()),
                n_obs=int(len(aligned)),
            )
        )
        if i % 250 == 0:
            logger.info(
                "  progress: %d/%d pairs checked, %d kept", i, len(pair_list), len(results)
            )

    results.sort(key=lambda r: r.cointegration_pvalue)
    logger.info("Screen finished: %d cointegrated pairs", len(results))
    return results


def save_pairs(db_path: Path, results: list[PairResult]) -> None:
    if not results:
        return
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    rows = []
    for r in results:
        rows.append(
            (
                r.pair_id,
                r.ticker_a,
                r.ticker_b,
                r.hedge_ratio,
                r.cointegration_pvalue,
                r.adf_stat,
                r.correlation,
                r.test_start,
                r.test_end,
                r.n_obs,
                now,
            )
        )
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO pairs (
                pair_id, ticker_a, ticker_b, hedge_ratio, cointegration_pvalue,
                adf_stat, correlation, test_start, test_end, n_obs, screened_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_pairs(db_path: Path, max_pairs: int | None = None) -> pd.DataFrame:
    query = "SELECT * FROM pairs ORDER BY cointegration_pvalue ASC"
    if max_pairs is not None:
        query += f" LIMIT {int(max_pairs)}"
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def run_screen(cfg: dict[str, Any]) -> pd.DataFrame:
    """End-to-end screen using config + cached prices."""
    from src.pipeline.ingest import list_cached_tickers

    data_cfg = cfg["data"]
    screen_cfg = cfg["screening"]
    sectors = tickers_from_config(cfg)
    cache_dir = Path(data_cfg["cache_dir"])

    # Only screen names that survived ingest / exist on disk.
    cached = set(list_cached_tickers(cache_dir))
    available = sorted(t for t in sectors if t in cached)
    # Also require min history rows in cache (pairwise screen still needs enough obs)
    min_hist = int(data_cfg.get("min_history_days", 756))
    kept: list[str] = []
    for t in available:
        n = len(pd.read_parquet(cache_dir / f"{t}.parquet"))
        if n >= min_hist:
            kept.append(t)
    available = kept

    prices = load_price_panel(available, cache_dir, join="outer")
    pairs = candidate_pairs(
        available,
        sectors=sectors,
        same_sector_only=bool(screen_cfg.get("same_sector_only", False)),
    )
    results = screen_pairs(
        prices,
        pairs,
        significance=float(screen_cfg.get("significance_level", 0.05)),
        min_correlation=float(screen_cfg.get("min_correlation", 0.5)),
        lookback_days=screen_cfg.get("lookback_days", 756),
        min_obs=int(screen_cfg.get("lookback_days", 756)) // 2,
    )
    max_pairs = screen_cfg.get("max_pairs")
    if max_pairs is not None:
        results = results[: int(max_pairs)]

    db_path = Path(data_cfg["db_path"])
    save_pairs(db_path, results)
    return pd.DataFrame([r.to_row() for r in results])
