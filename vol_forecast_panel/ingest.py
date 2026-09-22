"""Download daily OHLCV, cache to Parquet, drop gappy names. Never interpolate prices."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

PRICE_COLS = ["open", "high", "low", "close", "volume"]


def _parquet_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.upper()}.parquet"


def _normalize_ohlcv(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["ticker", "date", *PRICE_COLS])

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance may return (field, ticker) or (ticker, field).
        level0 = [str(c).lower() for c in df.columns.get_level_values(0)]
        if any(name in level0 for name in ("open", "close", "high", "low")):
            df.columns = [str(c[0]) for c in df.columns]
        else:
            df = df.xs(ticker, axis=1, level=0) if ticker in df.columns.get_level_values(0) else df
            df.columns = [str(c) if not isinstance(c, tuple) else str(c[0]) for c in df.columns]

    df = df.reset_index()
    rename = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    df = df.rename(columns=rename)
    if "date" not in df.columns:
        for candidate in ("index", "datetime"):
            if candidate in df.columns:
                df = df.rename(columns={candidate: "date"})
                break
    if "date" not in df.columns:
        raise ValueError(f"No date column for {ticker}: {list(df.columns)}")

    keep = [c for c in PRICE_COLS if c in df.columns]
    if "close" not in keep:
        raise ValueError(f"No close column for {ticker}: {list(df.columns)}")
    out = df[["date", *keep]].copy()
    out["ticker"] = ticker.upper()
    out["date"] = pd.to_datetime(out["date"]).dt.tz_localize(None)
    out = out.dropna(subset=["close"])
    out = out.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    for col in PRICE_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[["ticker", "date", *PRICE_COLS]].reset_index(drop=True)


def download_ticker(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    return _normalize_ohlcv(raw, ticker)


def load_or_download(
    ticker: str,
    start: str,
    end: str | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _parquet_path(cache_dir, ticker)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    df = download_ticker(ticker, start=start, end=end)
    if not df.empty:
        df.to_parquet(path, index=False)
        logger.info("Cached %s (%d rows)", ticker, len(df))
    return df


def _max_trading_gap(present: pd.Series) -> int:
    """Longest run of False in a boolean calendar-aligned series."""
    if present.all():
        return 0
    longest = 0
    run = 0
    for ok in present.tolist():
        if ok:
            run = 0
        else:
            run += 1
            longest = max(longest, run)
    return longest


def reference_calendar(frames: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    """Dates present in at least half the tickers — a common US session calendar."""
    from collections import Counter

    counts: Counter = Counter()
    for df in frames.values():
        counts.update(pd.to_datetime(df["date"]).dt.normalize())
    if not counts:
        return pd.DatetimeIndex([])
    thresh = 0.5 * len(frames)
    dates = sorted(d for d, n in counts.items() if n >= thresh)
    return pd.DatetimeIndex(dates)


def drop_gappy(
    frames: dict[str, pd.DataFrame],
    min_coverage: float,
    max_gap_trading_days: int,
    min_rows: int,
    listed_by: str | None = None,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """
    Drop names with large holes. Prices are never interpolated — a name either
    stays with its native trading days or it is removed from the universe.

    Gaps are measured only after the first observed print, so a 2012 IPO is not
    punished for missing January 2012. `listed_by` (study start) still requires
    that first print to fall on or before the freeze date.
    """
    calendar = reference_calendar(frames)
    listed_ts = pd.Timestamp(listed_by) if listed_by else None
    kept: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    for ticker, df in frames.items():
        if len(df) < min_rows:
            logger.warning("Skip %s: %d rows < min_rows=%d", ticker, len(df), min_rows)
            skipped.append(ticker)
            continue
        dates = pd.DatetimeIndex(pd.to_datetime(df["date"]).dt.normalize().unique()).sort_values()
        if listed_ts is not None and dates.min() > listed_ts:
            logger.warning("Skip %s: first print %s after freeze %s", ticker, dates.min().date(), listed_ts.date())
            skipped.append(ticker)
            continue
        present = pd.Series(calendar.isin(dates), index=calendar)
        observed = present[present]
        if observed.empty:
            skipped.append(ticker)
            continue
        span = present.loc[observed.index[0] : observed.index[-1]]
        coverage = float(span.mean()) if len(span) else 0.0
        gap = _max_trading_gap(span)
        if coverage < min_coverage or gap > max_gap_trading_days:
            logger.warning(
                "Skip %s: coverage=%.2f gap=%d (need coverage>=%.2f, gap<=%d)",
                ticker,
                coverage,
                gap,
                min_coverage,
                max_gap_trading_days,
            )
            skipped.append(ticker)
            continue
        kept[ticker] = df
    return kept, skipped


def download_many(tickers: list[str], start: str, end: str | None) -> dict[str, pd.DataFrame]:
    """Batch download. Falls back to one-at-a-time if the combined frame is unusable."""
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if not tickers:
        return {}
    if len(tickers) == 1:
        df = download_ticker(tickers[0], start, end)
        return {tickers[0]: df} if not df.empty else {}

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out

    if isinstance(raw.columns, pd.MultiIndex):
        level0 = set(str(c) for c in raw.columns.get_level_values(0))
        for ticker in tickers:
            key = ticker if ticker in level0 else ticker.upper() if ticker.upper() in level0 else None
            if key is None:
                continue
            try:
                df = _normalize_ohlcv(raw[key], ticker)
            except Exception:
                logger.exception("Failed to parse %s from batch download", ticker)
                continue
            if not df.empty:
                out[ticker] = df
        return out

    # Single-ticker shaped frame even though we asked for many.
    if len(tickers) == 1:
        df = _normalize_ohlcv(raw, tickers[0])
        if not df.empty:
            out[tickers[0]] = df
    return out


def ingest_universe(cfg: dict[str, Any], force: bool = False, limit: int = 0) -> dict[str, pd.DataFrame]:
    from config import load_universe

    data_cfg = cfg["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    tickers = load_universe(data_cfg["universe_path"], limit=limit)

    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for ticker in tickers:
        path = _parquet_path(cache_dir, ticker)
        if path.exists() and not force:
            df = pd.read_parquet(path)
            df["date"] = pd.to_datetime(df["date"])
            frames[ticker] = df
        else:
            missing.append(ticker)

    chunk_size = 40
    for i in range(0, len(missing), chunk_size):
        chunk = missing[i : i + chunk_size]
        logger.info("Downloading %d tickers (%d–%d of %d missing)", len(chunk), i + 1, i + len(chunk), len(missing))
        batch = download_many(chunk, start, end)
        still = [t for t in chunk if t not in batch or batch[t].empty]
        for ticker, df in batch.items():
            if df.empty:
                continue
            df.to_parquet(_parquet_path(cache_dir, ticker), index=False)
            frames[ticker] = df
        for ticker in still:
            logger.info("Retry %s individually", ticker)
            try:
                df = download_ticker(ticker, start, end)
            except Exception:
                logger.exception("Failed %s", ticker)
                continue
            if df.empty:
                continue
            df.to_parquet(_parquet_path(cache_dir, ticker), index=False)
            frames[ticker] = df

    kept, gappy = drop_gappy(
        frames,
        min_coverage=float(data_cfg.get("min_coverage", 0.90)),
        max_gap_trading_days=int(data_cfg.get("max_gap_trading_days", 5)),
        min_rows=int(data_cfg.get("min_rows", 1000)),
        listed_by=data_cfg.get("study_start"),
    )
    failed = [t for t in tickers if t not in frames]
    logger.info("Ingest: %d kept, %d gappy, %d failed", len(kept), len(gappy), len(failed))
    return kept
