"""
Data ingestion: download OHLCV via yfinance, clean, cache to Parquet, index in SQLite.

Design notes (interview-friendly):
- Cache to Parquet so repeated research runs do not re-hit the API.
- Store a thin SQLite index for SQL queries (JD skill) without needing a server DB.
- Align calendars across tickers by inner-joining on trading dates later, not here.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

from src.utils.config import ensure_dir
from src.utils.universe import tickers_from_config

logger = logging.getLogger(__name__)

PRICE_SCHEMA = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]


def _parquet_path(cache_dir: Path, ticker: str) -> Path:
    return cache_dir / f"{ticker.upper()}.parquet"


def download_ticker(
    ticker: str,
    start: str,
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV for one ticker and normalize column names."""
    end = end or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if raw.empty:
        logger.warning("No data returned for %s", ticker)
        return pd.DataFrame(columns=PRICE_SCHEMA)

    # yfinance sometimes returns MultiIndex columns even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    df = raw.reset_index().rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    df["ticker"] = ticker.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df[PRICE_SCHEMA].dropna(subset=["close", "adj_close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return df.reset_index(drop=True)


def load_or_download(
    ticker: str,
    start: str,
    end: str | None,
    cache_dir: Path,
    force: bool = False,
) -> pd.DataFrame:
    """Return cached Parquet if present (unless force), else download and cache."""
    ensure_dir(cache_dir)
    path = _parquet_path(cache_dir, ticker)
    if path.exists() and not force:
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        logger.debug("Cache hit: %s (%d rows)", ticker, len(df))
        return df

    df = download_ticker(ticker, start=start, end=end)
    if not df.empty:
        df.to_parquet(path, index=False)
        logger.info("Cached %s -> %s (%d rows)", ticker, path.name, len(df))
    return df


def init_db(db_path: Path) -> None:
    """Create SQLite schema if missing."""
    ensure_dir(db_path.parent)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prices (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );

            CREATE TABLE IF NOT EXISTS pairs (
                pair_id TEXT PRIMARY KEY,
                ticker_a TEXT NOT NULL,
                ticker_b TEXT NOT NULL,
                hedge_ratio REAL,
                cointegration_pvalue REAL,
                adf_stat REAL,
                correlation REAL,
                test_start TEXT,
                test_end TEXT,
                n_obs INTEGER,
                screened_at TEXT
            );

            CREATE TABLE IF NOT EXISTS signals (
                pair_id TEXT NOT NULL,
                date TEXT NOT NULL,
                spread REAL,
                zscore REAL,
                signal INTEGER,
                PRIMARY KEY (pair_id, date)
            );

            CREATE TABLE IF NOT EXISTS backtest_results (
                run_id TEXT PRIMARY KEY,
                pair_id TEXT,
                sharpe REAL,
                sortino REAL,
                max_drawdown REAL,
                total_return REAL,
                calmar REAL,
                win_rate REAL,
                n_trades INTEGER,
                params TEXT,
                created_at TEXT
            );
            """
        )


def upsert_prices(db_path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    with sqlite3.connect(db_path) as conn:
        out.to_sql("prices_tmp", conn, if_exists="replace", index=False)
        conn.execute(
            """
            INSERT OR REPLACE INTO prices
            SELECT ticker, date, open, high, low, close, adj_close, volume
            FROM prices_tmp
            """
        )
        conn.execute("DROP TABLE prices_tmp")


def ingest_universe(cfg: dict[str, Any], force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Download/cache the configured universe and optionally sync into SQLite.

    Returns a dict ticker -> cleaned OHLCV frame.
    """
    data_cfg = cfg["data"]
    cache_dir = Path(data_cfg["cache_dir"])
    db_path = Path(data_cfg["db_path"])
    start = data_cfg["start_date"]
    end = data_cfg.get("end_date")
    min_history = int(data_cfg.get("min_history_days", 1000))

    tickers = tickers_from_config(cfg)
    init_db(db_path)

    frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []

    for i, ticker in enumerate(sorted(tickers), start=1):
        logger.info("[%d/%d] Ingesting %s", i, len(tickers), ticker)
        try:
            df = load_or_download(ticker, start, end, cache_dir, force=force)
        except Exception as exc:  # network / parse failures should not kill the batch
            logger.exception("Failed to ingest %s: %s", ticker, exc)
            skipped.append(ticker)
            continue

        if len(df) < min_history:
            logger.warning(
                "Skipping %s: only %d rows (need >= %d)", ticker, len(df), min_history
            )
            skipped.append(ticker)
            continue

        upsert_prices(db_path, df)
        frames[ticker] = df

    logger.info(
        "Ingest complete: %d kept, %d skipped. Cache=%s DB=%s",
        len(frames),
        len(skipped),
        cache_dir,
        db_path,
    )
    if skipped:
        logger.info("Skipped tickers: %s", ", ".join(skipped))
    return frames


def load_price_panel(
    tickers: list[str],
    cache_dir: Path,
    price_col: str = "adj_close",
    join: str = "outer",
) -> pd.DataFrame:
    """
    Build a date x ticker panel of prices from Parquet cache.

    Default join is outer so short-history names do not destroy the panel for
    everyone else. Pairwise tests should `.dropna()` on the two columns they use.
    Use join='inner' when you truly need a common calendar (e.g. a 2-ticker backtest).
    """
    series: list[pd.Series] = []
    for ticker in tickers:
        path = _parquet_path(cache_dir, ticker)
        if not path.exists():
            raise FileNotFoundError(f"Missing cache for {ticker}: {path}")
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        s = df.set_index("date")[price_col].rename(ticker.upper())
        series.append(s)

    panel = pd.concat(series, axis=1, join=join).sort_index()
    return panel


def list_cached_tickers(cache_dir: Path) -> list[str]:
    """Return tickers that have a Parquet cache file."""
    return sorted(p.stem.upper() for p in Path(cache_dir).glob("*.parquet"))
