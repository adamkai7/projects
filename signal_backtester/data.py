"""Thin data-loading wrapper around yfinance with a local CSV cache, so
repeated runs don't re-hit the network."""

from __future__ import annotations

import os

import pandas as pd

CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")


def load_prices(ticker: str, start: str = "2015-01-01", end: str | None = None) -> pd.DataFrame:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}_{start}_{end or 'latest'}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.to_csv(cache_path)
    return df
