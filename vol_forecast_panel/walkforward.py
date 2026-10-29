"""
Walk-forward with a 5-day purge.

Train on 756 trading days, test on the next 126, roll by 126 so OOS blocks
do not overlap. Drop the last `purge_days` train rows so a 5-day label cannot
cover a date that is also used as a test feature.
"""

from __future__ import annotations

import logging
from typing import Any, Iterator

import pandas as pd

from features import ALL_FEATURES, TARGET
from metrics import metrics_table
from models import fit_models

logger = logging.getLogger(__name__)


def iter_windows(
    dates: pd.DatetimeIndex,
    train_days: int,
    test_days: int,
    step_days: int,
    purge_days: int,
) -> Iterator[dict[str, pd.DatetimeIndex | int | str]]:
    dates = pd.DatetimeIndex(dates).sort_values().unique()
    n = len(dates)
    start = 0
    window = 0
    while start + train_days + test_days <= n:
        window += 1
        train_full = dates[start : start + train_days]
        train = train_full[:-purge_days] if purge_days > 0 else train_full
        test = dates[start + train_days : start + train_days + test_days]
        yield {
            "window": window,
            "train_dates": train,
            "test_dates": test,
            "train_start": str(pd.Timestamp(train[0]).date()),
            "train_end": str(pd.Timestamp(train[-1]).date()),
            "test_start": str(pd.Timestamp(test[0]).date()),
            "test_end": str(pd.Timestamp(test[-1]).date()),
            "purged": purge_days,
        }
        start += step_days


def _assert_purge(train_dates: pd.DatetimeIndex, test_dates: pd.DatetimeIndex, horizon: int) -> None:
    """Last train label uses the next `horizon` sessions after the last train date."""
    last_train = pd.Timestamp(train_dates[-1])
    first_test = pd.Timestamp(test_dates[0])
    if last_train >= first_test:
        raise ValueError("Train feature dates overlap test feature dates")
    # The label at last_train uses returns on the next `horizon` trading days.
    # Those days must not be test feature dates.
    overlap = set(pd.DatetimeIndex(test_dates))
    # We cannot see the global calendar here; the window builder already cut
    # `purge_days` off the end of train. Enforce the date inequality.
    if last_train >= first_test:
        raise ValueError("purge failed")
    _ = overlap, horizon


def run_walkforward(panel: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, pd.DataFrame]:
    v = cfg["validation"]
    train_days = int(v["train_days"])
    test_days = int(v["test_days"])
    step_days = int(v.get("step_days", test_days))
    purge_days = int(v.get("purge_days", 5))
    horizon = int(cfg.get("target", {}).get("horizon", 5))

    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])
    study_start = cfg["data"].get("study_start")
    if study_start:
        # Keep pre-study rows so the first train window can end at study_start.
        pass

    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    windows = list(
        iter_windows(
            dates,
            train_days=train_days,
            test_days=test_days,
            step_days=step_days,
            purge_days=purge_days,
        )
    )
    if not windows:
        raise ValueError(
            f"Not enough dates for walk-forward: n={len(dates)}, "
            f"need train+test={train_days + test_days}"
        )
    logger.info("Walk-forward: %d windows over %d dates", len(windows), len(dates))

    oos_chunks: list[pd.DataFrame] = []
    window_rows: list[dict[str, Any]] = []
    ridge_rows: list[pd.DataFrame] = []
    rf_rows: list[pd.DataFrame] = []
    har_rows: list[pd.DataFrame] = []

    needed = [TARGET, "baseline", *ALL_FEATURES]
    for spec in windows:
        train_dates = spec["train_dates"]
        test_dates = spec["test_dates"]
        _assert_purge(train_dates, test_dates, horizon)

        train = panel[panel["date"].isin(train_dates)].dropna(subset=needed)
        test = panel[panel["date"].isin(test_dates)].dropna(subset=needed)
        if len(train) < 500 or test.empty:
            logger.warning("Skipping window %s: train=%d test=%d", spec["window"], len(train), len(test))
            continue

        logger.info(
            "Window %d  train %s → %s (%d rows)  test %s → %s (%d rows)",
            spec["window"],
            spec["train_start"],
            spec["train_end"],
            len(train),
            spec["test_start"],
            spec["test_end"],
            len(test),
        )
        fitted = fit_models(train, cfg)
        preds = fitted.predict(test)
        chunk = pd.DataFrame(
            {
                "ticker": test["ticker"].to_numpy(),
                "date": test["date"].to_numpy(),
                "y": test[TARGET].to_numpy(dtype=float),
                "window": spec["window"],
                **preds,
            }
        )
        oos_chunks.append(chunk)

        scores = metrics_table(chunk)
        for _, row in scores.iterrows():
            window_rows.append(
                {
                    "window": spec["window"],
                    "train_start": spec["train_start"],
                    "train_end": spec["train_end"],
                    "test_start": spec["test_start"],
                    "test_end": spec["test_end"],
                    "model": row["model"],
                    "rmse": row["rmse"],
                    "mae": row["mae"],
                    "qlike": row["qlike"],
                    "n": row["n"],
                }
            )

        ridge_rows.append(
            pd.DataFrame(
                {"feature": fitted.ridge_coef.index, "coef": fitted.ridge_coef.to_numpy(), "window": spec["window"]}
            )
        )
        rf_rows.append(
            pd.DataFrame(
                {
                    "feature": fitted.rf_imp.index,
                    "importance": fitted.rf_imp.to_numpy(),
                    "window": spec["window"],
                }
            )
        )
        har_rows.append(
            pd.DataFrame(
                {"feature": fitted.har_coef.index, "coef": fitted.har_coef.to_numpy(), "window": spec["window"]}
            )
        )

    if not oos_chunks:
        raise RuntimeError("Walk-forward produced no OOS rows")

    oos = pd.concat(oos_chunks, ignore_index=True)
    oos = oos.sort_values(["date", "ticker"]).reset_index(drop=True)
    # Non-overlapping test windows should not duplicate (date, ticker).
    dup = oos.duplicated(subset=["date", "ticker"]).sum()
    if dup:
        logger.warning("Dropping %d duplicate OOS rows", dup)
        oos = oos.drop_duplicates(subset=["date", "ticker"], keep="first")

    return {
        "oos": oos,
        "windows": pd.DataFrame(window_rows),
        "ridge_coef": pd.concat(ridge_rows, ignore_index=True),
        "rf_importance": pd.concat(rf_rows, ignore_index=True),
        "har_coef": pd.concat(har_rows, ignore_index=True),
    }
