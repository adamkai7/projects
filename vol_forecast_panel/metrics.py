"""RMSE, MAE, and QLIKE on the original 5-day RV scale."""

from __future__ import annotations

import numpy as np
import pandas as pd

MODEL_COLS = ("baseline", "har", "ridge", "rf")


def rmse(y: np.ndarray, pred: np.ndarray) -> float:
    err = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean(err**2)))


def mae(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y, dtype=float) - np.asarray(pred, dtype=float))))


def qlike(y: np.ndarray, pred: np.ndarray) -> float:
    """
    Patton (2011) QLIKE on variance, with vol inputs: log(pred^2) + y^2 / pred^2.

    Lower is better. Misses on high-vol days are penalized more than RMSE.
    """
    y2 = np.asarray(y, dtype=float) ** 2
    p2 = np.asarray(pred, dtype=float) ** 2
    p2 = np.clip(p2, 1e-16, None)
    return float(np.mean(np.log(p2) + y2 / p2))


def score_block(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    return {"rmse": rmse(y, pred), "mae": mae(y, pred), "qlike": qlike(y, pred), "n": int(len(y))}


def metrics_table(frame: pd.DataFrame, models: tuple[str, ...] = MODEL_COLS) -> pd.DataFrame:
    y = frame["y"].to_numpy(dtype=float)
    rows = []
    for name in models:
        rows.append({"model": name, **score_block(y, frame[name].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def slice_metrics(frame: pd.DataFrame, by: str, models: tuple[str, ...] = MODEL_COLS) -> pd.DataFrame:
    pieces = []
    for key, grp in frame.groupby(by, sort=True):
        tbl = metrics_table(grp, models=models)
        tbl[by] = key
        pieces.append(tbl)
    return pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()


def calibration_deciles(frame: pd.DataFrame, pred_col: str, n_bins: int = 10) -> pd.DataFrame:
    tmp = frame[["y", pred_col]].dropna().copy()
    tmp["decile"] = pd.qcut(tmp[pred_col], q=n_bins, labels=False, duplicates="drop")
    out = (
        tmp.groupby("decile", observed=True)
        .agg(
            n=("y", "size"),
            pred_mean=(pred_col, "mean"),
            realized_mean=("y", "mean"),
            realized_std=("y", "std"),
        )
        .reset_index()
    )
    return out
