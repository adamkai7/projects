"""Study plots: calibration, residuals, feature importance."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from metrics import calibration_deciles

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


def plot_calibration(oos: pd.DataFrame, path: Path, models: tuple[str, ...] = ("baseline", "har", "ridge", "rf")) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    vmax = 0.0
    for name in models:
        cal = calibration_deciles(oos, name)
        ax.plot(cal["pred_mean"], cal["realized_mean"], marker="o", label=name)
        vmax = max(vmax, float(cal["pred_mean"].max()), float(cal["realized_mean"].max()))
    ax.plot([0, vmax], [0, vmax], ls="--", c="0.5", lw=1, label="y = x")
    ax.set_xlabel("Mean predicted 5-day RV")
    ax.set_ylabel("Mean realized 5-day RV")
    ax.set_title("Reliability: predicted vs realized vol by prediction decile")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_residuals(oos: pd.DataFrame, path: Path, pred_col: str = "har") -> None:
    y = oos["y"].to_numpy(dtype=float)
    pred = oos[pred_col].to_numpy(dtype=float)
    resid = y - pred
    rng = np.random.default_rng(0)
    if len(oos) > 25000:
        idx = rng.choice(len(oos), size=25000, replace=False)
        pred_s, resid_s = pred[idx], resid[idx]
    else:
        pred_s, resid_s = pred, resid

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.hexbin(pred_s, resid_s, gridsize=50, mincnt=1, cmap="Greys")
    ax.axhline(0.0, c="0.4", lw=1)
    # Crash-ish days: top 1% of cross-sectional median realized vol.
    daily = oos.groupby("date", as_index=False).agg(y_med=("y", "median"), pred_med=(pred_col, "median"))
    thresh = daily["y_med"].quantile(0.99)
    crash = daily[daily["y_med"] >= thresh]
    ax.scatter(crash["pred_med"], crash["y_med"] - crash["pred_med"], s=18, c="C3", label="top 1% vol days (median)", zorder=3)
    ax.set_xlabel(f"Predicted 5-day RV ({pred_col})")
    ax.set_ylabel("Residual (realized − predicted)")
    ax.set_title("Residuals vs predicted: are crash days systematically low?")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_importance(ridge_coef: pd.DataFrame, rf_imp: pd.DataFrame, path: Path) -> None:
    ridge = ridge_coef.groupby("feature")["coef"].mean().abs()
    ridge = ridge / ridge.sum()
    rf = rf_imp.groupby("feature")["importance"].mean()
    rf = rf / rf.sum()
    features = list(ridge.index)
    x = np.arange(len(features))
    width = 0.38

    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.bar(x - width / 2, [ridge.get(f, 0.0) for f in features], width, label="Ridge |coef| (scaled, share)")
    ax.bar(x + width / 2, [rf.get(f, 0.0) for f in features], width, label="RF impurity (share)")
    ax.set_xticks(x)
    ax.set_xticklabels(features, rotation=25, ha="right")
    ax.set_ylabel("Share of total importance")
    ax.set_title("Is it just RV_22d? Mean importance across walk-forward windows")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
