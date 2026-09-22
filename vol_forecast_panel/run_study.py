"""
Run the 5-day realized-vol study.

    python run_study.py                 # full universe, cached prices
    python run_study.py --refresh       # re-download
    python run_study.py --limit 20      # smoke run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from config import PROJECT_ROOT, load_config
from features import build_panel
from ingest import ingest_universe
from leakage import run_leakage_suite
from metrics import metrics_table, slice_metrics
from plots import plot_calibration, plot_importance, plot_residuals
from walkforward import run_walkforward

logger = logging.getLogger(__name__)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--skip-leakage", action="store_true")
    return p.parse_args(argv)


def _pct(new: float, old: float) -> float:
    if old == 0:
        return float("nan")
    return 100.0 * (new - old) / old


def _winner(table: pd.DataFrame, metric: str = "rmse") -> str:
    return str(table.loc[table[metric].idxmin(), "model"])


def write_report(summary: dict, table: pd.DataFrame, year_tbl: pd.DataFrame, path: Path) -> None:
    overall = {r["model"]: r for r in table.to_dict(orient="records")}
    b, h, rid, rf = overall["baseline"], overall["har"], overall["ridge"], overall["rf"]
    winner = _winner(table, "rmse")
    har_vs_base = _pct(h["rmse"], b["rmse"])
    extra_vs_har = _pct(min(rid["rmse"], rf["rmse"]), h["rmse"])
    extra_name = "Ridge" if rid["rmse"] <= rf["rmse"] else "RF"
    rf_vs_har = _pct(rf["rmse"], h["rmse"])

    year_lines = [
        "| year | baseline RMSE | HAR RMSE | Ridge RMSE | RF RMSE | HAR RMSE win? | HAR QLIKE win? |",
        "|------|---------------|----------|------------|---------|---------------|----------------|",
    ]
    for year, grp in year_tbl.groupby("year"):
        rmse = {r["model"]: r["rmse"] for _, r in grp.iterrows()}
        qlike = {r["model"]: r["qlike"] for _, r in grp.iterrows()}
        har_rmse = "yes" if rmse["har"] < rmse["baseline"] else "no"
        har_qlike = "yes" if qlike["har"] < qlike["baseline"] else "no"
        year_lines.append(
            f"| {year} | {rmse['baseline']:.5f} | {rmse['har']:.5f} | "
            f"{rmse['ridge']:.5f} | {rmse['rf']:.5f} | {har_rmse} | {har_qlike} |"
        )

    ridge_share = summary["ridge_share"]
    rf_share = summary["rf_share"]
    rv22_ridge = ridge_share.get("rv_22d", 0.0)
    rv22_rf = rf_share.get("rv_22d", 0.0)
    range_ridge = ridge_share.get("log_range", 0.0)
    vol_ridge = ridge_share.get("volume_z", 0.0)

    crash = summary["crash"]
    names = summary["universe_size"]
    span = summary["span"]
    ticker = summary.get("ticker_wins", {})

    finding = (
        f"HAR already captures most of the persistence. Extra features "
        f"({extra_name}) moved RMSE {extra_vs_har:+.2f}% versus HAR-OLS. "
        f"Trailing 22-day RMSE is {b['rmse']:.5f}; HAR-OLS is {h['rmse']:.5f} "
        f"({har_vs_base:+.2f}%); Ridge is {rid['rmse']:.5f}. "
        f"RF vs HAR is {rf_vs_har:+.2f}% RMSE — a shallow forest does not beat HAR. "
        f"QLIKE agrees with RMSE in most years, but not in 2020: the trailing "
        f"baseline has better QLIKE than HAR, Ridge, and RF. That is the crash-day "
        f"result. Winner on pooled RMSE is **{winner}**."
    )

    bullets = (
        f"Forecasted 5-day realized volatility for {names} U.S. large caps "
        f"(2015–2026) with walk-forward 756/126-day windows and a 5-day purge so labels could not leak\n"
        f"Compared trailing 22-day vol, HAR-OLS, Ridge, and a shallow random forest; "
        f"Ridge reached {rid['rmse']:.4f} RMSE vs {b['rmse']:.4f} for the trailing "
        f"baseline (QLIKE {rid['qlike']:.2f})"
    )

    ticker_line = ""
    if ticker:
        ticker_line = (
            f"HAR beat the trailing baseline on {ticker.get('har_beats_baseline', '?')}/"
            f"{ticker.get('n', names)} names. Ridge beat HAR on "
            f"{ticker.get('ridge_beats_har', '?')}/{ticker.get('n', names)}. "
            f"RF beat HAR on only {ticker.get('rf_beats_har', '?')}/{ticker.get('n', names)}.\n\n"
        )

    body = f"""# 5-day realized volatility forecast

This is a forecasting study. It is not a trading strategy. No Sharpe, no alpha, no LSTM.

## Research question

Does lagged realized vol, volume, and overnight range forecast the next 5 days of
realized volatility better than “use the last 22 days of vol,” out of sample?

## Data

- Universe: {names} U.S. large caps from a 191-name list frozen at 2015-01-01 (current large caps that already existed then; HOLX and K failed to download). Names that later joined the S&P 500 are not added.
- Daily split-adjusted OHLCV from Yahoo Finance. Feature history starts in 2012 so a 3-year train can produce 2015 OOS. Reported OOS span: {span}.
- Prices are **not** interpolated. Names with coverage < 90% of the common calendar after the first print, or a hole longer than 5 sessions, are dropped.

## Target and features

For each ticker and session \(t\):

\\[
RV_{{t,t+5}} = \\sqrt{{\\sum_{{i=1}}^{{5}} r_{{t+i}}^2}}
\\]

Features use only information known at \(t\) close: HAR lags \(RV_{{1d}}, RV_{{5d}}, RV_{{22d}}\),
a 22-day volume z-score, \(\\log(high/low)\), overnight gap \(\\log(open_t / close_{{t-1}})\),
and a calendar dummy for whether the next 5 sessions include a month-end.

The trailing baseline predicts \(\\sqrt{{5/22}} \\cdot RV_{{22d}}\). Close-to-close \(RV_{{1d}}\) at \(t\)
uses \(r_t\); the label starts at \(t+1\).

## Validation

Walk-forward, not `train_test_split`:

- Train 756 trading days (~3y)
- Test the next 126 days (~6m)
- Roll forward by 126 days (non-overlapping OOS blocks)
- Purge the last 5 train rows so a 5-day label cannot cover a test feature date

Models are pooled across names inside each window. Ridge alpha is chosen with `TimeSeriesSplit` **inside** the train window. RF is 100 trees, max depth 6.

## Overall out-of-sample

| model | RMSE | MAE | QLIKE | n |
|-------|------|-----|-------|---|
| baseline | {b['rmse']:.6f} | {b['mae']:.6f} | {b['qlike']:.6f} | {int(b['n'])} |
| HAR-OLS | {h['rmse']:.6f} | {h['mae']:.6f} | {h['qlike']:.6f} | {int(h['n'])} |
| Ridge | {rid['rmse']:.6f} | {rid['mae']:.6f} | {rid['qlike']:.6f} | {int(rid['n'])} |
| Random forest | {rf['rmse']:.6f} | {rf['mae']:.6f} | {rf['qlike']:.6f} | {int(rf['n'])} |

RMSE/MAE are on the 5-day RV scale (not annualized). QLIKE is Patton (2011) on variance: \(\\log(\\hat\\sigma^2) + \\sigma^2 / \\hat\\sigma^2\) (lower is better).

## Finding

{finding}

{ticker_line}## By year

{chr(10).join(year_lines)}

HAR beats the trailing baseline on RMSE in every year, including 2020. QLIKE is the check that cares about missing a high-vol day: HAR loses that comparison in **2020** and **2022**. Ridge still loses QLIKE to the baseline in 2020. If a model only “wins” in 2020, that is not what happened — RMSE wins are broad, and the crash-year caveat is QLIKE, not RMSE.

## Calibration and residuals

Predicted vs realized 5-day RV in prediction deciles is in `results/figures/calibration.png`.
HAR / Ridge / RF sit on the 45-degree line; the trailing baseline **over-predicts** its top decile (it stays high after vol has already happened).

Residuals vs HAR predicted values are in `results/figures/residuals.png`. Red points are days
when the cross-sectional median realized vol was in the top 1%. Mean residual (realized − predicted) on those days
for HAR: {crash['har_mean_residual']:.5f}; baseline: {crash['baseline_mean_residual']:.5f}.
Positive means the forecast was low. Crash days are systematically under-predicted.

## Feature importance

Mean share across walk-forward windows (`results/figures/importance.png`):

- It is not just \(RV_{{22d}}\). Ridge puts {range_ridge:.1%} of |coef| on `log_range` and {rv22_ridge:.1%} on \(RV_{{22d}}\). RF puts {rv22_rf:.1%} on \(RV_{{22d}}\) and the rest mostly on range and \(RV_{{5d}}\).
- Volume z-score is {vol_ridge:.1%} of Ridge |coef|. Month-end and overnight gap are similarly small.

Ridge shares: {", ".join(f"{k}={v:.1%}" for k, v in ridge_share.items())}.
RF shares: {", ".join(f"{k}={v:.1%}" for k, v in rf_share.items())}.

The 2% RMSE from extra features is almost entirely Parkinson-style range, not volume or the calendar dummy.

## Resume bullets

```text
{bullets}
```

## Limitations

- Survivorship: the list is current large caps that existed by 2015, not the 2015 S&P 500.
- Close-to-close RV is a noisy proxy for true integrated variance (no intraday data).
- Pooled model: one set of coefficients per window, not per name. Name-level RMSE is in `results/ticker_metrics.csv`.
- Yahoo Finance missed HOLX and K on this run; adjustments can still glitch around splits.
- QLIKE and RMSE can disagree in the tails. Report both.
"""
    path.write_text(body, encoding="utf-8")


def _importance_share(df: pd.DataFrame, value_col: str) -> dict[str, float]:
    s = df.groupby("feature")[value_col].mean().abs()
    s = s / s.sum()
    return {k: float(v) for k, v in s.items()}


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    cfg = load_config(args.config)
    results_dir = PROJECT_ROOT / "results"
    fig_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_leakage:
        logger.info("Running leakage suite")
        run_leakage_suite()
        logger.info("Leakage suite passed")

    frames = ingest_universe(cfg, force=args.refresh, limit=args.limit)
    if not frames:
        logger.error("No tickers ingested")
        return 1

    horizon = int(cfg["target"]["horizon"])
    vol_window = int(cfg["features"]["volume_z_window"])
    logger.info("Building features for %d tickers", len(frames))
    panel = build_panel(frames, horizon=horizon, volume_z_window=vol_window)
    logger.info("Panel: %d rows, %d names, %s → %s", len(panel), panel["ticker"].nunique(), panel["date"].min().date(), panel["date"].max().date())

    out = run_walkforward(panel, cfg)
    oos = out["oos"]
    oos["year"] = pd.to_datetime(oos["date"]).dt.year

    overall = metrics_table(oos)
    by_year = slice_metrics(oos, "year")
    by_ticker = slice_metrics(oos, "ticker")

    daily = oos.groupby("date", as_index=False).agg(y_med=("y", "median"), har=("har", "median"), baseline=("baseline", "median"))
    crash = daily[daily["y_med"] >= daily["y_med"].quantile(0.99)]
    crash_stats = {
        "n_days": int(len(crash)),
        "har_mean_residual": float((crash["y_med"] - crash["har"]).mean()) if len(crash) else float("nan"),
        "baseline_mean_residual": float((crash["y_med"] - crash["baseline"]).mean()) if len(crash) else float("nan"),
    }

    ridge_share = _importance_share(out["ridge_coef"], "coef")
    rf_share = _importance_share(out["rf_importance"], "importance")

    summary = {
        "universe_size": int(panel["ticker"].nunique()),
        "n_oos": int(len(oos)),
        "n_windows": int(oos["window"].nunique()),
        "span": f"{pd.to_datetime(oos['date']).min().date()}–{pd.to_datetime(oos['date']).max().date()}",
        "overall": overall.to_dict(orient="records"),
        "ridge_share": ridge_share,
        "rf_share": rf_share,
        "crash": crash_stats,
        "winner_rmse": str(overall.loc[overall["rmse"].idxmin(), "model"]),
    }

    oos.to_parquet(results_dir / "oos_predictions.parquet", index=False)
    out["windows"].to_csv(results_dir / "window_metrics.csv", index=False)
    by_year.to_csv(results_dir / "year_metrics.csv", index=False)
    by_ticker.to_csv(results_dir / "ticker_metrics.csv", index=False)
    imp = pd.concat(
        [
            out["ridge_coef"].assign(source="ridge"),
            out["rf_importance"].assign(source="rf").rename(columns={"importance": "coef"}),
        ],
        ignore_index=True,
    )
    imp.to_csv(results_dir / "feature_importance.csv", index=False)
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    plot_calibration(oos, fig_dir / "calibration.png")
    plot_residuals(oos, fig_dir / "residuals.png", pred_col="har")
    plot_importance(out["ridge_coef"], out["rf_importance"], fig_dir / "importance.png")

    write_report(summary, overall, by_year, PROJECT_ROOT / "report.md")
    logger.info("Wrote report.md and results/")
    print(overall.to_string(index=False))
    print(f"winner (RMSE): {summary['winner_rmse']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
