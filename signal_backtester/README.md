# Signal Backtester

A small, honest backtesting harness for equity signals — built to replace
"I eyeballed the chart and it looked right" with real, comparable metrics
(hit rate, Sharpe, max drawdown, total return — always shown against a
buy-and-hold baseline on the same ticker/period).

This grew out of translating years of discretionary ThinkScript/thinkorswim
indicator-watching (trend regression fitting, z-score mean-reversion,
momentum/breakout screens) into code that can actually be back-tested and
quoted with a number instead of a gut feeling.

## Why this exists

Most of my ThinkScript studies were built to *look at* — visual aids for
discretionary or paper trading, not codified, backtestable rules. That's
fine for idea generation, but it means I never had a rigorous answer to
"how much better than random/baseline is this, really?" This tool answers
that question for any signal you can express as a position rule.

## What's here

- `data.py` — pulls daily OHLC from Yahoo Finance via `yfinance`, with a
  local CSV cache so repeated runs don't re-hit the network.
- `signals.py` — reference implementations of the *signal families* behind
  common ThinkScript studies:
  - `zscore_mean_reversion` — rolling z-score fade (mirrors `MeanRev1_L`,
    `ZScorePopDrop`, `ZScore_LSTRATEGY`, etc.)
  - `regression_trend_momentum` — rolling OLS slope trend filter (mirrors
    `ModR2TrendSTRATEGY`, `SwingRegressionSTUDY`, etc.)
  - `nday_momentum` — simple N-day return breakout (mirrors
    `MultiReturnTrend`/`ReturnMA`-style studies)
  - `custom_signal_template` — copy this and translate your exact
    ThinkScript entry/exit logic into pandas to backtest it directly.
- `backtest.py` — the engine: positions are lagged one day (no look-ahead),
  transaction costs + slippage are charged on turnover, and it reports
  total return, Sharpe (rf-adjusted), max drawdown, trade count, and
  per-trade hit rate — each against a buy-and-hold baseline on the same
  ticker/period.
- `run.py` — CLI entry point.

## Usage

```bash
pip install -r requirements.txt

# Z-score mean-reversion on QQQ
python run.py --ticker QQQ --strategy zscore --start 2019-01-01 --window 20 --entry-z 1.5 --exit-z 0.3

# Regression-trend-following on QQQ
python run.py --ticker QQQ --strategy trend --start 2019-01-01 --window 30

# N-day momentum on AAPL
python run.py --ticker AAPL --strategy momentum --start 2019-01-01 --lookback 15
```

Example output:

```
QQQ | strategy=zscore | 2019-01-01 -> latest
------------------------------------------------------------
Trades:                 42
Hit rate (per trade):   72.6%
Total return:           +34.0%   (buy-and-hold: +389.4%)
Sharpe (rf=2%):         +0.20   (buy-and-hold: +0.91)
Max drawdown:           -23.4%   (buy-and-hold: -35.1%)
```

Note the honest result here: high per-trade hit rate (72.6%) but the
strategy still badly underperforms buy-and-hold in total return and Sharpe,
because QQQ trended strongly over this window and a mean-reversion strategy
is structurally fighting the tape. That's the kind of finding this tool is
for — a real number instead of "the signal looked good on the chart."

## Adapting your own signal

Translate the exact entry/exit rule from your ThinkScript study into
`custom_signal_template()` in `signals.py` (it takes a `pandas.Series` of
daily closes and returns a position series of `{-1, 0, 1}`), then point
`run.py` at it, or just call `run_backtest()` directly from a notebook:

```python
from data import load_prices
from signals import custom_signal_template
from backtest import run_backtest

close = load_prices("AAPL", start="2020-01-01")["Close"]
position = custom_signal_template(close)
result = run_backtest(close, position)
print(result.summary())
```

## Caveats

- Costs/slippage default to 5 bps + 2 bps per position change — adjust with
  `--cost-bps` / `--slippage-bps` for your actual broker/asset.
- This is daily-bar, single-asset, long/short-flat only — it does not model
  intraday execution, options, or portfolio-level risk.
- A backtest on one ticker/period is not proof of an edge; treat results
  here the same way the rest of this portfolio treats backtest results —
  as evidence to be walk-forward validated, not a final verdict.
