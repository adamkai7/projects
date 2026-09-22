# 5-day realized volatility forecast

This is a forecasting study. It is not a trading strategy. No Sharpe, no alpha, no LSTM.

## Research question

Does lagged realized vol, volume, and overnight range forecast the next 5 days of
realized volatility better than “use the last 22 days of vol,” out of sample?

## Data

- Universe: 189 U.S. large caps, frozen at 2015-01-01 (the same 187-name list as the thinkScript study, after dropping gappy series). Names that later joined the S&P 500 are not added.
- Daily split-adjusted OHLCV from Yahoo Finance, 2015-02-06–2026-02-13.
- Prices are **not** interpolated. Names with coverage < 90% of the common calendar, or a hole longer than 5 sessions, are dropped.

## Target and features

For each ticker and session \(t\):

\[
RV_{t,t+5} = \sqrt{\sum_{i=1}^{5} r_{t+i}^2}
\]

Features use only information known at \(t\) close: HAR lags \(RV_{1d}, RV_{5d}, RV_{22d}\),
a 22-day volume z-score, \(\log(high/low)\), overnight gap \(\log(open_t / close_{t-1})\),
and a calendar dummy for whether the next 5 sessions include a month-end.

The trailing baseline predicts \(\sqrt{5/22} \cdot RV_{22d}\). Close-to-close \(RV_{1d}\) at \(t\)
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
| baseline | 0.024758 | 0.015732 | -5.385381 | 523908 |
| HAR-OLS | 0.022756 | 0.014013 | -5.438857 | 523908 |
| Ridge | 0.022199 | 0.013670 | -5.472126 | 523908 |
| Random forest | 0.022636 | 0.013676 | -5.467373 | 523908 |

RMSE/MAE are on the 5-day RV scale (not annualized). QLIKE is Patton (2011) on variance: \(\log(\hat\sigma^2) + \sigma^2 / \hat\sigma^2\).

## Finding

HAR already captures most of the persistence; extra features added -2.45% RMSE via Ridge versus HAR-OLS (-2.45% is an improvement if negative). The trailing 22-day baseline RMSE is 0.02476; HAR-OLS is 0.02276 (-8.09%). Winner on RMSE is **ridge**.

## By year

| year | baseline RMSE | HAR RMSE | Ridge RMSE | RF RMSE | HAR win? |
|------|---------------|----------|------------|---------|----------|
| 2015 | 0.01938 | 0.01752 | 0.01737 | 0.01723 | yes |
| 2016 | 0.02268 | 0.02024 | 0.01976 | 0.01946 | yes |
| 2017 | 0.01901 | 0.01689 | 0.01665 | 0.01641 | yes |
| 2018 | 0.02098 | 0.01967 | 0.01917 | 0.01906 | yes |
| 2019 | 0.02045 | 0.01776 | 0.01746 | 0.01733 | yes |
| 2020 | 0.04177 | 0.03931 | 0.03803 | 0.04147 | yes |
| 2021 | 0.01865 | 0.01791 | 0.01748 | 0.01732 | yes |
| 2022 | 0.02433 | 0.02306 | 0.02247 | 0.02260 | yes |
| 2023 | 0.02088 | 0.01950 | 0.01900 | 0.01876 | yes |
| 2024 | 0.02502 | 0.02230 | 0.02191 | 0.02151 | yes |
| 2025 | 0.02901 | 0.02577 | 0.02517 | 0.02517 | yes |
| 2026 | 0.02739 | 0.02718 | 0.02651 | 0.02641 | yes |

If a model only wins in 2020, that is in the table. HAR win is RMSE vs the trailing baseline.

## Calibration and residuals

Predicted vs realized 5-day RV in prediction deciles is in `results/figures/calibration.png`.
A well-calibrated forecast sits on the 45-degree line.

Residuals vs HAR predicted values are in `results/figures/residuals.png`. Red points are days
when the cross-sectional median realized vol was in the top 1%. Mean residual on those days
for HAR: 0.06650 (negative = under-prediction). Baseline:
0.03312.

## Feature importance

Mean share across walk-forward windows (`results/figures/importance.png`):

- Ridge |coef| on scaled features: \(RV_{22d}\) share = 31.0%.
- RF impurity: \(RV_{22d}\) share = 47.3%.

Ridge shares: log_range=36.4%, month_end=2.0%, overnight_gap=4.1%, rv_1d=7.6%, rv_22d=31.0%, rv_5d=17.0%, volume_z=2.0%.
RF shares: log_range=30.6%, month_end=0.4%, overnight_gap=1.2%, rv_1d=0.7%, rv_22d=47.3%, rv_5d=17.5%, volume_z=2.3%.

## Resume bullets

```text
Forecasted 5-day realized volatility for 189 U.S. large caps (2015-02-06–2026-02-13) with walk-forward 756/126-day windows and a 5-day purge so labels could not leak
Compared trailing 22-day vol, HAR-OLS, Ridge, and a shallow random forest; ridge reached 0.02220 RMSE vs 0.02476 for the trailing baseline (QLIKE -5.4721)
```

## Limitations

- Survivorship: the 187 names are current large caps that existed by 2015, not the 2015 S&P 500.
- Close-to-close RV is a noisy proxy for true integrated variance (no intraday data).
- Pooled model: one set of coefficients per window, not per name. Name-level RMSE is in `results/ticker_metrics.csv`.
- Yahoo Finance adjustments can still glitch around splits; gappy names are dropped rather than repaired.
