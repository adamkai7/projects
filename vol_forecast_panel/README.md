# 5-day realized volatility forecast

Forecast next-5-day realized vol on a frozen U.S. large-cap panel, and test
whether anything beats a trailing-vol baseline once time is respected.

**Honest result:** numbers are filled by `python run_study.py` into
[`report.md`](report.md). If HAR-OLS wins, that is the finding — vol is
persistent, and the three-lag HAR is the literature baseline a dashboard
22-day number has to beat.

This is data science (target, features, temporal split, metric). It is not a
trading strategy.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_study.py            # writes report.md and results/
python -m pytest               # leakage + target construction
```

`--limit 20` is a smoke run. `--refresh` re-downloads prices. Daily OHLCV is
cached under `data/cache/`.

## Files

| file | purpose |
|------|---------|
| `ingest.py` | Yahoo Finance OHLCV, Parquet cache, drop gappy names (no interpolation) |
| `features.py` | 5-day forward RV target and HAR / volume / range / calendar features |
| `walkforward.py` | 756 / 126 walk-forward with a 5-day purge |
| `models.py` | trailing baseline, HAR-OLS, Ridge, shallow RF |
| `leakage.py` | features at t ignore the future; label starts at t+1 |
| `run_study.py` | study driver |
| `report.md` | one-page note with RMSE / MAE / QLIKE, year slices, importance |
| `universe.txt` | 187 names frozen at 2015 |

## Resume bullets

See the bottom of `report.md` after a run. Template:

```text
Forecasted 5-day realized volatility for 187 U.S. large caps (2015–2026) with walk-forward 756/126-day windows and a 5-day purge so labels could not leak
Compared trailing 22-day vol, HAR-OLS, Ridge, and a shallow random forest; [HAR / Ridge] reached X RMSE vs Y for the trailing baseline (QLIKE Z)
```
