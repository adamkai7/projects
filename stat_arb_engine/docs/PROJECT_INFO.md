# Statistical Arbitrage Research & Backtesting Engine

## What this project is

A Python research framework for **statistical arbitrage pairs trading** on Nasdaq-100 / QQQ equities.

The pipeline:

1. Ingest and cache daily price history for QQQ / Nasdaq-100 constituents  
2. Screen all pairs for **Engle–Granger cointegration**  
3. Form a hedge-adjusted **spread** and generate **z-score mean-reversion** signals  
4. Backtest with lagged execution, transaction costs, slippage, and borrow  
5. Validate with **walk-forward** out-of-sample windows and a **Monte Carlo** timing test  

Core stack: Python, pandas, NumPy, statsmodels, yfinance, Parquet caching, SQLite, click CLI, pytest.

---

## Method 

**Spread.** For a pair \((A,B)\), estimate hedge ratio \(\beta\) by OLS over a ~3-year window:

\[
P^A_t \approx \alpha + \beta P^B_t, \qquad
S_t = P^A_t - \beta P^B_t - \alpha
\]

**Signal.** Rolling 60-day z-score of the spread:

\[
z_t = \frac{S_t - \mu_t}{\sigma_t}
\]

Enter when \(|z| \ge 2\), exit near \(0.5\), stop near \(4\). Positions are lagged one day so the close-based signal does not claim same-day returns.

**Costs (research defaults).** 5 bps transaction + 2 bps slippage on turnover; 50 bps/year borrow while short; 10% notional position size.

**Walk-forward.** Train ~756 days (estimate \(\beta\)) → test ~126 days (trade with frozen \(\beta\)) → roll forward.

---

## Universe and screen

| Item | Value |
|------|--------|
| Universe | Nasdaq-100 / QQQ constituents |
| Kept after history filter | 94 names |
| Candidate pairs | 4,371 |
| Cointegration passes (p < 0.05) | 172 |
| Kept for follow-up | top 40 by p-value |
| Cointegration lookback | ~756 trading days |
| Correlation pre-filter | \|corr\| ≥ 0.6 |

Example cointegrated pairs from the screen included ADI–KLAC, KLAC–MPWR, AMZN–NVDA, and DDOG–PANW. Some cross-industry pairs also passed statistically (e.g. GILD–XEL).

---

## Results

### Full-sample backtest (costs on; screened \(\beta\))

| Pair | Total return | Ann. vol | Sharpe (rf = 2%) | Max drawdown | Trades |
|------|--------------|----------|------------------|--------------|--------|
| ADI_KLAC | +8.6% | 1.9% | −0.67 | −3.2% | 57 |
| AMZN_NVDA | +4.8% | 2.1% | −0.75 | −5.6% | 56 |
| DDOG_PANW | +10.5% | 3.5% | −0.13 | −6.8% | 33 |
| KLAC_MPWR | +15.2% | 3.9% | −0.18 | −10.0% | 57 |

Positive total return with negative Sharpe can occur when excess return over the risk-free rate is weak relative to volatility and capital is often flat / only partly deployed.

### Walk-forward out-of-sample

| Pair | OOS total return | OOS Sharpe (rf = 2%) | OOS max DD | Windows |
|------|------------------|----------------------|------------|---------|
| ADI_KLAC | +12.6% | −0.34 | −3.4% | 33 |
| AMZN_NVDA | +4.8% | −0.68 | −4.9% | 33 |
| DDOG_PANW | +9.5% | **+0.15** | −3.6% | 14 |
| KLAC_MPWR | +2.6% | −0.66 | −4.8% | 33 |
| GILD_XEL | +7.9% | −0.56 | −4.6% | 33 |

Among inspected pairs, **DDOG_PANW** had the strongest OOS Sharpe, and it was still only modest. Most pairs did not show strong cost-aware out-of-sample risk-adjusted performance.

### Monte Carlo timing test

Positions were shuffled against the fixed spread-return path (a return-shuffle null is invalid for Sharpe because mean and volatility are order-invariant). For the pairs tested, the z-score rule beat random timing, but that did not imply a strong economic edge after costs.

---

## Main finding

Cointegration screening on QQQ pairs finds many statistically mean-reverting spreads. Under a simple z-score trading rule with costs and walk-forward validation, most candidates do **not** look like robust strategies. **Cointegration is easier to find than durable, cost-aware pairs-trading alpha.**

---

## How to run

```bash
cd stat-arb-engine
source .venv/bin/activate
pip install -e ".[dev]"   # if needed

stat-arb ingest-data
stat-arb screen-pairs
stat-arb list-pairs --max-pairs 20
stat-arb run-backtest --pair-id ADI_KLAC
stat-arb run-walkforward --pair-id ADI_KLAC
stat-arb run-significance-test --pair-id DDOG_PANW --source walkforward
```

Config: `configs/default.yaml`  
Tests: `pytest -q`

---

## Disclaimer

Educational research only. Not investment advice. Past simulated performance does not imply future results.
