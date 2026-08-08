# Statistical Arbitrage Research Engine — Full Research Report

**Audience:** I am a rising 3rd-year undergrad (math / scientific computation), preparing for quant research / quant analyst / ML internships.

**Universe:** Nasdaq-100 / QQQ constituents (≈101 tickers; GOOG dropped vs GOOGL to avoid trivial dual-class cointegration).

**Disclaimer:** Educational research only. Not investment advice. Past results ≠ future results.

---

## 0. How to run this yourself (do this once)

From the project root:

```bash
cd /Users/adam/projects/stat-arb-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 1) Download ~10y of daily prices for QQQ names (cached to Parquet)
stat-arb ingest-data

# 2) Engle–Granger screen across all pairs
stat-arb screen-pairs
stat-arb list-pairs --max-pairs 20

# 3) Backtest one pair (example: semiconductors ADI vs KLAC)
stat-arb run-backtest --pair-id ADI_KLAC

# 4) Walk-forward out-of-sample validation
stat-arb run-walkforward --pair-id ADI_KLAC

# 5) Monte Carlo timing significance test
stat-arb run-significance-test --pair-id ADI_KLAC --source walkforward

# Optional UI
pip install -e ".[dashboard]"
stat-arb launch-dashboard
```

**Re-run after code/config changes:** cache hits mean `ingest-data` is fast. Use `stat-arb ingest-data --force` to re-download. Config knobs live in `configs/default.yaml`.

**Tests:**

```bash
pytest -q
ruff check src tests
```

---

## 1. What problem this project solves

Most undergrad “stock” projects ask: *can I predict the next return?* and report an accuracy number.

A research desk asks a harder question:

> If I claim a **statistical arbitrage** edge from **mean-reverting spreads**, can I show that the edge is not (a) correlation theater, (b) look-ahead bias, (c) transaction-cost fiction, or (d) multiple-testing luck?

This engine is built around that skepticism:

1. Screen QQQ names for **cointegration** (not just correlation).
2. Trade the **spread** with a **z-score** rule.
3. Simulate with **costs / slippage / borrow**.
4. Validate with **walk-forward** windows (β estimated only in-sample).
5. Test timing skill with a **Monte Carlo position-shuffle** null.

Honest outcome of the first full run: **cointegration is common under naïve screens; economically meaningful, cost-aware, OOS alpha is not.** That conclusion is *more* internship-credible than a fake Sharpe of 3.

---

## 2. Map of the codebase (what each piece does)

```text
stat-arb-engine/
├── configs/default.yaml     # all research knobs (windows, costs, thresholds)
├── configs/universes.yaml   # QQQ ticker list + coarse sectors
├── src/pipeline/ingest.py   # yfinance → Parquet cache + SQLite prices
├── src/screening/cointegration.py  # Engle–Granger pair screen
├── src/signals/zscore.py    # spread + rolling z + position state machine
├── src/backtest/engine.py   # lagged execution, costs, equity curve
├── src/backtest/metrics.py   # Sharpe, Sortino, max drawdown, …
├── src/validation/walkforward.py  # walk-forward + Monte Carlo
├── src/cli.py               # click CLI
├── tests/test_core.py       # synthetic cointegration + metric tests
└── docs/RESEARCH_REPORT.md  # this document
```

**Pipeline flow**

```text
OHLCV ingest → cointegration screen → z-score signals → backtest (+costs)
                                                      → walk-forward OOS
                                                      → Monte Carlo timing test
```

---

## 3. Definitions 

### 3.1 Random walk / I(1)

A price series is often modeled as:

\[
P_t = P_{t-1} + \varepsilon_t, \quad \varepsilon_t \text{ noise.}
\]

Then \(P_t\) is **integrated of order 1** (I(1)): the *level* wanders; the *difference* \(\Delta P_t\) is roughly stationary. You **cannot** treat raw prices like independent samples in a normal regression without care.

### 3.2 Stationarity

A series is (weakly) **stationary** if its mean and variance don’t systematically drift over time. Mean-reversion strategies need a stationary object to trade. Prices usually aren’t; **spreads** of cointegrated assets can be.

### 3.3 Correlation vs cointegration

- **Correlation:** \(P^a\) and \(P^b\) tend to move *together day-to-day*.
- **Cointegration:** even if each price is a random walk, there exists \(\beta\) such that

\[
e_t = P^a_t - \alpha - \beta P^b_t
\]

is **stationary**. They share a long-run equilibrium; deviations \(e_t\) tend to snap back.

Two stocks can be highly correlated and **not** cointegrated (they wander apart permanently). Two stocks can cointegrate with moderate correlation. **Pairs trading needs cointegration of the spread, not just correlation.**

### 3.4 Engle–Granger two-step test

1. Regress \(P^a_t = \alpha + \beta P^b_t + \varepsilon_t\) (OLS). \(\hat\beta\) is the **hedge ratio**.
2. Test whether residuals \(\hat\varepsilon_t\) have a unit root (ADF / `statsmodels.tsa.stattools.coint`).  
   - Low p-value ⇒ reject “no cointegration” ⇒ evidence the spread is mean-reverting.

**Hedge ratio intuition:** if \(\hat\beta = 1.35\) for ADI on KLAC, a long-spread trade is roughly “long ADI, short 1.35 price-units of KLAC” (converted to dollars carefully in the backtest).

### 3.5 Z-score signal

Spread \(S_t\) (we use the residual form \(P^a - \hat\beta P^b - \hat\alpha\)). Rolling window \(W\) (default 60 days):

\[
z_t = \frac{S_t - \mu_t}{\sigma_t}.
\]

| Rule | Meaning |
|------|---------|
| \(z \le -2\) | spread “too low” → **long spread** (buy A, short β B) |
| \(z \ge +2\) | spread “too high” → **short spread** |
| \(|z| \le 0.5\) | back near mean → **exit** |
| \(|z| \ge 4\) | blow-out → **stop** (relationship may have broken) |

### 3.6 Look-ahead bias

Using information that would not have been available at decision time. Examples we guard against:

- Estimating \(\beta\) on the **same** days you trade (full-sample β).
- Entering at today’s close using a signal that needed today’s close, then booking today’s close-to-close return.  
  **Our fix:** signal at close of \(t\) → position earns returns from \(t \to t+1\) (`shift(1)`).

### 3.7 Walk-forward validation

Repeat:

1. **Train** window (~3y): estimate \(\beta\).
2. **Test** window (~6m): trade with frozen \(\beta\).
3. Roll forward by ~1 quarter; stitch OOS equity.

This answers: “Does the idea work **out of sample** when parameters aren’t fit on the evaluation period?”

### 3.8 Sharpe / Sortino / max drawdown

For daily returns \(r_t\), risk-free daily rate \(r_f\):

\[
\text{Sharpe} = \sqrt{252}\;\frac{\overline{r - r_f}}{\sigma(r)}.
\]

**Sortino** replaces \(\sigma\) with downside deviation (only negative excess returns).  
**Max drawdown** = worst peak-to-trough drop on the equity curve.

### 3.9 Monte Carlo timing test (and a subtlety you should mention)

**Wrong null (we initially almost shipped this):** shuffle the strategy’s *net returns* and recompute Sharpe.  
That does **nothing** — mean and variance are invariant to order, so Sharpe never changes.

**Correct null (what we use):** keep the market path (spread returns) fixed; **shuffle the position series**. Rebuild PnL + costs. Ask:

\[
p = \mathbb{P}(\text{Sharpe}_{\text{random timing}} \ge \text{Sharpe}_{\text{strategy}}).
\]

This tests **timing skill**, not “are returns lucky in an order-invariant sense.”

---

## 4. Data construction

| Item | Choice | Why |
|------|--------|-----|
| Universe | Nasdaq-100 via Nasdaq API → `configs/universes.yaml` `qqq` | Matches “QQQ stocks” request; liquid large-caps |
| Dual class | Drop `GOOG`, keep `GOOGL` | Same company → near-perfect cointegration is not alpha |
| Prices | `yfinance` adj close, Parquet cache | Free, reproducible research |
| History filter | ≥ 756 trading days | Need enough data for EG + walk-forward |
| First run | **94 / 101** kept; skipped short history: ALAB, ARM, CRWV, HONA, NBIS, SNDK, SPCX | Honest data hygiene |

SQLite tables: `prices`, `pairs`, `signals`, `backtest_results` — lets you show SQL without running Postgres.

---

## 5. Screening logic (what the computer did)

1. Form all pairs among kept tickers: \(\binom{94}{2} = 4371\) candidates.
2. For each pair, **align dates pairwise** (critical bugfix: do **not** inner-join all 94 into one panel — that collapsed history to ~36 days).
3. Pre-filter: \(|\text{corr}| \ge 0.6\) (cheap).
4. Engle–Granger at 5% significance on the last ~756 days.
5. Keep top 40 by p-value.

**First-run headline:** **172** pairs passed the 5% EG screen; top 40 stored.

### Multiple testing

At \(\alpha=0.05\), even if **no** pair were cointegrated, you’d expect roughly \(0.05 \times 4371 \approx 218\) false positives. So a raw EG screen **overstates** how many “real” equilibria you found. Desk-quality work would add:

- Bonferroni / FDR control,
- economic filters (same industry, borrowability, dollar neutrality),
- and **OOS trading tests** (which we do).

---

## 6. Backtest mechanics (intuition)

On each day:

1. Build spread with hedge ratio \(\beta\).
2. Compute rolling \(z_t\); update desired position \(s_t \in \{-1,0,+1\}\).
3. Hold yesterday’s position through today’s return (no look-ahead).
4. Approximate spread return using a dollar-aware hedge weight.
5. Charge costs on **turnover** (commission + slippage in bps) and a crude daily borrow fee when short.

Default costs (`configs/default.yaml`): 5 bps commission + 2 bps slippage each way; 50 bps/year borrow.

“My first mental model without costs looked fine; after costs and lagged execution, most ‘edges’ evaporated. That’s the point of the project.”

---

## 7. Empirical results (this machine’s first full run)

### 7.1 Top cointegrated pairs (by EG p-value)

| Pair | \(\hat\beta\) | EG p-value | Corr | Economic note |
|------|---------------|------------|------|----------------|
| **ADI_KLAC** | 1.35 | 0.00009 | 0.96 | Semis equipment / analog — sensible |
| MU_ROST | 6.26 | 0.00037 | 0.88 | Cross-sector — treat skeptically |
| ADI_ROST | 1.88 | 0.00038 | 0.94 | Cross-sector |
| AXON_BKNG | 5.32 | 0.00052 | 0.95 | Cross-sector |
| **KLAC_MPWR** | 0.15 | 0.00069 | 0.93 | Semis — sensible |
| **AMZN_NVDA** | 0.64 | 0.0014 | 0.93 | Mega-cap tech — famous, crowded |
| GILD_XEL | 2.36 | 0.0018 | 0.96 | Health care vs utility — suspicious economically |
| **DDOG_PANW** | (from screen list) | ~0.007 | 0.91 | Software / cybersecurity — sensible |

Rule of thumb for interviews: **statistical cointegration + economic story** beats “lowest p-value alone.”

### 7.2 Full-sample backtest (costs on; still uses screen β — optimistic)

| Pair | Total return | Ann. vol | Sharpe (rf=2%) | Max DD | Trades |
|------|--------------|----------|----------------|--------|--------|
| ADI_KLAC | +8.6% | 1.9% | **-0.67** | -3.2% | 57 |
| AMZN_NVDA | +4.8% | 2.1% | **-0.75** | -5.6% | 56 |
| DDOG_PANW | +10.5% | 3.5% | **-0.13** | -6.8% | 33 |
| KLAC_MPWR | +15.2% | 3.9% | **-0.18** | -10.0% | 57 |

Why is Sharpe negative while total return is positive?  
Position size is only 10% of capital and the book is often flat, so **annualized excess return vs a 2% risk-free rate** is weak. The equity curve drifts up slowly; cash-at-rf would have done fine. This is a feature of honest reporting, not a bug.

### 7.3 Walk-forward OOS (the number that matters)

| Pair | OOS total return | OOS Sharpe (rf=2%) | OOS max DD | Windows |
|------|------------------|--------------------|------------|---------|
| ADI_KLAC | +12.6% | **-0.34** | -3.4% | 33 |
| AMZN_NVDA | +4.8% | **-0.68** | -4.9% | 33 |
| **DDOG_PANW** | +9.5% | **+0.15** | -3.6% | 14 |
| KLAC_MPWR | +2.6% | **-0.66** | -4.8% | 33 |
| GILD_XEL | +7.9% | **-0.56** | -4.6% | 33 |

**Best OOS Sharpe among these:** DDOG_PANW ≈ 0.15 — still modest; windows mix positive and negative Sharpes (not a stable money printer).

### 7.4 Monte Carlo timing test (fixed market path, shuffled positions)

| Pair | Observed Sharpe | Null mean Sharpe | p-value | Significant vs random timing? |
|------|-----------------|------------------|---------|-------------------------------|
| ADI_KLAC | -0.38 | -3.02 | ≈0 | Yes |
| DDOG_PANW | +0.13 | -1.56 | 0.002 | Yes |
| AMZN_NVDA | -0.72 | -2.31 | ≈0 | Yes |


> “Our z-score timing is **significantly better than random position shuffles** on the same spread path — so the signal isn’t pure noise relative to a dumb null — but that does **not** imply the strategy clears the risk-free hurdle or survives capacity, borrow, or regime change. Economic significance ≠ statistical timing significance.”


---

## 8. What we learned (failure modes = resume strength)

1. **Cointegration screen ≠ tradeable alpha.** EG at 5% on thousands of pairs finds many relationships; most don’t pay after costs.
2. **Pairwise alignment matters.** A global inner join across the universe destroyed the sample (36 days). Fixing that was a real research bug, not cosmetics.
3. **Permutation tests must match the claim.** Shuffling returns cannot test Sharpe; shuffling **positions** can test timing.
4. **Negative Sharpe with positive PnL** happens when capital sits in cash and rf > strategy excess return — report both total return and risk-adjusted metrics.
5. **Economic coherence filter** should come next (same industry, positive correlation only, borrow constraints).

---

## 9. Interview question bank (with your project’s answers)

**Q: Cointegration vs correlation?**  
A: Correlation is about co-movement; cointegration is about a stationary linear combination of I(1) prices — a long-run equilibrium for the spread.

**Q: How did you avoid look-ahead?**  
A: β from train windows only; trade on test windows; signal at \(t\) earns \(t\to t+1\) returns.

**Q: Why walk-forward instead of one train/test split?**  
A: Markets are non-stationary; one split can be a lucky regime. Rolling windows stress-test stability.

**Q: How do you know results aren’t luck?**  
A: Timing Monte Carlo (shuffle positions vs fixed spread returns). Also discuss multiple testing on the screen.

**Q: Why did many Sharpes go negative after costs?**  
A: Spread mean-reversion edge is small; turnover taxes it; rf hurdle and low deployment make excess Sharpe harsh — which is realistic.

**Q: How would you scale to 5,000 pairs?**  
A: Vectorize correlation pre-filter; parallelize EG; store only top-N; use DuckDB; consider Numba on the position loop.

**Q: What would you do next on a desk?**  
A: Industry-constrained universe, FDR control, rolling β, borrow-aware shorts, multi-pair risk parity, regime filter, and paper-trade only pairs that clear OOS + significance + capacity checks.

---

## 10. Suggested resume bullets (use *your* numbers)

- Built a Python statistical-arbitrage research engine screening **4,371** Nasdaq-100 pairs with Engle–Granger cointegration, caching **94** liquid names via yfinance/Parquet/SQLite.
- Implemented z-score mean-reversion signals with lagged execution, transaction costs, and walk-forward OOS validation across **33** rolling windows.
- Applied a Monte Carlo **position-shuffle** significance test (correct null for Sharpe) and documented that timing skill ≠ economic edge after costs — the research discipline expected on a QR desk.
- Delivered pytest coverage of core math plus GitHub Actions CI (ruff + pytest).

---

## 11. Parameter cheat sheet (`configs/default.yaml`)

| Knob | Default | Meaning |
|------|---------|---------|
| `screening.min_correlation` | 0.6 | Pre-filter before EG |
| `screening.significance_level` | 0.05 | EG p-value cutoff |
| `screening.lookback_days` | 756 | ~3y test window |
| `signals.zscore_window` | 60 | Rolling μ, σ |
| `signals.entry_z` / `exit_z` / `stop_z` | 2.0 / 0.5 / 4.0 | Trade rules |
| `backtest.transaction_cost_bps` | 5 | Commission-like |
| `backtest.slippage_bps` | 2 | Adverse move |
| `validation.train_days` | 756 | WF train |
| `validation.test_days` | 126 | WF test |
| `validation.n_permutations` | 1000 | Monte Carlo |

---

## 12. Bottom line

You now have a **real research artifact**, not a toy predictor:

- Math: cointegration, z-scores, risk metrics, hypothesis testing.
- Engineering: caching, SQLite, vectorized/lagged backtest, CLI, tests, CI.
- Judgment: the strategy mostly **does not** clear an economic bar on QQQ pairs under these assumptions — and you can **explain why**, with numbers.

That combination — technical execution + statistical humility — is what a 2027 QR Markets internship screen is looking for.
