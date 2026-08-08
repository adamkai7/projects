# Project Summary — Statistical Arbitrage Pairs Engine

A plain-language overview of what was built, what we found, and what it means.

---

## What this project is

A Python research tool that looks for **pairs of QQQ (Nasdaq-100) stocks** that tend to move back together when their prices drift apart, then **tests whether trading that idea would have worked** after realistic frictions.

Educational research only — not a live trading system and not investment advice.

---

## The idea in everyday terms

Imagine two similar companies. Their stock prices both wander over time, but the **difference between them** (after adjusting for a hedge ratio) often has a “normal” range.

- If that difference gets **unusually large**, maybe it will shrink → trade as if it will shrink.
- If it gets **unusually small**, maybe it will widen back → trade the other way.

That difference is the **spread**. “Unusually” is measured with a **z-score** (how many standard deviations away from the recent average).

The statistical name for “these two wandering prices still share a stable long-run relationship” is **cointegration**.

---

## What we actually did

1. **Universe:** ~100 Nasdaq-100 / QQQ stocks.  
2. **Cleaning:** dropped names without enough price history.  
3. **Scan:** checked essentially every pair for cointegration (thousands of pairs).  
4. **Signals:** for promising pairs, used a z-score rule to go long/short the spread.  
5. **Backtest:** simulated trades with costs and without using tomorrow’s information today.  
6. **Extra checks:** walk-forward (fit on past, trade on later periods) and a randomness check on timing.

---

## Results

### Scan
- After cleaning, about **94** stocks remained.
- About **172** pairs looked cointegrated under our statistical cutoff (we kept the strongest ~40 for follow-up).
- Some pairs made sense economically (similar businesses, e.g. certain semiconductor or software names).
- Some pairs looked more like statistical coincidences across unrelated industries.

### Trading performance
- Turning “cointegrated on a test” into “good trading strategy” was much harder.
- After costs and out-of-sample checks, **most pairs did not look strong**.
- Risk-adjusted results were often weak or negative compared with a simple risk-free / cash hurdle.
- Among pairs we inspected more carefully, **DDOG vs PANW** was one of the better out-of-sample cases, and even that was only **modest**, not a clear winner.

### Important nuance
- The trading rule often did **better than randomly timed** long/short positions on the same spread.
- That means the signal had *some* timing content.
- It does **not** automatically mean the strategy is economically attractive after costs, risk, and real-world constraints.

---

## Final research insight

**Cointegration screening finds relationships. It does not automatically find profitable trades.**

On this QQQ run, with a simple z-score mean-reversion rule and cost-aware out-of-sample testing:

- cointegration was relatively common under a naïve screen,
- durable, cost-adjusted edge was not.

---

## How to run it (short)

```bash
cd stat_arb_engine
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

stat-arb ingest-data
stat-arb screen-pairs
stat-arb list-pairs --max-pairs 20
stat-arb run-backtest --pair-id ADI_KLAC
stat-arb run-walkforward --pair-id ADI_KLAC
```

More technical detail and tabulated results: `docs/PROJECT_INFO.md`
