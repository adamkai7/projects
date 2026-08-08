# Internship Prep Notes — Stat Arb / Cointegration / Mean Reversion / Backtesting

**Goal:** land a quant research / quant analyst internship by owning this project in interviews and on your resume/ATS.

**Primary story keywords:** statistical arbitrage · cointegration · mean reversion · pairs trading · walk-forward validation · backtesting · transaction costs · Sharpe ratio · hypothesis testing · time series · Python

Study order: **Resume/ATS → Elevator pitch → Definitions → Interview Q&A → Mistakes to avoid**.

Related docs:
- Plain overview: `docs/PROJECT_SUMMARY.md`
- Deeper technical write-up: `docs/RESEARCH_REPORT.md`

---

# 1) Resume & ATS (internship-focused)

## ATS / JD keyword bank (use naturally)

**Strategy / research:** statistical arbitrage, pairs trading, cointegration, Engle-Granger, mean reversion, spread, z-score, signal generation, alpha research  

**Validation / risk:** walk-forward validation, out-of-sample testing, look-ahead bias, transaction costs, slippage, Sharpe ratio, Sortino, max drawdown, hypothesis testing, Monte Carlo, multiple testing  

**Engineering:** Python, pandas, NumPy, statsmodels, vectorized backtesting, SQLite, Parquet, pytest, GitHub Actions, CLI  

**Soft signal words recruiters like:** end-to-end, research discipline, reproducible, cost-aware, out-of-sample

## Resume bullets (pick 3–4; use real numbers)

- Built a Python **statistical arbitrage** research engine screening **4,000+** Nasdaq-100/QQQ pairs for **Engle–Granger cointegration** and generating **z-score mean-reversion** signals on hedge-adjusted spreads.
- Implemented a cost-aware **backtester** with lagged execution (no look-ahead), transaction costs/slippage, and risk metrics (**Sharpe, Sortino, max drawdown**).
- Validated candidates with **walk-forward out-of-sample** windows and a **Monte Carlo position-shuffle** timing test; showed cointegration hits ≠ robust tradeable alpha after costs.
- Delivered reproducible research infra: Parquet caching, SQLite storage, pytest suite, and GitHub Actions CI.

## One-line resume project title options

- Statistical Arbitrage Pairs Trading Engine (Cointegration + Walk-Forward Backtest)
- QQQ Mean-Reversion Stat Arb Research Framework
- Cointegration Screening & Cost-Aware Pairs Backtester

## How to describe results without sounding like a failed project

> Identified statistically cointegrated QQQ pairs, then showed that a simple z-score mean-reversion rule mostly did **not** produce strong cost-aware out-of-sample Sharpe — demonstrating research rigor (falsifying a weak strategy carefully).

That framing is stronger for QR internships than “strategy returned +X%.”

---

# 2) Elevator pitches (memorize)

## 15 seconds
> I built a QQQ statistical-arbitrage pairs engine: screen for cointegration, trade mean reversion on the spread with z-scores, and validate with costs and walk-forward backtests.

## 30 seconds
> Using Nasdaq-100 names, I screened thousands of pairs for cointegration, then tested a classic mean-reversion pairs trade on the hedge-adjusted spread. After transaction costs and out-of-sample walk-forward tests, most pairs were not strong strategies. The research insight is that cointegration screening is easier than finding cost-aware alpha.

## 90 seconds
> I built an end-to-end Python research pipeline for statistical arbitrage pairs trading. I ingest QQQ/Nasdaq-100 prices, screen pairs with Engle–Granger cointegration, form a hedge-adjusted spread, and trade mean reversion with rolling z-scores. The backtest lags execution one day to avoid look-ahead and includes costs. Walk-forward keeps the hedge ratio out of the test window. Results: many pairs looked cointegrated, but simple z-score trading was mostly weak out of sample. A position-shuffle test suggested some timing content, but not clear economic edge. Next I’d add industry filters and stronger multiple-testing control.

## Fixed “what I did” (your explanation, corrected)
> We took ~100 QQQ stocks, cleaned for history, tested all pairs for **cointegration** (a mean-reverting hedge-adjusted **spread**, not just correlation), and when the spread z-score was unusually high/low we shorted/longed the spread. Then we checked whether that worked after **costs** and **out of sample**.

---

# 2b) Spread + z-score pipeline (detail — memorize this)

## Four steps (simple backtest / after β is known)

```text
1) Fit line once on ~3y (~756 trading days)  →  get β, α
2) Each day t:  S_t = A_t - β B_t - α        (residual / distance to that fixed line)
3) Each day t:  look at last 60 S values → mean μ_t, std σ_t → z_t
4) If |z_t| ≥ 2 → enter mean-reversion trade on the spread
```

### Step 1 — Fit the line (hedge ratio)

Over the cointegration / train window (about **756** days ≈ **3 years**), run OLS:

\[
P^A_t \approx \alpha + \beta P^B_t
\]

- \(\beta\) = slope = hedge ratio (“how many B per A”)
- \(\alpha\) = intercept
- This line is fit **once** for that window (not refit every 60 days in the current project)

### Step 2 — Daily spread (residual to the line)

For **every day** \(t\) (including the early days):

\[
S_t = P^A_t - \beta P^B_t - \alpha
\]

- \(S_t\) is already “how far today’s prices sit from the fitted line”
- Positive \(S\) ⇒ A is rich vs the hedged B relationship
- Negative \(S\) ⇒ A is cheap vs that relationship

### Step 3 — Rolling z-score (60-day window)

\[
\mu_t = \text{mean}(S_{t-59},\ldots,S_t), \quad
\sigma_t = \text{std}(S_{t-59},\ldots,S_t)
\]

\[
z_t = \frac{S_t - \mu_t}{\sigma_t}
\]

Config defaults:
- `zscore_window = 60`
- `entry_z = 2.0`
- `exit_z = 0.5`
- `stop_z = 4.0`

In our code, the rolling window **includes today** in the 60 points (standard pandas rolling).

### Step 4 — Trade rule on the spread (not “higher/lower stock price”)

If spread is \(S = A - \beta B - \alpha\):

| Condition | Meaning | Trade |
|-----------|---------|--------|
| \(z_t \ge +2\) | spread rich | **short spread**: short A, long \(\beta\) B |
| \(z_t \le -2\) | spread cheap | **long spread**: long A, short \(\beta\) B |
| \(\|z_t\| \le 0.5\) | back near normal | exit to flat |
| \(\|z\|\) blows past ~4 | possible break | stop out |

**Important wording fix:** you do **not** simply “long the cheaper stock and short the more expensive stock” by raw dollar price. You long/short relative to the **hedged relationship** (\(\beta\)).

Also: signal at close \(t\) → position earns returns from \(t \to t+1\) (1-day lag).

---

## Day-by-day intuition (your question — mostly yes)

Yes:

1. For the first **59** days of aligned prices you can still compute residuals \(S_1,S_2,\ldots\), but you **cannot** form a 60-day z-score yet → no trade signal (flat). Those early days are “warmup” for the z-window.
2. On **day 60**: first full window \(S_1,\ldots,S_{60}\) → compute \(\mu,\sigma\) → \(z_{60}\). If \(|z_{60}|\ge 2\), enter.
3. On **day 61**: window slides to \(S_2,\ldots,S_{61}\) → new \(z_{61}\), etc.
4. On **day 62**: window \(S_3,\ldots,S_{62}\), and so on.

So your sliding-window picture is right for the **z-score**.  
The **line** (\(\beta,\alpha\)) stays fixed while that window slides (in the simple / screened-β backtest).

### Tiny corrections to your wording
- “Residual = price A − slope×price B − intercept” → **yes**, that’s \(S_t\).
- “Compare to days 1–60 on day 61” → on day 61 it’s days **2–61** (sliding), not 1–60.
- “Long lower / short higher” → better: **long/short the spread** using \(\beta\) as in the table above.

---

## Two different windows (don’t mix them)

| Window | Length | What it does |
|--------|--------|----------------|
| Fit / cointegration window | ~**756** days (~3y) | estimate \(\beta,\alpha\); test cointegration |
| Z-score window | **60** days | decide if *today’s* \(S_t\) is unusual vs recent \(S\) |

---

## Walk-forward version (same idea, safer β)

1. Fit \(\beta,\alpha\) on a **train** ~756d window only  
2. On the following **test** window, compute \(S_t\) with that **frozen** β  
3. Still use rolling 60d z-scores on \(S\) (often warm up z using the end of train so test days aren’t blank)  
4. Trade only on test days; then roll the windows forward  

---

# 3) Core definitions (interview vocabulary)

## Statistical arbitrage (stat arb)
- Systematic strategies that try to capture small, repeated edges from statistical relationships (here: mean-reverting spreads), often market-neutral-ish, not directional stock-picking.

## Pairs trading
- Trade two related names against each other: long one, short the other, betting the relationship normalizes.

## Random walk / I(1) prices
- Price ≈ yesterday + noise; levels wander; no fixed “fair price” by themselves.

## Stationarity
- Series whose typical mean/variance don’t wander off permanently.
- We want the **spread** closer to stationary.

## Correlation
- Do returns/moves tend to go together? Range roughly \([-1,1]\).
- **Not enough** for pairs trading: highly correlated stocks can still permanently drift apart.

## Cointegration
- Two wandering prices share a long-run equilibrium: a linear combination (the **spread**) mean-reverts.
- Dog-and-leash intuition: both can wander, but the **distance** doesn’t run away forever.  
  **Fix:** they don’t have to end at the *same place*; the leash bounds the gap.

## Spread
\[
S_t \approx P^A_t - \beta P^B_t - \alpha
\]
- The object we trade for mean reversion.

## Hedge ratio \(\beta\)
- From regressing A on B: how many units of B offset one unit of A in the spread.
- Example: \(\beta=1.35\) ⇒ spread \(\approx A - 1.35B\). Short spread ⇒ short A, long ~1.35 B.

## Mean reversion
- When something stretched away from its typical level tends to come back (here: the spread).

## Z-score
\[
z_t = \frac{S_t - \mu_t}{\sigma_t}
\]
- \(|z|\) large ⇒ unusually rich/cheap spread ⇒ enter mean-reversion trade.
- Near 0 ⇒ exit; extreme blowout ⇒ stop (relationship may have broken).

## Engle–Granger cointegration test
1. Regress A on B → \(\hat\beta,\hat\alpha\)
2. Test whether residuals look stationary
3. Low p-value ⇒ evidence of cointegration

## Look-ahead bias
- Using info you wouldn’t have had at decision time.
- Our fix: signal at close \(t\) earns returns from \(t \to t+1\).

## Walk-forward validation
- Fit on a train window (e.g. estimate \(\beta\)) → trade only on later test window → roll forward → stitch OOS results.

## Sharpe ratio
- Risk-adjusted excess return over a risk-free rate, scaled by volatility.
- Positive total PnL can still mean negative Sharpe if you don’t beat cash enough for the risk (and if capital is often idle).

## Multiple testing
- Test thousands of pairs at 5% ⇒ expect many lucky false positives.
- Remedies: stricter p-values / FDR, economic/same-industry filters, OOS trading tests.
- **Wrong wording:** “do multiple tests to eliminate false positives.”  
- **Right:** many tests *create* false positives; use corrections/filters.

## Monte Carlo timing test
- **Wrong null:** shuffle strategy returns (mean/vol unchanged ⇒ Sharpe unchanged).
- **Right null:** shuffle **positions** vs fixed spread path ⇒ tests timing skill.

## Economic significance vs statistical significance
- Timing can beat random and still not be a good strategy after costs/risk/capacity.

---

# 4) What we scanned / built / found (talk track)

## Scanned for
- Cointegrated QQQ pairs (Engle–Granger), with correlation as a cheap pre-filter only.

## Built
- Data pipeline → cointegration screen → z-score mean-reversion signals → cost-aware backtest → walk-forward + Monte Carlo.

## Found (first full run talking points)
- ~94 names after cleaning; ~4,300+ pairs screened; ~172 EG passes @5%; top ~40 kept.
- Sensible: **ADI–KLAC** (semis), **DDOG–PANW** (software), **AMZN–NVDA** (mega-cap / AI-compute story).
- Suspicious: **GILD–XEL** (pharma vs utility) — statistical hit, weak economic story.
- After costs + walk-forward: most pairs weak; best inspected OOS ~ **DDOG–PANW Sharpe ≈ 0.15** (modest).
- Insight: **cointegration ≠ alpha**.

---

# 5) Interview Q&A bank (polished answers)

Practice out loud. These are the cleaned answers from our mock interview.

### Q1. What does this project do?
> Using Nasdaq-100/QQQ stocks, I screen pairs for cointegration, then test a mean-reversion trade on the hedge-adjusted spread with z-scores. The goal is to see whether that statistical-arbitrage idea works after costs and out of sample — not just whether pairs look cointegrated.

### Q2. Correlation vs cointegration? Why does pairs trading care?
> Correlation: do they tend to move together? Cointegration: does a hedge-adjusted spread mean-revert around an equilibrium? Pairs trading needs cointegration of the spread. Correlation alone is not enough because prices can still permanently drift apart. In this project, correlation was only a pre-filter.

### Q3. Why aren’t ~170 cointegrated pairs ~170 strategies?
> A cointegration hit is a candidate, not a strategy. With thousands of tests at 5%, some passes are luck (multiple testing). Even real relationships can be too noisy/slow, fail after costs, lack an economic link, or break out of sample. Walk-forward showed most were weak.

### Q4. z-score = +2.5 — what do you do?
> If spread \(S=A-\beta B\) is rich (\(z=+2.5\)), short the spread: short A, long \(\beta B\). We don’t need prices to become equal — we need the spread to mean-revert.

### Q5. Did the project fail?
> The trading hypothesis mostly failed; the research succeeded. Simple z-score pairs trading on these QQQ cointegrated pairs was not a strong cost-aware OOS strategy. Finding cointegration was easier than finding durable alpha.

### Q6. What is \(\beta\)?
> Hedge ratio from regressing A on B: how many units of B per unit of A when forming the spread. If \(\beta=1.35\), spread \(\approx A-1.35B\).

### Q7. Why lag positions one day?
> To avoid look-ahead bias. The close-based signal is known at today’s close, so PnL should start from today→tomorrow.

### Q8. Positive return but negative Sharpe — how?
> Sharpe uses excess return over the risk-free rate divided by volatility. You can make money overall and still have negative Sharpe if excess return is weak vs cash, especially if capital is often idle and only partly deployed.

### Q9. Monte Carlo: why not shuffle returns?
> Mean and volatility are basically unchanged by shuffling returns, so Sharpe barely changes. Shuffle positions against the same market path to test timing skill.

### Q10. Multiple testing problem?
> Thousands of pairs at 5% significance produce expected false positives. Remedies: stricter p-values/FDR, economic filters, OOS validation.

### Q11. Sensible vs suspicious pair?
> Sensible: AMZN–NVDA (mega-cap / AI-compute linked) or ADI–KLAC (semis). Suspicious: GILD–XEL (pharma vs utility) — weak economic reason for a stable equilibrium even if p-value is tiny.

### Q12. What would you improve next?
> Add multiple-testing control and same-industry/economic filters on the screen, then re-run walk-forward on the cleaner set before tuning z-score parameters.

### Q13. Why walk-forward instead of one train/test split?
> Markets change; one split can be a lucky regime. Rolling windows stress-test whether the idea holds across periods with β estimated only in-sample.

### Q14. How do you know you didn’t curve-fit?
> Lagged execution, costs, walk-forward β, and not selecting only the prettiest in-sample chart. Still imperfect (raw screen has multiple-testing exposure), which I’d address next.

---

# 6) Mistakes from practice (don’t repeat)

| Mistake | Better |
|--------|--------|
| “Correlation is short-term, cointegration long-term” | Different concepts; not mainly a horizon split |
| “They end in the same place” | Spread mean-reverts; leash bounds the gap |
| “We wanted a spread between stocks” | Every pair has a gap; need **mean-reverting/cointegrated** spread |
| “Do multiple tests to eliminate false positives” | Many tests *create* false positives |
| “Top 40 because strongest signals” | Still luck/costs/OOS/economic-link issues |
| “Project failed” | Hypothesis failed; research succeeded |
| Vague “short high stock / long low stock” | Short/long the **spread** using \(\beta\) |
| Hedge ratio = “risk protected vs position size” | Units of B per A in the spread hedge |

---

# 7) Three lines to recite until automatic

1. **Correlation ≠ cointegration.**  
2. **Cointegration ≠ alpha.**  
3. **Signal at \(t\), PnL from \(t \to t+1\).**

---

# 8) “What could you have done better?” / “What would you explore next?”

These are very common. Structure: **admit a real limitation → say why it matters → say the concrete next step.** Don’t trash the whole project.

## Short answer (30 seconds) — memorize

> The biggest gap was discovery control: I tested thousands of pairs at 5%, so some cointegration hits were likely false positives, and I didn’t hard-filter for economic/same-industry links. Next I’d add FDR or stricter p-values plus an industry filter, then re-run walk-forward only on that cleaner set before tuning signal parameters.

## What I could have done better (pick 2–3 in an interview)

### 1) Multiple-testing control on the screen *(highest leverage)*
- **Gap:** ~4,300 pairs @ 5% ⇒ expect many lucky passes.
- **Better:** Bonferroni / FDR, or a much stricter p-value cutoff.
- **Why it matters:** otherwise “172 cointegrated pairs” overstates how many relationships are real.

### 2) Economic / same-industry filter before trading
- **Gap:** statistical pairs like GILD–XEL can pass with weak business reason.
- **Better:** only trade within sector/industry (e.g. semis with semis).
- **Why it matters:** desks want a reason the equilibrium should *persist*, not just a p-value.

### 3) Don’t tune or rank only on in-sample cointegration p-values
- **Gap:** top-40 by EG p-value is not the same as top-40 by OOS tradability.
- **Better:** rank candidates by walk-forward metrics after costs.
- **Why it matters:** cointegration strength ≠ trading edge.

### 4) Hedge ratio & signal assumptions were simple
- **Gap:** fixed/train-window β; fixed z-entry/exit; daily bars only.
- **Better:** rolling β, sensitivity analysis on z thresholds, maybe Johansen for >2 assets later.
- **Why it matters:** relationships and volatility regimes change.

### 5) Cost / shorting model was stylized
- **Gap:** flat bps costs + crude borrow.
- **Better:** name-specific borrow, bid-ask by liquidity, reject hard-to-short names.
- **Why it matters:** pairs need shorts; borrow can kill the edge.

### 6) Research engineering bugs I actually hit *(great honesty story)*
- Global inner-join across all tickers crushed history → fixed with pairwise alignment.
- First Monte Carlo shuffled returns (useless for Sharpe) → fixed by shuffling positions.
- **Say this:** “I improved the methodology when I realized the null was wrong.”

## What I should explore next (future work — prioritized)

**Near-term (best interview answer):**
1. Same-industry universe + FDR / stricter discovery thresholds  
2. Re-run walk-forward / significance only on that filtered set  
3. Report whether OOS Sharpe improves or stays weak (either result is publishable research)

**Medium-term:**
4. Rolling hedge ratio; parameter sensitivity (entry_z, window) without overfitting  
5. Multi-pair portfolio with simple risk limits / risk-parity sizing  
6. Regime filter (pause in high-vol regimes)

**Stretch / advanced:**
7. Intraday data / tighter execution assumptions  
8. Johansen cointegration / small baskets (not just pairs)  
9. Paper-trading / live signal monitoring (process, not just backtest)

## Phrases that sound internship-ready

> “If I had another two weeks, I wouldn’t add complexity first — I’d tighten discovery and economic filters, then re-validate.”

> “A lot of undergrad projects stop at finding cointegration. I’d rather show I know what would make the research desk-quality.”

> “The result might still be ‘no strong edge’ after cleaner filters — that would still be a useful conclusion.”

## Weak answers to avoid

- “I would use machine learning / deep learning on prices” (sounds like avoiding the real issues)
- “I would get more data and optimize more parameters” (overfitting smell)
- “Nothing — it was perfect” (not credible)
- Listing 10 future ideas with no priority (sounds unfocused)

---

# 9) Internship behavioral hooks (use this project)

**“Tell me about a time your initial approach was wrong.”**  
> First cointegration screen looked exciting (~170 pairs). After costs and walk-forward, most edges disappeared. I also had to fix a data bug (global date join crushed history) and a bad Monte Carlo null (shuffling returns doesn’t test Sharpe).

**“How do you know a strategy works?”**  
> Not in-sample PnL. Out-of-sample validation, costs, bias checks, and skepticism about multiple testing.

**“Why quant research?”**  
> I like turning a market idea into a testable hypothesis and trying to break it. This project is exactly that workflow.

**“What would you do differently?”**  
> Use the short answer in section 8: discovery control + industry filter, then re-run OOS before adding fancy features.

---

# 10) Quick self-test (no notes)

1. Elevator pitch (30s)  
2. Correlation vs cointegration  
3. What \(\beta\) means with 1.35 example  
4. z=+2.5 trade  
5. Why lag one day  
6. Why ~170 ≠ 170 strategies  
7. Multiple testing in one sentence  
8. Research insight / did it fail?  
9. One sensible pair + one suspicious pair  
10. What you’d do next / what you could have done better  

If you can answer those cleanly, you’re interview-ready on this project.
