# Options Lab

Vanilla Black–Scholes first, then the actual project: **simulate a 1x2 call ratio on 100 shares**, including assignment (stock called away).

## `pricer.cpp`

1. Closed-form BS call/put, Greeks, put-call parity.
2. Monte Carlo **vanilla** call: draw \(S_T\) at expiry, \(\max(S_T-K,0)\), \(N=10^6\). Check vs closed form (should be \(\sim 0.02\)).
3. **1x2 on 100 shares:** \(+100\) shares, \(+1\) ATM call, \(-2\) calls 5% OTM.
   - Physical settlement at expiry:
     - \(S_T > K_1\): exercise the long call → **+100 shares**, pay \(K_1\).
     - \(S_T > K_2\): assigned on both shorts → **−200 shares**, receive \(2K_2\).
     - Terminal shares: 100 / 200 / **0**. Above \(K_2\) the stock is called away.
   - MC: P(exercise), P(called away), E[shares], option-package value vs BS.
   - Print a small **value surface** (spot × vol).

`make && ./pricer`

## `smile.py`

Same structure, with plots.

- `smile.png` — OTM SPY (or `--demo`) IV smile; ATM IV feeds the sim.
- `ratio_spread.png` — expiry P&L of 100 shares + 1x2, shaded: long exercised vs **stock called away**.
- `surface.png` — **ratio mark** vs spot and vol, not a raw IV surface.

`python smile.py` or `python smile.py --demo`

## Interview one-liner

“I built a BS pricer, then used expiry MC to simulate a 1x2 call ratio sitting on 100 shares — when the short strikes finish ITM you get assigned on 200 calls and the stock is called away.”

Do not say American early exercise. This is **European expiry assignment** only.
