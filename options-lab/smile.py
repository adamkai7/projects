#!/usr/bin/env python3
"""IV smile + surface from a live SPY chain or synthetic data)."""

from __future__ import annotations

import argparse
import math
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq #root finding


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

R = 0.05 #risk free rate
Q = 0.01 #dividend yield


def norm_pdf(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def bs_price(S, K, T, sigma, r=R, q=Q, call=True):
    if T <= 0 or sigma <= 0:
        intrinsic = max(S * math.exp(-q * T) - K * math.exp(-r * T), 0.0)
        return intrinsic if call else max(K * math.exp(-r * T) - S * math.exp(-q * T), 0.0)

    #Black-Scholes formula
    v = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    if call:
        #call option price
        return disc_S * norm_cdf(d1) - disc_K * norm_cdf(d2)
    #put option price
    return disc_K * norm_cdf(-d2) - disc_S * norm_cdf(-d1)


def bs_greeks(S, K, T, sigma, r=R, q=Q, call=True):
    """Delta, gamma, vega (per 1.0 vol), theta (per year). Same formulas as pricer.cpp."""
    sqrtT = math.sqrt(T)
    v = sigma * sqrtT
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / v
    d2 = d1 - v
    disc_S = S * math.exp(-q * T)
    disc_K = K * math.exp(-r * T)
    n1 = norm_pdf(d1)
    gamma = math.exp(-q * T) * n1 / (S * sigma * sqrtT)
    vega = disc_S * n1 * sqrtT
    if call:
        delta = math.exp(-q * T) * norm_cdf(d1)
        theta = -disc_S * n1 * sigma / (2.0 * sqrtT) - r * disc_K * norm_cdf(d2) + q * disc_S * norm_cdf(d1)
    else:
        delta = -math.exp(-q * T) * norm_cdf(-d1)
        theta = -disc_S * n1 * sigma / (2.0 * sqrtT) + r * disc_K * norm_cdf(-d2) - q * disc_S * norm_cdf(-d1)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_vol(mid, S, K, T, call=True):
    intrinsic = max(S * math.exp(-Q * T) - K * math.exp(-R * T), 0.0) if call else max(
        K * math.exp(-R * T) - S * math.exp(-Q * T), 0.0
    )
    #if the mid price is less than the intrinsic value, return nan
    if mid <= intrinsic + 1e-8:
        return np.nan

    def objective(sig):
        return bs_price(S, K, T, sig, call=call) - mid
    #try to find the implied volatility using the brentq root finding algorithm
    lo, hi = 1e-4, 3.0
    if objective(lo) * objective(hi) > 0:
        return np.nan
    return brentq(objective, lo, hi, maxiter=50)


def years_to_expiry(expiry_str):
    exp = datetime.strptime(expiry_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    return max((exp - now).total_seconds() / (365.0 * 24 * 3600), 1.0 / 365.0)


def clean_chain(calls, puts, S):
    frames = []
    for df, is_call in ((calls, True), (puts, False)):
        d = df.copy()
        d = d[(d["bid"] > 0) & (d["ask"] > d["bid"])]
        d["mid"] = 0.5 * (d["bid"] + d["ask"])
        d["is_call"] = is_call
        # OTM only: puts below spot, calls above spot
        d = d[d["strike"] < S] if not is_call else d[d["strike"] > S]
        frames.append(d[["strike", "mid", "is_call"]])
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["k"] = np.log(out["strike"] / S)
    return out


def demo_points(S=600.0):
    """Synthetic smiles so plots still generate with no market data."""
    rows = []
    for T in (30 / 365, 60 / 365, 90 / 365, 180 / 365):
        ks = np.linspace(-0.25, 0.25, 21)
        for k in ks:
            K = S * math.exp(k)
            sigma = 0.18 + 0.08 * k + 0.60 * k * k  # mild smirk
            call = k >= 0
            mid = bs_price(S, K, T, sigma, call=call)
            rows.append({"T": T, "strike": K, "k": k, "mid": mid, "is_call": call, "iv": sigma})
    return S, pd.DataFrame(rows)


def live_points(ticker="SPY", min_dte=21, n_expiries=5):
    import yfinance as yf

    t = yf.Ticker(ticker)
    hist = t.history(period="5d")
    if hist.empty:
        raise RuntimeError("yfinance returned no spot history")
    S = float(hist["Close"].iloc[-1])

    kept = []
    for exp in t.options:
        T = years_to_expiry(exp)
        if T * 365 < min_dte:
            continue
        chain = t.option_chain(exp)
        df = clean_chain(chain.calls, chain.puts, S)
        if df.empty:
            continue
        df["T"] = T
        ivs = [
            implied_vol(row.mid, S, row.strike, T, call=bool(row.is_call))
            for row in df.itertuples(index=False)
        ]
        df["iv"] = ivs
        df = df.dropna(subset=["iv"])
        df = df[(df["iv"] > 0.03) & (df["iv"] < 0.80) & (df["k"].abs() < 0.30)]
        if len(df) >= 8:
            kept.append(df)
        if len(kept) >= n_expiries:
            break

    if not kept:
        raise RuntimeError("no usable expiries after cleaning")
    return S, pd.concat(kept, ignore_index=True)


def fit_quadratic(k, iv):
    coef = np.polyfit(k, iv, 2)
    pred = np.polyval(coef, k)
    rmse = float(np.sqrt(np.mean((pred - iv) ** 2)))
    return coef, rmse


def plot_smile(df, S, path="smile.png", label="SPY"):
    T0 = df["T"].min()
    slice_ = df[np.isclose(df["T"], T0)].sort_values("k")
    coef, rmse = fit_quadratic(slice_["k"].to_numpy(), slice_["iv"].to_numpy())
    grid = np.linspace(slice_["k"].min(), slice_["k"].max(), 200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(slice_["k"], slice_["iv"] * 100, s=18, label="OTM market IV")
    ax.plot(grid, np.polyval(coef, grid) * 100, color="C1", label="quadratic fit")
    ax.set_xlabel("log-moneyness  ln(K/S)")
    ax.set_ylabel("implied vol (%)")
    ax.set_title(f"{label} smile  S={S:.2f}  T={T0 * 365:.0f}d  RMSE={rmse * 100:.2f} vol pts")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return rmse, T0


def terminal_shares(ST, K1, K2, shares0=100.0):
    """Physical settlement: exercise long call (+100), assigned on 2 shorts (-200)."""
    sh = np.full_like(ST, shares0, dtype=float)
    sh = sh + 100.0 * (ST > K1) - 200.0 * (ST > K2)
    return sh


def simulate_ratio(S, K1, K2, T, sigma, n=200_000, r=R, q=Q, seed=42):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * np.sqrt(T) * z)
    shares = terminal_shares(ST, K1, K2)
    opt_pay = np.maximum(ST - K1, 0.0) - 2.0 * np.maximum(ST - K2, 0.0)
    return ST, shares, opt_pay


def plot_ratio_surface(S0, K1, K2, T, path="surface.png"):
    """BS mark of the 1x2 option package vs spot and vol (the 'surface')."""
    spots = np.linspace(0.85 * S0, 1.15 * S0, 41)
    vols = np.linspace(0.10, 0.40, 31)
    Z = np.zeros((len(vols), len(spots)))
    for i, sig in enumerate(vols):
        for j, S in enumerate(spots):
            Z[i, j] = bs_price(S, K1, T, sig, call=True) - 2.0 * bs_price(S, K2, T, sig, call=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(
        Z,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        extent=[spots[0], spots[-1], vols[0] * 100, vols[-1] * 100],
    )
    ax.set_xlabel("spot")
    ax.set_ylabel("vol (%)")
    ax.set_title(f"1x2 call ratio mark  +1x{K1:.0f} / -2x{K2:.0f}  T={T*365:.0f}d")
    fig.colorbar(im, ax=ax, label="option package value")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_ratio_mc(S, T, sigma, path="ratio_spread.png"):
    """Expiry P&L of +100 shares +1 ATM call -2 OTM calls, plus assignment MC."""
    K1, K2 = S, S * 1.05
    long = bs_greeks(S, K1, T, sigma, call=True)
    short = bs_greeks(S, K2, T, sigma, call=True)
    net = {k: long[k] - 2.0 * short[k] for k in long}
    debit = bs_price(S, K1, T, sigma, call=True) - 2.0 * bs_price(S, K2, T, sigma, call=True)
    disc = math.exp(-R * T)

    ST, shares, opt_pay = simulate_ratio(S, K1, K2, T, sigma)
    mc_val = disc * float(opt_pay.mean())
    p_ex = float((ST > K1).mean())
    p_away = float((ST > K2).mean())
    e_sh = float(shares.mean())

    spots = np.linspace(0.80 * S, 1.25 * S, 400)
    stock_pnl = 100.0 * (spots - S)
    opt_pnl = 100.0 * ((np.maximum(spots - K1, 0.0) - 2.0 * np.maximum(spots - K2, 0.0)) - debit)
    total = stock_pnl + opt_pnl
    sh_line = terminal_shares(spots, K1, K2)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axvspan(K2, spots[-1], color="C3", alpha=0.12, label="stock called away (assigned on 2 shorts)")
    ax.axvspan(K1, K2, color="C2", alpha=0.10, label="long call exercised, keep extra shares")
    ax.plot(spots, total, color="C0", label="P&L: 100 sh + 1x2")
    ax.axhline(0.0, color="k", linewidth=0.8)
    ax.axvline(S, color="k", linestyle="--", linewidth=0.8)
    ax.set_xlabel("spot at expiry")
    ax.set_ylabel("P&L")
    ax.set_title(
        f"1x2 on 100 sh  K={K1:.0f}/{K2:.0f}  P(called away)={p_away:.1%}  "
        f"E[sh]={e_sh:.0f}  Δ={net['delta']+1:.2f}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(
        f"1x2 +100sh  debit={debit:.3f}  BS pkg={debit:.3f}  MC pkg={mc_val:.3f}  "
        f"delta={net['delta']:.3f} (with stock {net['delta']+1:.3f})  "
        f"P(exercise)={p_ex:.3f}  P(stock called away)={p_away:.3f}  "
        f"E[shares]={e_sh:.1f}"
    )
    _ = sh_line  # used conceptually on the plot via regions
    return K1, K2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", action="store_true", help="synthetic smile, no yfinance")
    p.add_argument("--ticker", default="SPY")
    args = p.parse_args()

    if args.demo:
        S, df = demo_points()
        print("demo mode: synthetic smirk around 18% ATM")
    else:
        S, df = live_points(args.ticker)
        print(f"spot {args.ticker} = {S:.2f}  contracts={len(df)}  expiries={df['T'].nunique()}")

    rmse, T0 = plot_smile(df, S, label="demo" if args.demo else args.ticker)
    T = max(T0, 21 / 365)
    atm_iv = float(df.loc[df["k"].abs().idxmin(), "iv"])
    K1, K2 = plot_ratio_mc(S, T, atm_iv)
    plot_ratio_surface(S, K1, K2, T)
    print(f"nearest expiry T={T0 * 365:.0f}d  smile RMSE={rmse * 100:.2f} vol pts  ATM IV={atm_iv*100:.1f}%")
    print("wrote smile.png, surface.png (ratio mark), ratio_spread.png (expiry P&L + assignment)")


if __name__ == "__main__":
    main()
