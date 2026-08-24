"""
CLI entry point.

Examples:
    python run.py --ticker AAPL --strategy zscore --start 2019-01-01
    python run.py --ticker QQQ  --strategy trend   --start 2019-01-01 --window 30
    python run.py --ticker SPY  --strategy momentum --start 2019-01-01 --lookback 15

Run --help for all options. See signals.py to add/adapt strategies, and
custom_signal_template() there for plugging in your own exact ThinkScript
logic once translated to pandas.
"""

from __future__ import annotations

import argparse

from backtest import run_backtest
from data import load_prices
from signals import nday_momentum, regression_trend_momentum, zscore_mean_reversion

STRATEGIES = {
    "zscore": zscore_mean_reversion,
    "trend": regression_trend_momentum,
    "momentum": nday_momentum,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest a signal against a ticker.")
    parser.add_argument("--ticker", required=True, help="e.g. AAPL, QQQ, SPY")
    parser.add_argument("--strategy", choices=STRATEGIES.keys(), required=True)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--window", type=int, default=20, help="zscore/trend window")
    parser.add_argument("--entry-z", type=float, default=1.5)
    parser.add_argument("--exit-z", type=float, default=0.3)
    parser.add_argument("--lookback", type=int, default=10, help="momentum lookback")
    parser.add_argument("--cost-bps", type=float, default=5.0)
    parser.add_argument("--slippage-bps", type=float, default=2.0)
    args = parser.parse_args()

    prices = load_prices(args.ticker, start=args.start, end=args.end)
    close = prices["Close"]

    if args.strategy == "zscore":
        position = zscore_mean_reversion(close, window=args.window, entry_z=args.entry_z, exit_z=args.exit_z)
    elif args.strategy == "trend":
        position = regression_trend_momentum(close, window=args.window)
    else:
        position = nday_momentum(close, lookback=args.lookback)

    result = run_backtest(close, position, cost_bps=args.cost_bps, slippage_bps=args.slippage_bps)

    print(f"\n{args.ticker} | strategy={args.strategy} | {args.start} -> {args.end or 'latest'}")
    print("-" * 60)
    print(result.summary())


if __name__ == "__main__":
    main()
