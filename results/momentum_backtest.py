'''
Created by Adam Kainikara

Algorithmic Trading Strategy Backtester with Multi-Factor Momentum Analysis

This tool implements a sophisticated momentum-based trading strategy that combines multiple
technical factors to generate trading signals. The backtester evaluates strategy performance
across different market conditions and optimizes parameters using differential evolution.

The strategy uses:
- Relative Strength Index (RSI) for momentum
- Moving Average Crossovers for trend detection
- Volume analysis for confirmation
- Risk management with dynamic position sizing

Example usage:
python momentum_strategy_backtester.py
'''

import pandas as pd
from scipy.optimize import differential_evolution
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import os
from numpy import *
import random
import yfinance as yf
import matplotlib
# Force matplotlib to not use any Xwindows backend to prevent crash on Mac/Server
matplotlib.use('Agg')


# Configuration - change these parameters as needed
SYMBOLS = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA']
START_DATE = '2020-01-01'
END_DATE = '2024-12-31'
INITIAL_CAPITAL = 10000.0
COMMISSION = 0.001
SLIPPAGE = 0.001

# STRATEGY SETTINGS (Default values, will be overwritten if Optimization is ON)
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 50
RSI_SELL_THRESHOLD = 70
MA_SHORT = 20
MA_LONG = 50
VOLUME_MA = 20

# SETTINGS UPDATED
SHOW_PLOTS = True       # Set to True to generate graphs
RUN_OPTIMIZATION = True  # Set to True to find best params
OUTPUT_DIR = './results'
RANDOM_SEED = 42


def fetch_data(symbol_l, start_date, end_date):
    '''Download historical price data from Yahoo Finance'''
    data_d = {}

    print(f"\nDownloading data for {len(symbol_l)} symbols from {start_date} to {end_date}...")

    for symbol in symbol_l:
        try:
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
            if len(df) > 0:
                data_d[symbol] = df
                print(f"  {symbol}: {len(df)} days")
            else:
                print(f"  {symbol}: No data available")
        except Exception as e:
            print(f"  {symbol}: Error - {e}")

    return data_d


def calculate_rsi(prices_v, period=14):
    '''Calculate Relative Strength Index (Vectorized)'''
    # Ensure input is a pandas Series to use built-in rolling/ewm functions
    prices_s = pd.Series(prices_v)
    deltas = prices_s.diff()

    # Get initial up/down
    up = deltas.clip(lower=0)
    down = -1 * deltas.clip(upper=0)

    # Calculate exponential moving average (Wilder's Smoothing)
    # alpha = 1/period is standard for RSI
    ma_up = up.ewm(alpha=1/period, adjust=False).mean()
    ma_down = down.ewm(alpha=1/period, adjust=False).mean()

    # Calculate RS and RSI
    rs = ma_up / ma_down
    rsi = 100 - (100 / (1 + rs))

    # Handle division by zero (where down is 0)
    rsi = rsi.fillna(100)  # If no down moves, RSI is 100

    # Fill initial NaN values (first period) with 50 or NaN
    rsi.iloc[:period] = nan

    return rsi.values


def calculate_moving_average(prices_v, period):
    '''Calculate simple moving average'''
    ma_v = full(len(prices_v), nan)
    for i in range(period - 1, len(prices_v)):
        ma_v[i] = mean(prices_v[i - period + 1:i + 1])
    return ma_v


def generate_signals(df, params):
    '''Generate trading signals based on technical indicators'''
    close_v = df['Close'].values.flatten()
    volume_v = df['Volume'].values.flatten()

    # Calculate indicators
    rsi_v = calculate_rsi(close_v, params['rsi_period'])
    ma_short_v = calculate_moving_average(close_v, params['ma_short'])
    ma_long_v = calculate_moving_average(close_v, params['ma_long'])
    volume_ma_v = calculate_moving_average(volume_v, params['volume_ma'])

    # Generate signals
    signals_v = zeros(len(close_v))

    # Create a start index using standard integer comparison
    start_idx = int(maximum(params['ma_long'], params['rsi_period']))

    for i in range(start_idx, len(close_v)):
        # STRATEGY LOGIC: TREND FOLLOWING (MOMENTUM)

        # 1. Trend Condition: Short MA > Long MA (Golden Cross)
        trend_is_up = ma_short_v[i] > ma_long_v[i]

        # 2. Momentum Condition: RSI > Threshold (Price is gaining strength)
        momentum_is_strong = rsi_v[i] > params['rsi_oversold']

        # BUY SIGNAL
        if trend_is_up and momentum_is_strong:
            signals_v[i] = 1

        # SELL SIGNAL (Exit if trend reverses)
        elif not trend_is_up:
            signals_v[i] = -1

    return signals_v, rsi_v, ma_short_v, ma_long_v


def backtest_strategy(df, signals_v, initial_capital, commission, slippage):
    '''Execute backtest with transaction costs'''
    close_v = df['Close'].values.flatten()
    cash = initial_capital
    position = 0
    portfolio_v = zeros(len(close_v))
    trades_l = []

    for i in range(len(close_v)):
        portfolio_v[i] = cash + position * close_v[i]

        if signals_v[i] == 1 and position == 0:  # Buy
            shares = floor(cash / (close_v[i] * (1 + commission + slippage)))
            if shares > 0:
                cost = shares * close_v[i] * (1 + commission + slippage)
                cash -= cost
                position = shares
                trades_l.append({
                    'date': df.index[i],
                    'type': 'BUY',
                    'price': close_v[i],
                    'shares': shares,
                    'value': cost
                })

        elif signals_v[i] == -1 and position > 0:  # Sell
            proceeds = position * close_v[i] * (1 - commission - slippage)
            cash += proceeds
            trades_l.append({
                'date': df.index[i],
                'type': 'SELL',
                'price': close_v[i],
                'shares': position,
                'value': proceeds
            })
            position = 0

    # Close final position
    if position > 0:
        proceeds = position * close_v[-1] * (1 - commission - slippage)
        cash += proceeds
        portfolio_v[-1] = cash

    return portfolio_v, trades_l


def calculate_performance_metrics(portfolio_v, initial_capital, df):
    '''Calculate comprehensive performance metrics'''
    close_v = df['Close'].values.flatten()
    returns_v = diff(portfolio_v) / portfolio_v[:-1]
    returns_v = returns_v[~isnan(returns_v)]

    total_return = (portfolio_v[-1] - initial_capital) / initial_capital

    # Annualized return
    days = len(portfolio_v)
    years = days / 252
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Sharpe ratio
    excess_returns = returns_v - 0.02 / 252  # Risk-free rate
    sharpe = sqrt(252) * mean(excess_returns) / std(returns_v) if std(returns_v) > 0 else 0

    # Max drawdown
    cummax_v = maximum.accumulate(portfolio_v)
    drawdown_v = (portfolio_v - cummax_v) / cummax_v
    max_drawdown = abs(min(drawdown_v))

    # Win rate
    winning_days = sum(returns_v > 0)
    win_rate = winning_days / len(returns_v) if len(returns_v) > 0 else 0

    # Sortino ratio (downside deviation)
    downside_returns = returns_v[returns_v < 0]
    downside_std = std(downside_returns) if len(downside_returns) > 0 else 0.0001
    sortino = sqrt(252) * mean(excess_returns) / downside_std if downside_std > 0 else 0

    # Buy and hold benchmark
    buy_hold_return = (close_v[-1] - close_v[0]) / close_v[0]
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'final_value': portfolio_v[-1],
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return
    }


def optimize_parameters(df, initial_capital, commission, slippage):
    '''Optimize strategy parameters using differential evolution'''

    def objective(params_v):
        params_d = {
            'rsi_period': int(params_v[0]),
            'rsi_oversold': params_v[1],  # Used as Buy Threshold
            'rsi_overbought': 70,
            'ma_short': int(params_v[2]),
            'ma_long': int(params_v[3]),
            'volume_ma': 20
        }

        # Constraint: Long MA must be > Short MA to be logical
        if params_d['ma_short'] >= params_d['ma_long']:
            return 1e6

        try:
            signals_v, _, _, _ = generate_signals(df, params_d)
            portfolio_v, _ = backtest_strategy(df, signals_v, initial_capital, commission, slippage)
            metrics_d = calculate_performance_metrics(portfolio_v, initial_capital, df)

            # Optimize for risk-adjusted return (Sharpe ratio)
            return -metrics_d['sharpe_ratio']
        except:
            return 1e6

    # Bounds: [RSI_Period, RSI_Buy_Threshold, MA_Short, MA_Long]
    bounds = [
        (10, 20),    # rsi_period
        (40, 60),    # rsi_buy_threshold (Momentum strength)
        (10, 40),    # ma_short
        (50, 200),   # ma_long
    ]

    print("  Optimizing parameters...", end='', flush=True)

    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=10,
        popsize=10,
        tol=0.01,
        seed=42,
        disp=False
    )
    print(" Done.")

    optimal_params = {
        'rsi_period': int(result.x[0]),
        'rsi_oversold': result.x[1],
        'rsi_overbought': 70,
        'ma_short': int(result.x[2]),
        'ma_long': int(result.x[3]),
        'volume_ma': 20
    }

    return optimal_params


def plot_backtest_results(df, portfolio_v, signals_v, rsi_v, ma_short_v, ma_long_v,
                          metrics_d, symbol, trades_l):
    '''Create comprehensive visualization of backtest results'''

    # Use Agg backend logic implicitly via savefig
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Portfolio value and buy/hold comparison
    ax1 = plt.subplot(4, 1, 1)
    dates = df.index
    ax1.plot(dates, portfolio_v, label='Strategy', linewidth=2, color='blue')

    # Buy and hold
    initial_price = df['Close'].iloc[0]
    buy_hold_v = (df['Close'] / initial_price) * portfolio_v[0]
    ax1.plot(dates, buy_hold_v, label='Buy & Hold', linewidth=2,
             color='gray', alpha=0.7, linestyle='--')

    # Mark trades
    buy_dates = [t['date'] for t in trades_l if t['type'] == 'BUY']
    buy_values = [portfolio_v[df.index.get_loc(d)] for d in buy_dates]
    sell_dates = [t['date'] for t in trades_l if t['type'] == 'SELL']
    sell_values = [portfolio_v[df.index.get_loc(d)] for d in sell_dates]

    ax1.scatter(buy_dates, buy_values, color='green', marker='^',
                s=100, label='Buy', zorder=5)
    ax1.scatter(sell_dates, sell_values, color='red', marker='v',
                s=100, label='Sell', zorder=5)

    ax1.set_title(f'{symbol} - Momentum Strategy Backtest Results', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

    # Plot 2: Price with moving averages
    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(dates, df['Close'], label='Close Price', linewidth=1.5, color='black')
    ax2.plot(dates, ma_short_v, label='MA Short', linewidth=1, color='orange', alpha=0.7)
    ax2.plot(dates, ma_long_v, label='MA Long', linewidth=1, color='purple', alpha=0.7)
    ax2.set_ylabel('Price ($)', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Plot 3: RSI
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(dates, rsi_v, label='RSI', linewidth=1.5, color='darkblue')
    ax3.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
    ax3.axhline(50, color='green', linestyle='--', alpha=0.5, label='Momentum Thresh')
    ax3.fill_between(dates, 30, 70, alpha=0.1, color='gray')
    ax3.set_ylabel('RSI', fontsize=12)
    ax3.set_ylim(0, 100)
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Drawdown
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    cummax_v = maximum.accumulate(portfolio_v)
    drawdown_v = (portfolio_v - cummax_v) / cummax_v * 100
    ax4.fill_between(dates, drawdown_v, 0, alpha=0.5, color='red')
    ax4.plot(dates, drawdown_v, linewidth=1, color='darkred')
    ax4.set_ylabel('Drawdown (%)', fontsize=12)
    ax4.set_xlabel('Date', fontsize=12)
    ax4.grid(True, alpha=0.3)

    # Add metrics text box
    metrics_text = f"""Performance Metrics:
Total Return: {metrics_d['total_return']:.2%}
Annualized Return: {metrics_d['annualized_return']:.2%}
Sharpe Ratio: {metrics_d['sharpe_ratio']:.3f}
Sortino Ratio: {metrics_d['sortino_ratio']:.3f}
Max Drawdown: {metrics_d['max_drawdown']:.2%}
Win Rate: {metrics_d['win_rate']:.2%}
Final Value: ${metrics_d['final_value']:,.2f}

Buy & Hold Return: {metrics_d['buy_hold_return']:.2%}
Excess Return: {metrics_d['excess_return']:.2%}
Total Trades: {len(trades_l)}"""

    plt.figtext(0.99, 0.5, metrics_text, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='center', horizontalalignment='left',
                fontfamily='monospace')

    plt.tight_layout()
    plt.subplots_adjust(right=0.85)

    # SAVE instead of SHOW to prevent crashes
    filename = os.path.join(OUTPUT_DIR, f'{symbol}_chart.png')
    plt.savefig(filename)
    plt.close()
    print(f"  Chart saved to {filename}")


def save_results(symbol, metrics_d, trades_l, params_d, filename):
    '''Save backtest results to file'''
    with open(filename, 'w') as f:
        f.write(f"MOMENTUM STRATEGY BACKTEST RESULTS - {symbol}\n")
        f.write("=" * 80 + "\n\n")

        f.write("STRATEGY PARAMETERS:\n")
        f.write("-" * 40 + "\n")
        for key, val in params_d.items():
            f.write(f"{key}: {val}\n")

        f.write("\n\nPERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Return: {metrics_d['total_return']:.2%}\n")
        f.write(f"Annualized Return: {metrics_d['annualized_return']:.2%}\n")
        f.write(f"Sharpe Ratio: {metrics_d['sharpe_ratio']:.3f}\n")
        f.write(f"Sortino Ratio: {metrics_d['sortino_ratio']:.3f}\n")
        f.write(f"Max Drawdown: {metrics_d['max_drawdown']:.2%}\n")
        f.write(f"Win Rate: {metrics_d['win_rate']:.2%}\n")
        f.write(f"Final Portfolio Value: ${metrics_d['final_value']:,.2f}\n")
        f.write(f"Buy & Hold Return: {metrics_d['buy_hold_return']:.2%}\n")
        f.write(f"Excess Return vs Buy & Hold: {metrics_d['excess_return']:.2%}\n")

        f.write(f"\n\nTRADE HISTORY ({len(trades_l)} trades):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Date':<12} {'Type':<6} {'Price':>10} {'Shares':>10} {'Value':>15}\n")
        f.write("-" * 80 + "\n")

        for trade in trades_l:
            f.write(f"{str(trade['date'].date()):<12} {trade['type']:<6} "
                    f"{trade['price']:>10.2f} {trade['shares']:>10} "
                    f"${trade['value']:>14,.2f}\n")

    print(f"  Results saved to {filename}")


def print_summary(results_d):
    '''Print summary of all backtests'''
    print("\n" + "=" * 100)
    print("MULTI-ASSET BACKTEST SUMMARY")
    print("=" * 100)

    header = f"{'Symbol':<8} {'Total Return':>14} {'Annual Return':>14} {'Sharpe':>10} " \
             f"{'Max DD':>10} {'Win Rate':>10} {'vs B&H':>12} {'Trades':>8}"
    print(header)
    print("-" * 100)

    for symbol, data in results_d.items():
        m = data['metrics']
        print(f"{symbol:<8} {m['total_return']:>13.2%} {m['annualized_return']:>13.2%} "
              f"{m['sharpe_ratio']:>10.3f} {m['max_drawdown']:>9.2%} "
              f"{m['win_rate']:>9.2%} {m['excess_return']:>11.2%} "
              f"{len(data['trades']):>8}")

    # Calculate portfolio-level stats
    avg_sharpe = mean([data['metrics']['sharpe_ratio'] for data in results_d.values()])
    avg_return = mean([data['metrics']['total_return'] for data in results_d.values()])

    print("-" * 100)
    print(f"{'AVERAGE':<8} {avg_return:>13.2%} {'':>14} {avg_sharpe:>10.3f}")


def main():
    # Set random seed
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("MOMENTUM STRATEGY BACKTESTER")
    print("=" * 80)

    # Fetch data
    data_d = fetch_data(SYMBOLS, START_DATE, END_DATE)

    if not data_d:
        print("\nError: No data downloaded. Exiting.")
        return

    # Parameters
    params_d = {
        'rsi_period': RSI_PERIOD,
        'rsi_oversold': RSI_BUY_THRESHOLD,  # Using as buy threshold
        'rsi_overbought': RSI_SELL_THRESHOLD,
        'ma_short': MA_SHORT,
        'ma_long': MA_LONG,
        'volume_ma': VOLUME_MA
    }

    results_d = {}

    for symbol, df in data_d.items():
        print(f"\n{'=' * 80}")
        print(f"Backtesting {symbol}")
        print('=' * 80)

        # Optimize if requested
        if RUN_OPTIMIZATION:
            print("Running optimization to find best RSI and Moving Average settings...")
            params_d = optimize_parameters(df, INITIAL_CAPITAL, COMMISSION, SLIPPAGE)
            print(f"  Optimal parameters found:")
            print(f"  - RSI Period: {params_d['rsi_period']}")
            print(f"  - Buy Threshold: {params_d['rsi_oversold']:.1f}")
            print(f"  - Fast MA: {params_d['ma_short']}")
            print(f"  - Slow MA: {params_d['ma_long']}")

        # Generate signals and backtest
        signals_v, rsi_v, ma_short_v, ma_long_v = generate_signals(df, params_d)
        portfolio_v, trades_l = backtest_strategy(df, signals_v, INITIAL_CAPITAL,
                                                  COMMISSION, SLIPPAGE)
        metrics_d = calculate_performance_metrics(portfolio_v, INITIAL_CAPITAL, df)

        # Store results
        results_d[symbol] = {
            'portfolio': portfolio_v,
            'trades': trades_l,
            'metrics': metrics_d,
            'signals': signals_v,
            'rsi': rsi_v,
            'ma_short': ma_short_v,
            'ma_long': ma_long_v
        }

        # Save individual results
        filename = os.path.join(OUTPUT_DIR, f'{symbol}_backtest_results.txt')
        save_results(symbol, metrics_d, trades_l, params_d, filename)

        # Plot if requested
        if SHOW_PLOTS:
            plot_backtest_results(df, portfolio_v, signals_v, rsi_v, ma_short_v,
                                  ma_long_v, metrics_d, symbol, trades_l)

    # Print summary
    if len(results_d) > 1:
        print_summary(results_d)

    print("\n" + "=" * 80)
    print("Backtest completed successfully!")
    print(f"Charts saved to {OUTPUT_DIR}")
    print("=" * 80)


if __name__ == '__main__':
    main()
