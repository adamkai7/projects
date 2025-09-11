'''
Created by Adam Kainikara

This tool performs rolling linear regression analysis to calculate beta coefficients
and momentum factors for individual stocks relative to a benchmark ticker. The analysis imple-
ments a two-stage regression approach: first calculating Capital Asset Pricing Model (CAPM) beta
coefficients through rolling windows, then extending the analysis to capture momentum effects by
examining how stock residuals respond to lagged market returns
'''
import os
from numpy import *
import random

from volta.utils import expand_path, time_to_slice, parse_date_range
from volta.cmdparser import code_relative_path, CommandParser
from niven.data import load_catalog, load_aligned_series, StockDataset
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def parse_args():
    #data loader
    WORK_DIR = os.getenv('WORK_DIR') or expand_path('/data/tmp/indus/yfin')
    RES_DIR = expand_path('~/kalari/indus/resources')
    cmdparser = CommandParser()
    cmdparser.arg('--data_dir', type=str, default=f'{WORK_DIR}/data/stocks/history', help='Data directory')
    cmdparser.arg('--res_dir', type=str, default=RES_DIR, help='Resource directory')
    cmdparser.arg('--interval', type=str, default='1d', help='Sampling interval of the data')

    cmdparser.arg('--input', '-i', type=str, action='append', default=[], help='Input files (csv, symbols)')
    cmdparser.arg('--train', type=str, default='none',
                  help='Time range of training data. e.g.: 2012-01-01,2012-01-01T00:00:00  2012-01-01,now 2012-01-01,15 15,now 15')

    cmdparser.arg('--show', action='store_true', help='Show plots')
    cmdparser.arg('--out', '-o', type=str, default=f'{WORK_DIR}/results', help='Output dir.')
    cmdparser.arg('--seed', type=int, default=0, help='random seed.')

    cmdparser.arg('symbol_l', nargs='*')

    args = cmdparser.eval()

    args.train_t = parse_date_range(args.train, dt_unit='D')
    args.symbol_l = [symbol.upper()
                     for symbol in args.symbol_l]  # TD API expects upper case

    if args.input:
        args.symbol_l = unique(
            args.symbol_l + list(load_catalog(args.input, args.res_dir)))

    if args.seed:
        sys_set_seed(args.seed)
    return args


def parse_training_period(args, date_v):
    # finish up arg parsing that can only be done once we have
    # loaded the dates.
    ts = slice(0, date_v.shape[0])

    if args.train_t[0]:
        ts = time_to_slice(date_v, *args.train_t)
    return ts


# ----------------------------------------------------------------------------------------------------------


def log_returns(data_a):
    log_prices = log(data_a)
    returns = log_prices[:, 1:] - log_prices[:, :-1]
    return returns


def linear_fit(data_a, x_v, window_size):
    N, T = data_a.shape
    n_windows = T - window_size + 1

    alphas_a = zeros((N, n_windows))
    betas_a = zeros((N, n_windows))

    for w in range(n_windows):
        start, end = w, w + window_size
        # column of ones and x_v slice
        X = vstack([ones(window_size), x_v[start:end]]).T  # shape (window_size, 2)

        for i in range(N):
            y = data_a[i, start:end]                # asset i returns
            coef, *_ = linalg.lstsq(X, y, rcond=None)
            alphas_a[i, w] = coef[0]  # intercept
            betas_a[i, w] = coef[1]  # slope

    return alphas_a, betas_a


def momentum_way(data_a, x_v, window_size, betas_a):
    """
    given rolling betas_a (shape N×n_windows) from linear_fit, compute
    the momentum coefficient by regressing the in-window residuals
      res_i[t] = r_i[t] − β_i[w]·r_m[t]
    on the lagged market return r_m[t−1].
    
    """
    N, T = data_a.shape
    n_windows = T - window_size + 1

    momentums_a = zeros((N, n_windows))

    for w in range(n_windows):
        s, e = w, w + window_size
        rm_win_v = x_v[s:e]               # (window_size,)

        for i in range(N):
            # residual series in window
            res_win_v = data_a[i, s:e] - betas_a[i, w] * rm_win_v
            # regress res[1:] on rm[:-1]
            X_a = rm_win_v[:-1].reshape(-1, 1)  # (window_size-1,1)
            y_v = res_win_v[1:]                 # (window_size-1,)
            gamma_v, *_ = linalg.lstsq(X_a, y_v, rcond=None)
            momentums_a[i, w] = gamma_v[0]

    return betas_a, momentums_a
'''

VISUALS FOR REPORT
VISUALS FOR REPORT
VISUALS FOR REPORT
VISUALS FOR REPORT
VISUALS FOR REPORT


'''

def plot_rolling_coefficients(date_v, symbols_v, alphas_a, betas_a, momentums_a, window_size, reference_symbol):
    
    #plots of rolling alpha, beta, and momentum coefficients for all stocks
    
    #  date array for plotting (adjust for log returns reduction + window size)
    # log returns reduces length by 1, then rolling window reduces by (window_size-1)
    plot_dates = date_v[window_size:]
    
    # convert date strings to datetime objects 
    if isinstance(plot_dates[0], str):
        plot_dates = [datetime.strptime(d, "%Y-%m-%d") for d in plot_dates]
    
    # find out reference symbol from plotting
    non_ref_indices = [i for i, sym in enumerate(symbols_v) if sym != reference_symbol]
    plot_symbols = [symbols_v[i] for i in non_ref_indices]
    
    # sub subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))
    
    #  Rolling Alphas
    ax1.set_title(f'Rolling {window_size}-Day Alpha Coefficients vs {reference_symbol}', fontsize=14)
    for i, idx in enumerate(non_ref_indices):
        ax1.plot(plot_dates, alphas_a[idx, :], label=plot_symbols[i], linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax1.set_ylabel('Alpha', fontsize=12)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    #  Rolling Betas
    ax2.set_title(f'Rolling {window_size}-Day Beta Coefficients vs {reference_symbol}', fontsize=14)
    for i, idx in enumerate(non_ref_indices):
        ax2.plot(plot_dates, betas_a[idx, :], label=plot_symbols[i], linewidth=2)
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.7, label='Market Beta = 1')
    ax2.set_ylabel('Beta', fontsize=12)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    #  Rolling Momentum
    ax3.set_title(f'Rolling {window_size}-Day Momentum Coefficients vs {reference_symbol}', fontsize=14)
    for i, idx in enumerate(non_ref_indices):
        ax3.plot(plot_dates, momentums_a[idx, :], label=plot_symbols[i], linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.7)
    ax3.set_ylabel('Momentum Coefficient', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # format x-axis dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))
    
    plt.tight_layout()
    plt.show()


def plot_beta_distribution(symbols_v, betas_a, reference_symbol):
    
    #plot distribution of beta coefficients for each stock
    
    # filter out reference symbol
    non_ref_indices = [i for i, sym in enumerate(symbols_v) if sym != reference_symbol]
    plot_symbols = [symbols_v[i] for i in non_ref_indices]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # create box plot of beta distributions
    beta_data = [betas_a[idx, :] for idx in non_ref_indices]
    box_plot = ax.boxplot(beta_data, tick_labels=plot_symbols, patch_artist=True)
    
    # color the boxes
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Market Beta = 1')
    ax.set_title(f'Distribution of Rolling Beta Coefficients vs {reference_symbol}', fontsize=14)
    ax.set_ylabel('Beta Coefficient', fontsize=12)
    ax.set_xlabel('Stock Symbol', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def print_coefficient_summary(symbols_v, alphas_a, betas_a, momentums_a, reference_symbol):
    
    #print summary statistics for coefficients
    
    print(f"\nSUMMARY STATISTICS - ROLLING REGRESSION ANALYSIS vs {reference_symbol}")
    print("=" * 80)
    print(f"{'Symbol':<8} {'Avg Alpha':<12} {'Avg Beta':<12} {'Avg Momentum':<15} {'Beta Std':<12}")
    print("-" * 80)
    
    for i, symbol in enumerate(symbols_v):
        if symbol != reference_symbol:
            avg_alpha = mean(alphas_a[i, :])
            avg_beta = mean(betas_a[i, :])
            avg_momentum = mean(momentums_a[i, :])
            beta_std = std(betas_a[i, :])
            
            print(f"{symbol:<8} {avg_alpha:<12.4f} {avg_beta:<12.4f} {avg_momentum:<15.4f} {beta_std:<12.4f}")


def save_results_to_file(date_v, symbols_v, alphas_a, betas_a, momentums_a, window_size, reference_symbol, filename="rolling_regression_results.txt"):

    #saves  results to text file
    
    plot_dates = date_v[window_size:]
    non_ref_indices = [i for i, sym in enumerate(symbols_v) if sym != reference_symbol]
    
    with open(filename, 'w') as f:
        f.write(f"ROLLING REGRESSION ANALYSIS RESULTS\n")
        f.write(f"Reference Symbol: {reference_symbol}\n")
        f.write(f"Window Size: {window_size} days\n")
        f.write(f"Analysis Period: {plot_dates[0]} to {plot_dates[-1]}\n")
        f.write("=" * 80 + "\n\n")
        
        for idx in non_ref_indices:
            symbol = symbols_v[idx]
            f.write(f"\n{symbol} COEFFICIENTS\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Date':<12} {'Alpha':<12} {'Beta':<12} {'Momentum':<12}\n")
            f.write("-" * 40 + "\n")
            
            for t in range(len(plot_dates)):
                f.write(f"{plot_dates[t]:<12} {alphas_a[idx, t]:<12.6f} {betas_a[idx, t]:<12.6f} {momentums_a[idx, t]:<12.6f}\n")
    
    print(f"\nDetailed results saved to {filename}")


def plot_scatter_with_regression(log_return_data_a, ref_log_returns_v, symbols_v, reference_symbol, window_idx=None):
    """
    Plot scatter plots of stock returns vs market returns with regression lines
    If window_idx is None, uses all data. Otherwise uses specific window.
    """
    non_ref_indices = [i for i, sym in enumerate(symbols_v) if sym != reference_symbol]
    plot_symbols = [symbols_v[i] for i in non_ref_indices]
    
    fig, axes = plt.subplots(1, len(plot_symbols), figsize=(5*len(plot_symbols), 4))
    if len(plot_symbols) == 1:
        axes = [axes]
    
    for i, (idx, symbol) in enumerate(zip(non_ref_indices, plot_symbols)):
        ax = axes[i]
        
        if window_idx is None:
            # Use all data
            x_data = ref_log_returns_v
            y_data = log_return_data_a[idx, :]
            title_suffix = "All Data"
        else:
            # Use specific window
            start = window_idx
            end = window_idx + 25  # window_size
            x_data = ref_log_returns_v[start:end]
            y_data = log_return_data_a[idx, start:end]
            title_suffix = f"Window {window_idx}"
        
        # Create scatter plot
        ax.scatter(x_data, y_data, alpha=0.6, s=20)
        
        # Fit regression line
        X = vstack([ones(len(x_data)), x_data]).T
        coef, *_ = linalg.lstsq(X, y_data, rcond=None)
        alpha_val, beta_val = coef[0], coef[1]
        
        # Plot regression line
        x_line = linspace(min(x_data), max(x_data), 100)
        y_line = alpha_val + beta_val * x_line
        ax.plot(x_line, y_line, 'r-', linewidth=2, 
                label=f'α={alpha_val:.4f}, β={beta_val:.2f}')
        
        ax.set_xlabel(f'{reference_symbol} Returns')
        ax.set_ylabel(f'{symbol} Returns')
        ax.set_title(f'{symbol} vs {reference_symbol}\n{title_suffix}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add 45-degree line (beta=1 reference) - FIXED
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        # Use explicit comparison to avoid numpy min/max issues
        lim_min = xlims[0] if xlims[0] < ylims[0] else ylims[0]
        lim_max = xlims[1] if xlims[1] > ylims[1] else ylims[1]
        ax.plot([lim_min, lim_max], [lim_min, lim_max], 'k--', alpha=0.5, linewidth=1, label='β=1')
    
    plt.tight_layout()
    plt.show()


def plot_rolling_comparison(date_v, log_return_data_a, symbols_v, alphas_a, betas_a, reference_symbol, window_size):
    
    #pl actual stock prices with periods of high/low beta highlighted
    
    non_ref_indices = [i for i, sym in enumerate(symbols_v) if sym != reference_symbol]
    plot_symbols = [symbols_v[i] for i in non_ref_indices]
    plot_dates = date_v[window_size:]
    
    if isinstance(plot_dates[0], str):
        plot_dates = [datetime.strptime(d, "%Y-%m-%d") for d in plot_dates]
    
    fig, axes = plt.subplots(len(plot_symbols), 1, figsize=(14, 4*len(plot_symbols)))
    if len(plot_symbols) == 1:
        axes = [axes]
    
    for i, (idx, symbol) in enumerate(zip(non_ref_indices, plot_symbols)):
        ax = axes[i]
        
        # Plot cumulative returns (approximation of price movement)
        cumulative_returns = cumsum(log_return_data_a[idx, :])
        
        # Create color map based on beta values
        betas = betas_a[idx, :]
        
        # Plot the price line
        ax.plot(plot_dates, cumulative_returns[window_size:], 'b-', linewidth=1, alpha=0.7)
        
        # Highlight high beta periods (beta > 1.5) in red
        high_beta_mask = betas > 1.5
        if any(high_beta_mask):
            ax.scatter(array(plot_dates)[high_beta_mask], 
                      cumulative_returns[window_size:][high_beta_mask],
                      c='red', s=20, alpha=0.7, label='High β (>1.5)')
        
        # Highlight low beta periods (beta < 0.5) in green  
        low_beta_mask = betas < 0.5
        if any(low_beta_mask):
            ax.scatter(array(plot_dates)[low_beta_mask],
                      cumulative_returns[window_size:][low_beta_mask], 
                      c='green', s=20, alpha=0.7, label='Low β (<0.5)')
        
        ax.set_title(f'{symbol} - Cumulative Returns with Beta Periods')
        ax.set_ylabel('Cumulative Log Returns')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    ads = StockDataset('data', *load_aligned_series(args.symbol_l,  # + benchmark_sym_l,
                                                    f'{args.data_dir}/{args.interval}',
                                                    method='percentile'))
    ads.report()

    if args.train != 'none':
        train_slice = parse_training_period(args, ads.date_v)
        # Training dataset
        tds = StockDataset('train', ads.sym_v, ads.data_a[:, train_slice])
        tds.report()
        
        # Use training slice for analysis
        date_v = array(ads.date_v)[train_slice]
        price_data = ads.price_a[:, train_slice]
    else:
        date_v = array(ads.date_v)
        price_data = ads.price_a

    symbols_v = ads.sym_v
    window_size = 25

    log_return_data_a = log_returns(price_data)

    reference_symbol = 'SPY'
    print(f"Using {reference_symbol} as reference")

    reference_idx = ads.sym_index_d[reference_symbol]
    ref_log_returns_v = log_return_data_a[reference_idx]

    print(f"Log returns shape: {log_return_data_a.shape}")
    print(f"Reference returns length: {ref_log_returns_v.size}")

    # Calculate rolling coefficients
    alphas_a, betas_a = linear_fit(log_return_data_a, ref_log_returns_v, window_size)
    betas_out_a, momentums_a = momentum_way(log_return_data_a, ref_log_returns_v, window_size, betas_a)

    print(f"Alphas shape: {alphas_a.shape}")
    print(f"Betas shape: {betas_a.shape}")
    print(f"Momentum shape: {momentums_a.shape}")

    # Print summary statistics
    print_coefficient_summary(symbols_v, alphas_a, betas_a, momentums_a, reference_symbol)

    # Create visualizations
    print("\nGenerating visualizations...")
    
    # 1. Original coefficient plots
    plot_rolling_coefficients(date_v, symbols_v, alphas_a, betas_a, momentums_a, window_size, reference_symbol)
    
    # 2. Beta distribution
    plot_beta_distribution(symbols_v, betas_a, reference_symbol)
    
    # 3. NEW: Scatter plots with regression lines (all data)
    plot_scatter_with_regression(log_return_data_a, ref_log_returns_v, symbols_v, reference_symbol)
    
    # 4. NEW: Recent window example (last 25 days)
    recent_window = log_return_data_a.shape[1] - window_size
    plot_scatter_with_regression(log_return_data_a, ref_log_returns_v, symbols_v, reference_symbol, window_idx=recent_window)
    
    # 5. NEW: Price movement with beta highlighting
    plot_rolling_comparison(date_v, log_return_data_a, symbols_v, alphas_a, betas_a, reference_symbol, window_size)

    # Save detailed results
    save_results_to_file(date_v, symbols_v, alphas_a, betas_a, momentums_a, window_size, reference_symbol)


if __name__ == "__main__":
    main()