
'''
created by adam kainikara

The purpose of this analysis is to examine if seasonal patterns exist in stock market
returns by reorganizing historical price data according to calendar days rather than chronological
time. This approach reveals whether certain calendar dates or periods during the year consistently
exhibit higher or lower returns across multiple years, potentially indicating market seasonality
effects.
Example of how to run: python density_of_returns.py MSFT --train 2020-01-01,now --show

'''
import os
from numpy import *
import random
from numpy.random import RandomState
import matplotlib.pyplot as plt
from volta.utils import expand_path, time_to_slice, parse_date_range,wmap

from volta.cmdparser import code_relative_path, CommandParser
from niven.data import load_catalog, load_aligned_series, StockDataset
from datetime import datetime
from itertools import combinations
from scipy.optimize import differential_evolution
import datetime as dt

from numpy.lib.stride_tricks import sliding_window_view
from collections import defaultdict

from scipy.interpolate import interp1d


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
                  help='Time range of training data. e.g.: 2012-01-01,2012-01-01T00:00:00')

    cmdparser.arg('--show', action='store_true', help='Show plots')
    cmdparser.arg('--out', '-o', type=str, default=f'{WORK_DIR}/results', help='Output dir.')
    cmdparser.arg('--seed', type=int, default=0, help='random seed.')

    cmdparser.arg('symbol_l', nargs='*')
    cmdparser.arg('--r', type=str, default='SPY', help='Reference symbol for correlation')
    cmdparser.arg('--t', type=float, default=0.7, help='T for high/low correlation classification')
    cmdparser.arg('--k', type=int, default=5, help='Number of stocks to select via DE')
    cmdparser.arg('--wsize', type=int, default=3, help='window size')

    args = cmdparser.eval()

    args.train_t = parse_date_range(args.train, dt_unit='D')
    args.symbol_l = [symbol.upper() for symbol in args.symbol_l]

    if args.input:
        args.symbol_l = unique(
            args.symbol_l + list(load_catalog(args.input, args.res_dir)))

    if args.seed:
        random.seed(args.seed)
    return args

def parse_training_period(args, date_v):
    ts = slice(0, date_v.shape[0])
    if args.train_t[0]:
        ts = time_to_slice(date_v, *args.train_t)
    return ts


def log_returns(data_a):
    log_prices = log(data_a)
    returns = log_prices[:, 1:] - log_prices[:, :-1]
    return returns

def log_prices(x_v):
    return log(x_v)


def compute_returns(price_a, wsize):
    # returns of price t/price t-wsize  - 1
    n_symbols, n_dates = price_a.shape
    returns = full((n_symbols, n_dates - wsize), nan)
    for i in range(wsize, n_dates):
        returns[:, i - wsize] = (price_a[:, i] / price_a[:, i - wsize]) - 1
    return returns

def gaussian_kernel(wsize, bandwidth):
    # need for convolution and kde
    half_window = wsize // 2
    x = arange(-half_window, half_window + 1)
    kernel = exp(-0.5 * (x / bandwidth)**2)
    kernel = kernel / (bandwidth * sqrt(2 * pi))
    kernel = kernel / kernel.sum()
    return kernel

def compute_kde_convolution(price_a, date_v, wsize=5, bandwidth=20.0):
    returns = compute_returns(price_a, wsize)
    return_date_v_a = date_v[wsize:]
    n_symbols, n_returns = returns.shape
    density_v = full((n_symbols, n_returns), nan)
    if wsize % 2 == 0:
        wsize += 1
    kernel = gaussian_kernel(wsize, bandwidth)

    for i in range(n_symbols):
        series = returns[i, :]
        if isnan(series).any():
            continue
        smoothed = convolve(series, kernel, mode='same')
        density_v[i, :] = smoothed

    return density_v, return_date_v_a, returns

def compute_kernel_density_returns(price_a, date_v, wsize=5):
    density_v, return_date_v_a, _ = compute_kde_convolution(price_a, date_v, wsize, bandwidth=20.0)
    return density_v, return_date_v_a

def compute_kernel_density_manual(price_a, date_v, wsize=5, bandwidth=0.1):
    density_v, return_date_v_a, _ = compute_kde_convolution(price_a, date_v, wsize, bandwidth=20.0)
    return density_v, return_date_v_a


def reorganize_by_calendar_day(data_a, date_v, price_a, wsize=1):
    
    
    
    '''reorganze returns by calendar day'''
    
    
    day_data = defaultdict(list)
    date_groups_d = defaultdict(list)
    day_prices_d = defaultdict(list)
    
    for i, date in enumerate(date_v):
        dt_obj = date.astype('datetime64[D]').astype('O')
        day_key = f"{dt_obj.month:02d}-{dt_obj.day:02d}"
        
        day_data[day_key].append(data_a[:, i])
        date_groups_d[day_key].append(date)
        day_prices_d[day_key].append(price_a[:, i])
    
    sorted_days_l = sorted(day_data.keys())
    n_symbols = data_a.shape[0]
    seasonal_data_a = zeros((n_symbols, len(sorted_days_l)))
    return_prices_v = zeros((n_symbols, len(sorted_days_l)))
    
    for j, day_key in enumerate(sorted_days_l):
        stacked_data = array(day_data[day_key]).T
        seasonal_data_a[:, j] = nanmean(stacked_data, axis=1)
        
        # ⬇️ Instead of averaging prices, compute returns
        stacked_prices = vstack(day_prices_d[day_key]).T  # shape (n_symbols, n_years)
        if stacked_prices.shape[1] > wsize:  # need enough years for return calc
            returns = compute_returns(stacked_prices, wsize=wsize)
            return_prices_v[:, j] = nanmean(returns, axis=1)
        else:
            return_prices_v[:, j] = nan  # not enough data for return
    print('h')
    print(type(seasonal_data_a))
    print(type(sorted_days_l))
    print(type(date_groups_d))
    print('cat')
    print(type(return_prices_v))
    print(return_prices_v)
    print('hiii')
    print(price_a)
    print(stacked_prices)
    print(type(day_prices_d))
    return seasonal_data_a, sorted_days_l, date_groups_d, return_prices_v, day_prices_d
    
    print('h')
    print(type(seasonal_data_a))
    print(type(sorted_days_l))
    print(type(date_groups_d))
    print(type(mean_price_v))
    print('h')

    print(type(day_prices_d))
    #return seasonal_data_a, sorted_days_l, date_groups_d, mean_price_v, day_prices_d

def plot_kernel_density_histogram(density_v, symbol_names, title="Kernel Density Distribution"):
    plt.figure(figsize=(12, 8))
    n_plots = min(4, density_v.shape[0])
    for i in range(n_plots):
        plt.subplot(2, 2, i+1)
        symbol_density = density_v[i, :]
        valid_density = symbol_density[~isnan(symbol_density)]
        plt.hist(valid_density, bins=50, alpha=0.7, edgecolor='black')
        plt.title(f'{symbol_names[i] if i < len(symbol_names) else f"Symbol {i}"}')
        plt.xlabel('Kernel Density')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.suptitle(title, y=1.02, fontsize=16)
    plt.show()

def plot_seasonal_returns_scatter(day_label_l, return_prices_v, symbol_names, single_symbol=True):
    """
    Plot seasonal returns across calendar days
    """
    if single_symbol:
        plt.figure(figsize=(14, 6))
        symbol_idx = 0
        plt.scatter(day_label_l, return_prices_v[symbol_idx], color='red', alpha=0.7, s=40,
                    label=f"{symbol_names[symbol_idx]} Seasonal Returns")
        
        # Filter to show only 1st and 15th of each month
        filtered_ticks = [day for day in day_label_l if day.endswith('-01') or day.endswith('-15')]
        plt.xticks(ticks=range(len(day_label_l)), labels=day_label_l, rotation=90, fontsize=8)
        ax = plt.gca()
        tick_positions = [i for i, day in enumerate(day_label_l) if day in filtered_ticks]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([day_label_l[i] for i in tick_positions], rotation=90, fontsize=8)
        
        plt.title(f"Seasonal Returns Across Calendar Days ({symbol_names[symbol_idx]})")
        plt.xlabel("Calendar Day (MM-DD)")
        plt.ylabel("Return")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 8))
        axes = axes.flatten()
        n_symbols_to_plot = min(4, len(symbol_names))
        
        filtered_ticks = [day for day in day_label_l if day.endswith('-01') or day.endswith('-15')]
        tick_positions = [i for i, day in enumerate(day_label_l) if day in filtered_ticks]
        
        for sym_idx in range(n_symbols_to_plot):
            ax = axes[sym_idx]
            ax.scatter(range(len(day_label_l)), return_prices_v[sym_idx], color='red', alpha=0.7, s=30)
            ax.set_title(f"{symbol_names[sym_idx]} Seasonal Returns")
            ax.set_xlabel("Calendar Day")
            ax.set_ylabel("Return")
            ax.grid(True, alpha=0.3)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([day_label_l[i] for i in tick_positions], rotation=90, labelsize=6)
        
        plt.tight_layout()
        plt.show()


def plot_seasonal_price_scatter(day_prices_d, day_label_l, symbol_names, single_symbol=True):
    """
    Plot scatter plots of  prices by calendar day
    """
    if single_symbol:
        plt.figure(figsize=(14, 8))
        
        symbol_idx = 0
        all_days = []
        all_prices = []
        
        for day_key in sorted(day_prices_d.keys()):
            for price_array in day_prices_d[day_key]:
                all_days.append(day_key)
                all_prices.append(price_array[symbol_idx])
        
        plt.scatter(all_days, all_prices, alpha=0.6, s=20,
                    label=f'{symbol_names[symbol_idx]} - Individual Days')
        
        # fiter to show only 1st and 15th of each month on graph
        unique_days = sorted(list(set(all_days)))
        filtered_ticks = [day for day in unique_days if day.endswith('-01') or day.endswith('-15')]
        tick_positions = [i for i, day in enumerate(unique_days) if day in filtered_ticks]
        
        ax = plt.gca()
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([unique_days[i] for i in tick_positions], rotation=90, fontsize=8)
        
        plt.title(f"Individual Daily Prices by Calendar Day ({symbol_names[symbol_idx]})")
        plt.xlabel("Calendar Day (MM-DD)")
        plt.ylabel("Price")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
    else:
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        n_symbols_to_plot = min(4, len(symbol_names))
        
        for sym_idx in range(n_symbols_to_plot):
            ax = axes[sym_idx]
            all_days, all_prices = [], []
            for day_key in sorted(day_prices_d.keys()):
                for price_array in day_prices_d[day_key]:
                    all_days.append(day_key)
                    all_prices.append(price_array[sym_idx])
            
            unique_days = sorted(list(set(all_days)))
            filtered_ticks = [day for day in unique_days if day.endswith('-01') or day.endswith('-15') or day == '01-01']
            tick_positions = [i for i, day in enumerate(unique_days) if day in filtered_ticks]
            
            ax.scatter(all_days, all_prices, alpha=0.5, s=15)
            ax.set_title(f'{symbol_names[sym_idx]} - Daily Prices')
            ax.set_xlabel('Calendar Day')
            ax.set_ylabel('Price')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([unique_days[i] for i in tick_positions], rotation=90, labelsize=6)
        
        plt.tight_layout()
        plt.show()

# ---------------------------- MAIN ----------------------------
def main():
    args = parse_args()

    # Load dataset
    ads = StockDataset('data', *load_aligned_series(
        args.symbol_l,
        f'{args.data_dir}/{args.interval}'
    ))
    ads.report()
    
    # Build symbol list
    if args.input:
        syms_l = unique([s.upper() for s in args.symbol_l] + list(load_catalog(args.input, args.res_dir)))
    else:
        syms_l = [s.upper() for s in args.symbol_l]

    # Training period slice
    if args.train != 'none':
        train_slice = parse_training_period(args, ads.date_v)
        tds = StockDataset('train', ads.sym_v, ads.data_a[:, train_slice])
        tds.report()
        date_v = array(ads.date_v)[train_slice]
        price_data_a = ads.price_a[:, train_slice]  
    else:
        date_v = array(ads.date_v)
        price_data_a = ads.price_a  

    #  Kernel density of returns 
    density_v, return_date_v = compute_kernel_density_returns(price_data_a, date_v, args.wsize)

    # Reorganize by calendar day (returns-based seasonality) 
    seasonal_density_v, day_label_l, grouped_date_d, return_prices_v, day_prices_d = reorganize_by_calendar_day(
        density_v, return_date_v, price_data_a[:, 1:], wsize=args.wsize
    )

    # Plots 
    if args.show:
        # scatter of raw daily prices
        plot_seasonal_price_scatter(day_prices_d, day_label_l, syms_l, single_symbol=True)

        # seasonal returns scatter plot
        plot_seasonal_returns_scatter(day_label_l, return_prices_v, syms_l, single_symbol=True)


if __name__ == "__main__":
    main()
'''
created by adam kainikara

The purpose of this analysis is to examine if seasonal patterns exist in stock market
returns by reorganizing historical price data according to calendar days rather than chronological
time. This approach reveals whether certain calendar dates or periods during the year consistently
exhibit higher or lower returns across multiple years, potentially indicating market seasonality
effects.
Example of how to run: python density_of_returns.py MSFT --train 2020-01-01,now --show

'''