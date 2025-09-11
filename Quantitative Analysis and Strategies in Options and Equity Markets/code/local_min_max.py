
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

'''
created by Adam Kainikara
This tool is designed to identify local minima and maxima of a stock price over
time. A wsize parameters show how long each window of time is.

How to run:
python local_min_max.py MSFT --train 2023-01-01,now --wsize 10 20 50

'''
def parse_args():
    #this is data loader
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
    cmdparser.arg('--wsize', type=int, nargs='*', default=[20, 50, 100], help='List of window sizes to find local extrema for (e.g., --wsize 10 20 50)')

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




def log_returns(data_a):
    log_prices = log(data_a)
    returns = log_prices[:, 1:] - log_prices[:, :-1]
    return returns



def find_local_extrema_v(price_v, date_v, lwsize=50, rwsize=50):
    """
    where types are 'min' or 'max'
    """
    n = price_v.shape[0]
    extrema_indices_l = []
    extrema_types_l = []

    # loops through valid indices where  can create full windows
    for i in range(lwsize, n - rwsize):
        # make window around current point
        window_start = i - lwsize
        window_end = i + rwsize + 1
        window = price_v[window_start:window_end]

        # find min and max indices within window
        window_min_idx = argmin(window)
        window_max_idx = argmax(window)

        # check if current point is the extremum
        if window_min_idx == lwsize:  # Current point is at lwsize position in window
            extrema_indices_l.append(i)
            extrema_types_l.append('min')
        elif window_max_idx == lwsize:  # Current point is at lwsize position in window
            extrema_indices_l.append(i)
            extrema_types_l.append('max')

    return array(extrema_indices_l), array(extrema_types_l)


def find_local_extrema_in_windows(date_v, price_v, window_size):
    
    #where types are 'min' or 'max' using windows
    
    n = price_v.shape[0]
    extrema_indices_l = []
    extrema_types_l = []

    for i in range(0, n, window_size):
        end = array([i + window_size, n]).min()

        window = price_v[i:end]

        if len(window) == 0:
            continue

        min_rel = argmin(window)
        max_rel = argmax(window)

        min_abs = i + min_rel
        max_abs = i + max_rel

        extrema_indices_l.append(min_abs)
        extrema_types_l.append('min')
        if max_abs != min_abs:  # prevent duplicate if both are same
            extrema_indices_l.append(max_abs)
            extrema_types_l.append('max')

    return array(extrema_indices_l), array(extrema_types_l)


def print_extrema_with_labels(date_v, price_v, extrema_indices_v, extrema_types, label="", train_slice=None):
   #prints the output and extremas to terminal
    print(f"\n{label} WINDOW EXTREMA")
    print("=" * 60)

    # Sort extrema by index
    sorted_idx = sorted(range(len(extrema_indices_v)), key=lambda i: extrema_indices_v[i])

    # detrime the start index based on training slice
    if train_slice is not None:
        start_date_idx = train_slice.start
    else:
        start_date_idx = 0  # defaults to beginning if no training slice specified

    # find where first extrema index >= start_date_idx
    start_pos = 0
    for pos, i in enumerate(sorted_idx):
        if extrema_indices_v[i] >= start_date_idx:
            start_pos = pos
            break

    # print from start_pos onwards
    for i in sorted_idx[start_pos:]:
        idx = extrema_indices_v[i]
        typ = extrema_types[i]
        print(f"{date_v[idx]} -> {price_v[idx]:6.2f}  ({typ})")

def save_extrema_to_file(date_v, price_v, extrema_indices_v, extrema_types, label="", train_slice=None, filename="local_min_max.txt", append=False):
    
    #saves extrema to a text file 
    
    mode = 'a' if append else 'w'
    
    with open(filename, mode) as f:
        f.write(f"\n{label} WINDOW EXTREMA\n")
        f.write("=" * 60 + "\n")

        # sortts extrema by index
        sorted_idx = sorted(range(len(extrema_indices_v)), key=lambda i: extrema_indices_v[i])

        # determines the start index based on training slice
        if train_slice is not None:
            start_date_idx = train_slice.start
        else:
            start_date_idx = 0  # defaults to beginning if no training slice specified

        # finds first extrema index >= start_date_idx
        start_pos = 0
        for pos, i in enumerate(sorted_idx):
            if extrema_indices_v[i] >= start_date_idx:
                start_pos = pos
                break

        # Write extrema from start_pos onwards
        for i in sorted_idx[start_pos:]:
            idx = extrema_indices_v[i]
            typ = extrema_types[i]
            f.write(f"{date_v[idx]} -> {price_v[idx]:6.2f}  ({typ})\n")

def plot_dates_and_closes(date_v, close_v, extrema_indices_l=None, extrema_types_l=None, title="Stock Prices Over Time"):
    #visual i made
    # convert date strings to datetime objects if necessary
    if isinstance(date_v[0], str):
        date_v = [datetime.strptime(d, "%Y-%m-%d") for d in date_v]

    plt.figure(figsize=(14, 6))
    plt.plot(date_v, close_v, label="Close Price", color='blue')

    # plt extrema points if provided
    if extrema_indices_l is not None and extrema_types_l is not None:
        for idx, typ in zip(extrema_indices_l, extrema_types_l):
            if typ == 'min':
                plt.scatter(date_v[idx], close_v[idx], color='red', marker='v', s=100, label='Local Min')
            elif typ == 'max':
                plt.scatter(date_v[idx], close_v[idx], color='green', marker='^', s=100, label='Local Max')

        # avoids duplicate labels in legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
    else:
        plt.legend()

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(title)
    plt.grid(True)

    # format the x-axis to show dates nicely
    plt.gca().xaxis.set_major_locator(mdates.YearLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()

    # add argument parsing for window sizes
    if not hasattr(args, 'wsize') or not args.wsize:
        wsize_list = [20, 50, 100]  # default sizes
    else:
        wsize_list = [int(w) for w in args.wsize]

    # Load aligned stock data
    ads = StockDataset('data', *load_aligned_series(
        args.symbol_l,
        f'{args.data_dir}/{args.interval}',
        method='percentile'
    ))
    ads.report()

    # determines training period slice
    if args.train != 'none':
        train_slice = parse_training_period(args, ads.date_v)
        tds = StockDataset('train', ads.sym_v, ads.data_a[:, train_slice])
        tds.report()
    else:
        train_slice = slice(0, len(ads.date_v))

    data_v = array(ads.date_v)
    closes_V = get_close_prices(ads)

    print(f"Data shape: {ads.data_a.shape}")

    # slices data to training period
    data_v_plot = data_v[train_slice]
    closes_V_plot = closes_V[train_slice]
    start_date = data_v_plot[0]
    first_iteration = True
    # computes and plot extrema for each window size separately
    for wsize in wsize_list:
        extrema_idx, extrema_type = find_local_extrema_in_windows(data_v, closes_V, window_size=wsize)

        # restricts extrema to training period and convert to relative indices
        mask = (extrema_idx >= train_slice.start) & (extrema_idx < train_slice.stop)
        rel_idx = extrema_idx[mask] - train_slice.start
        rel_type = extrema_type[mask]

        # Print extrema
        print_extrema_with_labels(data_v, closes_V, extrema_idx, extrema_type, label=f"{wsize} DAY", train_slice=train_slice)        
        save_extrema_to_file(data_v, closes_V, extrema_idx, extrema_type, label=f"{wsize} DAY", train_slice=train_slice, filename="local_min_max.txt", append=not first_iteration)
        first_iteration = False
        plot_dates_and_closes(
            data_v_plot,
            closes_V_plot,
            rel_idx,
            rel_type,
            title=f"{wsize}-Day Local Extrema")

    raise SystemExit

if __name__ == "__main__":
    main()

'''
created by Adam Kainikara
This tool is designed to identify local minima and maxima of a stock price over
time. A wsize parameters show how long each window of time is.

How to run:
python local_min_max.py MSFT --train 2023-01-01,now --wsize 10 20 50
'''