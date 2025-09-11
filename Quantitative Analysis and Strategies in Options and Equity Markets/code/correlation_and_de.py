import os
from numpy import *
from volta.utils import expand_path, time_to_slice, parse_date_range
from volta.cmdparser import CommandParser
from niven.data import load_catalog, load_aligned_series, StockDataset
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from itertools import combinations
from scipy.optimize import differential_evolution


'''
created by:
Adam Kainikara

This code analyzes stock correlations by computing Pearson correlation coefficients between a
reference stock and a broader universe of stocks using their log returns. The analysis focuses on
identifying stocks that exhibit strong positive correlation, strong negative correlation, or neutral
correlation relative to the chosen benchmark stock.

To run this file here is an example

python correlation_and_de.py -i sp500 --r MSFT --k 10 | tee /tmp/output.txt

the -i sp500 is what the stocks are eing compared with to the reference

--r is the reference stock, in this case msft

--k is the groups of stocks related to eacother part using  DE 

This section will discuss the second
component, which uses Differential Evolution (DE) which is an optimization algorithm.
In this analysis, DE is used to select a group of stocks that are most strongly correlated with each
other. Given a correlation matrix of all stocks, the algorithm searches for exactly k stocks whose
pairwise correlations maximize the overall similarity within the group. Each candidate solution is
treated as a binary vector (values above 0.5 indicate selection), and the objective function evaluates
how well the chosen group maximizes total correlation.

When you get 10 tickers from running DE, it means the algorithm found a group of 10 stocks that together have the highest total 
mutual correlation among themselves based on your correlation matrix C. 
In other words, these 10 stocks tend to move very similarly.
'''
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
    cmdparser.arg('--r', type=str, default='SPY', help='Reference symbol for correlation')
    cmdparser.arg('--t', type=float, default=0.7, help='T for high/low correlation classification')
    cmdparser.arg('--k', type=int, default=5, help='Number of stocks to select via DE')

    args = cmdparser.eval()

    args.train_t = parse_date_range(args.train, dt_unit='D')
    args.symbol_l = [symbol.upper()
                     for symbol in args.symbol_l]  # TD API expects upper case

    if args.input:
        args.symbol_l = unique(
            args.symbol_l + list(load_catalog(args.input, args.res_dir)))

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


def using_corrcoef(x_v, y_a):
    return corrcoef(x_v, y_a)


def extract_correlation_pairs2(x_a, ads, sym_v, reference_symbol, t=0.3):
    spy_idx = ads.sym_index_d[reference_symbol]
    spy_correlations = x_a[spy_idx, :]

    high_corr_l = []
    low_corr_l = []
    notmuch_corr_l = []

    for i, symbol in enumerate(sym_v):
        if i == spy_idx:
            continue  # skip SPY itself

        corr = spy_correlations[i]
        if corr > t:
            high_corr_l.append((symbol, corr))
        elif corr < -t:
            low_corr_l.append((symbol, corr))
        else:
            notmuch_corr_l.append((symbol, corr))

    return high_corr_l, low_corr_l, notmuch_corr_l


def select_topk_de(C, symbols_v, k=5, seed=0):
    n = C.shape[0]

    def objective(x):
        x_bin = (x > 0.5).astype(float)
        selected_count = int(x_bin.sum())
        if selected_count != k:
            return 1e6 + abs(selected_count - k) * 1e4
        return -x_bin @ C @ x_bin

    bounds = [(0, 1)] * n

    result = differential_evolution(
        objective,
        bounds,
        strategy='best1bin',
        maxiter=1000,
        popsize=15,
        tol=1e-7,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
        disp=True,
        polish=True,
    )

    x_bin = (result.x > 0.5).astype(bool)
    selected_symbols = [symbols_v[i] for i in range(n) if x_bin[i]]

    if len(selected_symbols) != k:
        print(f"\n[WARNING] DE returned {len(selected_symbols)}.")
        weight_idx = result.x.argsort()[::-1][:k]
        selected_symbols = [symbols_v[i] for i in weight_idx]
        x_bin = np.zeros(n, dtype=bool)
        x_bin[weight_idx] = True

    return selected_symbols, x_bin

'''

functions for visuals for report all analysis is done above
functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above

functions for visuals for report all analysis is done above


'''

def plot_correlation_heatmap(correlation_matrix, symbols_v, reference_symbol, reference_idx, 
                           top_k=20):
    ref_correlations = correlation_matrix[reference_idx, :]
    abs_correlations = np.abs(ref_correlations)
    top_indices = np.argsort(abs_correlations)[-top_k:][::-1]

    subset_corr_matrix = correlation_matrix[np.ix_(top_indices, top_indices)]
    subset_symbols = [symbols_v[i] for i in top_indices]

    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(subset_corr_matrix, dtype=bool), k=1)

    sns.heatmap(subset_corr_matrix, 
                mask=mask,
                annot=True, 
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                xticklabels=subset_symbols,
                yticklabels=subset_symbols,
                cbar_kws={"shrink": .8})

    plt.title(f'Correlation Heatmap: Top {top_k} Stocks Most Correlated with {reference_symbol}', 
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    return subset_symbols, subset_corr_matrix


def plot_de_results_heatmap(correlation_matrix, all_symbols, selected_symbols, objective_value):
    selected_indices = [all_symbols.tolist().index(s) for s in selected_symbols]
    submatrix = correlation_matrix[ix_(selected_indices, selected_indices)]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        submatrix,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        xticklabels=selected_symbols,
        yticklabels=selected_symbols,
    )

    plt.title(
        f'Correlation Matrix of the {len(selected_symbols)} Stocks Selected by DE\n'
        f'Objective Value (Sum of Correlations): {objective_value:.4f}',
        fontsize=14,
        fontweight='bold'
    )
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_correlation_histogram(correlation_matrix, reference_symbol, symbols_v, reference_idx, 
                             bins=50, show_stats=True):
    ref_correlations = correlation_matrix[reference_idx, :]
    correlations_excluding_self = np.concatenate([
        ref_correlations[:reference_idx], 
        ref_correlations[reference_idx+1:]
    ])

    plt.figure(figsize=(12, 8))
    n, bins_edges, patches = plt.hist(correlations_excluding_self, bins=bins, 
                                     alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)

    for i, patch in enumerate(patches):
        bin_center = (bins_edges[i] + bins_edges[i+1]) / 2
        if bin_center > 0.7:
            patch.set_facecolor('darkgreen')
        elif bin_center < -0.7:
            patch.set_facecolor('darkred')
        elif bin_center > 0.3:
            patch.set_facecolor('lightgreen')
        elif bin_center < -0.3:
            patch.set_facecolor('lightcoral')
        else:
            patch.set_facecolor('lightgray')

    plt.axvline(x=0.7, color='green', linestyle='--', linewidth=2, alpha=0.8, label='High correlation (0.7)')
    plt.axvline(x=-0.7, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Low correlation (-0.7)')
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Zero correlation')

    if show_stats:
        mean_corr = np.mean(correlations_excluding_self)
        median_corr = np.median(correlations_excluding_self)
        std_corr = np.std(correlations_excluding_self)
        min_corr = np.min(correlations_excluding_self)
        max_corr = np.max(correlations_excluding_self)

        high_corr_count = np.sum(correlations_excluding_self > 0.7)
        low_corr_count = np.sum(correlations_excluding_self < -0.7)
        moderate_corr_count = np.sum(np.abs(correlations_excluding_self) <= 0.7)

        stats_text = f"""Statistics:
Mean: {mean_corr:.3f}
Median: {median_corr:.3f}
Std Dev: {std_corr:.3f}
Min: {min_corr:.3f}
Max: {max_corr:.3f}

Stock Counts:
High corr (>0.7): {high_corr_count}
Low corr (<-0.7): {low_corr_count}
Moderate (±0.7): {moderate_corr_count}
Total stocks: {len(correlations_excluding_self)}"""

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontfamily='monospace', fontsize=10)

    plt.xlabel('Correlation Coefficient', fontsize=12)
    plt.ylabel('Number of Stocks', fontsize=12)
    plt.title(f'Distribution of Correlations with {reference_symbol}\n({len(correlations_excluding_self)} stocks)', 
              fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(-1.1, 1.1)
    plt.show()

    return correlations_excluding_self


def main():
    args = parse_args()
    ads = StockDataset('data', *load_aligned_series(
        args.symbol_l,
        f'{args.data_dir}/{args.interval}',
        method='percentile'
    ))
    ads.report()
    
    if args.train != 'none':
        train_slice = parse_training_period(args, ads.date_v)
        tds = StockDataset('train', ads.sym_v, ads.data_a[:, train_slice])
        tds.report()
        date_v = array(ads.date_v)[train_slice]
        price_data_a = ads.price_a[:, train_slice]
    else:
        date_v = array(ads.date_v)
        price_data_a = ads.price_a
    
    symbols_v = ads.sym_v
    reference_symbol = args.r.upper()
    print(f"Using {reference_symbol} as reference")
    
    log_return_data_a = log_returns(price_data_a)
    reference_idx = ads.sym_index_d[reference_symbol]
    ref_log_returns_v = log_return_data_a[reference_idx]
    
    corcoefway_a = using_corrcoef(ref_log_returns_v, log_return_data_a)
    high_corr_l, low_corr_l, notmuch_corr_l = extract_correlation_pairs2(
        corcoefway_a, ads, symbols_v, reference_symbol, args.t
    )
    C = corrcoef(log_return_data_a) 

    print(f"\n=== Highly correlated stocks to {reference_symbol} (>{args.t}) ===")
    for sym, corr in high_corr_l:
        print(f"{sym}: {corr:.3f}")
    
    print(f"\n=== Negatively correlated stocks to {reference_symbol} (<-{args.t}) ===")
    for sym, corr in low_corr_l:
        print(f"{sym}: {corr:.3f}")
    
    print(f"\n=== Middle correlated stocks (-{args.t} to {args.t}) ===")
    for sym, corr in notmuch_corr_l:
        print(f"{sym}: {corr:.3f}")
    print(f"\n=== Generating correlation distribution histogram ===")
    correlations_data = plot_correlation_histogram(
        corcoefway_a, reference_symbol, symbols_v, reference_idx,
        bins=50
    )
    
    print(f"\n=== Generating correlation heatmap for top stocks ===")
    top_symbols, top_corr_matrix = plot_correlation_heatmap(
        C, symbols_v, reference_symbol, reference_idx, 
        top_k=20
    )
    
    print(f"\nRunning Differential Evolution to select top {args.k} mutually correlated stocks...")
    C = corrcoef(log_return_data_a)
    selected_symbols, x_bin = select_topk_de(C, symbols_v, k=args.k, seed=args.seed)
    
    print(f"\n=== Differential Evolution selected top {args.k} mutually correlated stocks ===")
    for sym in selected_symbols:
        print(sym)
    
    print(f"\n=== Correlation submatrix for DE-selected {args.k} stocks ===")
    selected_indices = [i for i, flag in enumerate(x_bin) if flag]
    selected_matrix = C[ix_(selected_indices, selected_indices)]
    print(selected_matrix)
    
    objective_value = selected_matrix.sum()
    print(f"\nDE Objective Value (sum of correlations in submatrix): {objective_value:.4f}")
    print("\nSymbols in DE-selected submatrix:")
    for i in selected_indices:
        print(f"{i}: {symbols_v[i]}")
    
    plot_de_results_heatmap(C, symbols_v, selected_symbols, objective_value)
    raise SystemExit

if __name__ == "__main__":
    main()

