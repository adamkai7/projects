import os
from numpy import *
import matplotlib.pyplot as plt

from datetime import datetime
from volta.utils import expand_path, time_to_slice, parse_date_range
from volta.cmdparser import CommandParser
from niven.data import load_catalog, load_aligned_series, StockDataset

'''
created by Adam Kainikara
This analysis examines the relationship between market volatility (risk) and re-
turns (reward) in financial assets. A sliding window methodology is applied to test whether stocks
with higher volatility deliver proportionally higher returns—a common assumption when creating
portfolios.

Examples:
1) python risk_vs_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 5 --xwsize 1 --show
2) python risk_vs_reward.py AAPL --refdate 2025-04-01 --swsize 30 --rwsize 5 --xwsize 10 --show

to do on one stock for example AAPL do
python risk_vs_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show AAPL --xwsize 20


to do on sp500 do
python scatter_risk_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 5 --show

 python scatter_risk_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show AAPL
    python risk_vs_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 5 --show
    python risk_vs_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show MSFT
    python risk_vs_reward.py -i sp500 --refdate 2022-04-03 --swsize 50 --rwsize 50 --show --xwsize 1
    python risk_vs_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 50 --show --xwsize 1
    python risk_vs_reward.py -i sp500 --refdate 2023-04-13 --swsize 50 --rwsize 50 --show --xwsize 1


'''
def parse_args():
    #data loader
    WORK_DIR = os.getenv('WORK_DIR') or expand_path('/data/tmp/indus/yfin')
    RES_DIR = expand_path('~/kalari/indus/resources')
    parser = CommandParser()
    parser.arg('--data_dir', type=str,default=f'{WORK_DIR}/data/stocks/history', help='Data directory')
    parser.arg('--res_dir', type=str,default=RES_DIR, help='Resource directory')
    parser.arg('--interval', type=str, default='1d', help='Sampling interval')
    parser.arg('-i', '--input', type=str, action='append', default=[],help='Catalog names (e.g., sp500, etf)')
    parser.arg('--refdate', type=str, required=True,help='Reference date (YYYY-MM-DD)')
    parser.arg('--swsize', type=int, required=True,help='Sliding window size (days) before refdate for std')
    parser.arg('--rwsize', type=int, required=True,help='Return window size (days) after refdate')
    parser.arg('--xwsize', type=int, required=True,help='xwsize window size (days) after refdate')

    parser.arg('--show', action='store_true', help='Display plot')
    parser.arg('symbol_l', nargs='*', help='Explicit symbols')
    args = parser.eval()
    args.train_t = parse_date_range('none', dt_unit='D')  # 
    return args


def calculate_risk_reward(price_a, date_v, refdate, swsize, rwsize, xwsize):
    """
    - risk: std dev of log returns over the previous swsize days, scaled to rwsize
    - reward: log return over the next rwsize days
    """
    dates_v = array(date_v).astype('datetime64[D]')
    ref64_dt = datetime64(refdate, 'D')

    idx_v = where(dates_v == ref64_dt)[0] #array of indices where dates_v matches the reference date
    if idx_v.size == 0:
        raise ValueError(f"Ref date {refdate} not found in date_v")
    ref_index = idx_v[0] # integer index of the reference date in the dates array
    print(f'dates_v{type(dates_v)}')
    print(f'ref64_dt{type(ref64_dt)}')
    print(f'idx_v{type(idx_v)}')

    print(f'ref_index{type(ref_index)}')
    n, t = price_a.shape
    print(f'ref_index{type(n)}')
    print(f'ref_index{type(t)}')

    # detrmine the valid reference days
    #valid_idx_v = arange(ref_index, t - rwsize)
    #valid_idx_v = valid_idx_v[valid_idx_v - swsize >= 0]

    valid_idx_l = [] #list of valid indices starting at ref_index where enough past and future data exist to calculate risk and reward
    for shift in range(xwsize):
        i = ref_index + shift
        if i - swsize < 0 or i + rwsize >= t:
            continue
        valid_idx_l.append(i)
    #output arrays
    print(f'valid_idx_l{type(valid_idx_l)}')

    risk_v = zeros((n, len(valid_idx_l)))
    reward_v = zeros((n, len(valid_idx_l)))

    for j in range(len(valid_idx_l)):
        i = valid_idx_l[j]
        logp_a = log(price_a[:, i-swsize:i+1])
        #print(f'logp_a{type(logp_a)}')
        ret_a = logp_a[:, 1:] - logp_a[:, :-1]
        retstd_v = ret_a.std(axis=1) * (rwsize ** 0.5)  #needed so the standard deviation is calculated across the time window 
        #for each ticker separately, instead of over all tickers and all days
        future_return_v = (price_a[:, i+rwsize] / price_a[:, i]) - 1
        print('*'*10)
        #print(f'retstd_v{type(retstd_v)}')
        #print(f'future_return_v{type(future_return_v)}')


        risk_v[:, j] = retstd_v
        reward_v[:, j] = future_return_v

    print(f'risk_v{risk_v}')
    print(f'risk_v{reward_v}')

    return risk_v, reward_v


#    return risk_v, reward_v

def plot_risk_vs_reward(risk_v, reward_v, swsize, rwsize, input_label='', show=False):
    #visual
    plt.figure(figsize=(8, 6))
    plt.scatter(risk_v, reward_v, alpha=0.6)
    plt.xlabel(f'Std Dev over {swsize} days before')
    plt.ylabel(f'Return over {rwsize} days after')
    title = 'Risk vs Reward'
    if input_label:
        title += f' ({input_label})'
    plt.title(title)
    plt.grid(True)
    x = linspace(0, risk_v.max(), 10)
    plt.plot(x,x, label='y=x')
    plt.plot(x,2*x, label='y=2x')
    plt.plot(x,3*x, label='y=3x')   
    plt.plot(x,-x, label='y=x')
    plt.plot(x,-2*x, label='y=2x')
    plt.plot(x,-3*x, label='y=3x')  

    if show:
        plt.show()



def main():
    args = parse_args()

    if args.input:
        syms_l = unique([s.upper() for s in args.symbol_l] + list(load_catalog(args.input, args.res_dir)))
    else:
        syms_l = [s.upper() for s in args.symbol_l]

    ads = StockDataset('data', *load_aligned_series(syms_l,f"{args.data_dir}/{args.interval}",method='percentile'))
    date_v = ads.date_v
    price_a = ads.price_a
    print(f'typeofdate  {type(date_v)}')
    print(f'typeofprice {type(price_a)}')
    print(f'typeofsyms_l {type(syms_l)}')

    risk_v, reward_v = calculate_risk_reward(price_a, date_v, args.refdate, args.swsize, args.rwsize, args.xwsize)
    print(f'typeofrisk {type(risk_v)}')
    print(f'typeofreward {type(reward_v)}')

    input_label = ', '.join(args.input) if args.input else ' '.join(args.symbol_l)
    plot_risk_vs_reward(risk_v, reward_v, args.swsize, args.rwsize, input_label=input_label, show=args.show)

if __name__ == '__main__':
    main()

'''
created by Adam Kainikara
This analysis examines the relationship between market volatility (risk) and re-
turns (reward) in financial assets. A sliding window methodology is applied to test whether stocks
with higher volatility deliver proportionally higher returns—a common assumption when creating
portfolios.

Examples:
to do on one stock for example AAPL do
python risk_vs_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show AAPL --xwsize 20


to do on sp500 do
python scatter_risk_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 5 --show

 python scatter_risk_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show AAPL
    python risk_vs_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 5 --show
    python risk_vs_reward.py --refdate 2023-04-03 --swsize 50 --rwsize 5 --show MSFT
    python risk_vs_reward.py -i sp500 --refdate 2022-04-03 --swsize 50 --rwsize 50 --show --xwsize 1
    python risk_vs_reward.py -i sp500 --refdate 2023-04-03 --swsize 50 --rwsize 50 --show --xwsize 1
    python risk_vs_reward.py -i sp500 --refdate 2023-04-13 --swsize 50 --rwsize 50 --show --xwsize 1


'''