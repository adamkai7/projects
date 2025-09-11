import optmodel as opt
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

'''
Created by Adam Kainikara

How to use:
python variable_iv_analysis.py


This code analyzes the performance of an options trading strategy by calculating the best and worst returns achievable under a 
set of controlled conditions.

Here we investigates the effect of increasing implied volatility (IV) on option returns. In particular, we simulate scenarios where 
IV rises between the time of purchase and the time of sale, which is often observed in the days leading up to significant events 
such as earnings announcements. Implied volatility represents the market’s expectation of future asset price fluctuations,
 and an increase in IV tends to increase the price of options which could lead to potential gains for the option holder.

 This code, in large, is based off my other file constant_iv_analysis.py
'''




def analyze_IV_sell_strategy(s0,percent_change,rwsize,dte_window,strike_pct,iv_buy,iv_jump,rf,q,price_floor):
    
    #determine best returns by buying at fixed iv_buy and selling at iv_sell in across strikes, DTEs, and price outcome
    
    strike_min = int(s0 * (1 - strike_pct)) #minimum strike used in the range of calculation
    strike_max = int(s0 * (1 + strike_pct)) #maximum strike used in the range of calculation

    #  scenarios
    multipliers = linspace(1 - percent_change, 1 + percent_change, 2 * int(percent_change * 100) + 1)

    #  list of possible sell IVs
    iv_sell_max = iv_buy * (1 + iv_jump)
    iv_sell_levels = arange(iv_buy, iv_sell_max + 1e-6, 0.01)

    results_l = []
    # loop over every iv_sell, strike, dte, final price
    for iv_sell in iv_sell_levels:
        for m in multipliers:
            s_fin = s0 * m
            for strike in range(strike_min, strike_max + 1):
                for dte in range(rwsize + 1, rwsize + dte_window + 1):
                    price_buy = opt.gbs_call(s0, strike, dte / 365, iv_buy, rf, q).value
                    price_sell = max([s_fin - strike, 0.0])

                    if price_buy >= price_floor and price_sell >= price_floor:
                        ret = (price_sell - price_buy) / price_buy
                        results_l.append({
                            'return': ret,
                            'strike': strike,
                            'dte': dte,
                            's_fin': s_fin,
                            'iv_buy': iv_buy,
                            'iv_sell': iv_sell,
                            'price_buy': price_buy,
                            'price_sell': price_sell,})
    rets = array([r['return'] for r in results_l])
    sorted_index_v = argsort(rets)
    return results_l, sorted_index_v


def print_results_l(results_l, sorted_index_v, top_n=5, bottom_n=5):
    def fmt(x): return f"{x:.3f}"
    print(f"Top {top_n} scenarios:")
    for rank, index_v in enumerate(sorted_index_v[-top_n:][::-1], 1):
        r = results_l[index_v]
        print(
            f"{rank}. Strike={r['strike']} | DTE={r['dte']} | "
            f"S_fin={fmt(r['s_fin'])} | IVbuy={fmt(r['iv_buy'])}->IVsell={fmt(r['iv_sell'])} | "
            f"Buy={fmt(r['price_buy'])} Sell={fmt(r['price_sell'])} | Return={fmt(r['return']*100)}%")
    print(f"Bottom {bottom_n} scenarios:")
    for rank, index_v in enumerate(sorted_index_v[:bottom_n], 1):
        r = results_l[index_v]
        print(
            f"{rank}. Strike={r['strike']} | DTE={r['dte']} | "
            f"S_fin={fmt(r['s_fin'])} | IVbuy={fmt(r['iv_buy'])}->IVsell={fmt(r['iv_sell'])} | "
            f"Buy={fmt(r['price_buy'])} Sell={fmt(r['price_sell'])} | Return={fmt(r['return']*100)}%")
def plot_return_surface(results_l, s0, save_prefix=None, show=True):
    '''
    Visual i made for report which created 3D surface plot of Return vs DTE vs Underlying movement '''
    
    # pull arrays
    ret = array([d['return'] for d in results_l]) * 100  # convert return to %
    dte = array([d['dte'] for d in results_l])
    s_fin = array([d['s_fin'] for d in results_l])

    # convert underlying into % move relative to s0
    px_change = (s_fin / s0 - 1.0) * 100  # % change in price

    # build grid (unique sorted values)
    unique_dte = sort(unique(dte))
    unique_px = sort(unique(px_change))

    # create surface array
    Z = full((len(unique_dte), len(unique_px)), nan)

    for rr, dd, pp in zip(ret, dte, px_change):
        i = where(unique_dte == dd)[0][0]
        j = where(unique_px == pp)[0][0]
        if isnan(Z[i, j]):
            Z[i, j] = rr
        else:
            Z[i, j] = (Z[i, j] + rr) / 2.0  # average if multiple results_l

    # meshgrid for plotting
    P, D = meshgrid(unique_px, unique_dte)

    # create plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # plot surface
    surf = ax.plot_surface(P, D, Z, cmap=cm.viridis, linewidth=0, antialiased=True)

    ax.set_xlabel("Underlying Change (%)")
    ax.set_ylabel("DTE")
    ax.set_zlabel("Return (%)")
    ax.set_title("Option Return Surface vs Underlying Move and DTE")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="Return (%)")
    plt.show()

def save_strategy_results_l(results_l, sorted_index_v, s0, filename="strategy_results_l.txt", top_n=10, bottom_n=10):
    #save the best and worst  scenarios to a text file

    def fmt(x): return f"{x: >8.3f}"
    def pct(x): return f"{x: >7.2f}%"

    with open(filename, "w") as f:
        header = (
            f"{'Rank':<5} {'Strike':<8} {'DTE':<6} {'S0':>10} {'S_fin':>10} "
            f"{'IVbuy':>8} {'IVsell':>8} {'Price Buy':>12} {'Price Sell':>12} {'Return':>10}")

        f.write(f"\nTop {top_n} Scenarios (Highest Returns):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for rank, index_v in enumerate(sorted_index_v[-top_n:][::-1], 1):
            r = results_l[index_v]
            f.write(
                f"{rank:<5} {r['strike']:<8} {r['dte']:<6} {fmt(s0)} {fmt(r['s_fin'])} "
                f"{fmt(r['iv_buy'])} {fmt(r['iv_sell'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
                f"{pct(r['return']*100):>10}\n")

        f.write(f"\nBottom {bottom_n} Scenarios (Lowest Returns):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")

        for rank, index_v in enumerate(sorted_index_v[:bottom_n], 1):
            r = results_l[index_v]
            f.write(
                f"{rank:<5} {r['strike']:<8} {r['dte']:<6} {fmt(s0)} {fmt(r['s_fin'])} "
                f"{fmt(r['iv_buy'])} {fmt(r['iv_sell'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
                f"{pct(r['return']*100):>10}\n"
            )


def main():
    # these are parameters used for the analysis and can be changed
    s0 = 556.22
    percent_change = 0.10
    rwsize = 20
    dte_window = 3
    strike_pct = 0.10
    iv_buy = 0.20
    iv_jump = 0.50
    rf = 0.0425
    q = 0.0043
    price_floor = 0.10

    results_l, index_v = analyze_IV_sell_strategy(s0,percent_change,rwsize,dte_window,strike_pct,iv_buy,iv_jump,rf,q,price_floor)
    print_results_l(results_l, index_v, top_n=4000, bottom_n=4000)
    #plot_iv_expansion_analysis_histograms(results_l, s0=556.22)
    #plot_iv_timing_analysis(results_l, s0=556.22)
    plot_return_surface(results_l, s0)
    
    save_strategy_results_l(results_l, index_v, s0 , filename="variable_iv_analysis_results_l.txt", top_n=27000, bottom_n=12300)
    print(type(results_l))
    print(type)
if __name__ == '__main__':
    main()
