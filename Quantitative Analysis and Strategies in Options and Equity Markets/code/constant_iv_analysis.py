import optmodel as opt
from numpy import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


'''
constant_iv_analysis.py

Created by Adam Kainikara contact me for any questions 

To run this file just do python constant_iv_analysis.py

"""
    Analyze a buying call option strategy by simulating returns based on a range of final underlying prices, strike prices, and days to 
    expiry (DTE). The other params of the Black-Scholes equation (rf, iv, and dividend) are held constant for the purpose of the experiment. 
    Effectively I want to determine what combination of underlying price, final underlying, strikes, and DTE
    To see more details, please view the report write up. 
    """
'''


def analyze_option_strategy(initial_underlying, percent_change, max_days, window_size, strike_min, strike_max, strike_step=1, iv=0.2068,
    rf=0.0425, q=0.0043,price_floor=0.35):
    """
    Analyze a buying call option strategy by simulating returns based on a range of final underlying prices, strike prices, and days to 
    expiry (DTE). The other params of the Black-Scholes equation (rf, iv, and dividend) are held constant for the purpose of the experiment. 
    Effectively I want to determine what combination of underlying price, final underlying, strikes, and DTE
    To see more details, please view the report write up. 
    """
    multipliers_v = linspace(1 - percent_change, 1 + percent_change, int(percent_change * 100 * 2) + 1)  # this is for the
    #integer steps of the prices, the range generated is based off of the percent change (price band) defined by the user

    results_l = [] # this stores the results of the experiment 
    for price_multiplier in multipliers_v:
        final_underlying = initial_underlying * price_multiplier
        underlying_return = (final_underlying / initial_underlying) - 1

        for strike in range(strike_min, strike_max + 1, strike_step):
            t_buy_l = range(max_days, window_size, -window_size)
        # looping through various underlying and strike prices and combinations 
            for t_buy in t_buy_l:
                t_sell = t_buy - window_size # this creates the window size and dte calculation part of the experiment 
                price_buy = opt.gbs_call(initial_underlying, strike, t_buy / 365, iv, rf, q).value # this uses the Black-Scholes equation
                #to determine the option price that wit is being bought at
                if price_buy < price_floor:
                    continue #verifying that the option price is more than the price floor, which is set because low price options
                    #tend to be not liquid enough

                price_sell = opt.gbs_call(final_underlying, strike, max(t_sell, 0) / 365, iv, rf, q).value #sell price
                if price_sell < price_floor:
                    continue

                #determine if the result is in the money or out of the money
                moneyness_initial = (initial_underlying / strike) - 1 
                moneyness_label = "ITM" if moneyness_initial > 0.01 else "OTM" if moneyness_initial < -0.01 else "ATM"
                option_return = (price_sell - price_buy) / price_buy
                # these are all the stats/info that is saved, this can be used for data analysis later
                results_l.append({
                    'option_return': option_return,
                    'underlying_return': underlying_return,
                    'initial_underlying': initial_underlying,
                    'final_underlying': final_underlying,
                    'strike': strike,
                    'dte_buy': t_buy,
                    'dte_sell': t_sell,
                    'price_buy': price_buy,
                    'price_sell': price_sell,
                    'moneyness_initial': moneyness_initial,
                    'moneyness_label': moneyness_label,})

    

    returns_a = array([r['option_return'] for r in results_l])
    sorted_idex_v = argsort(returns_a)
    return results_l, returns_a, sorted_idex_v #arg sort the results_l so when it is printed, the highest/lowest returns are shown


def print_strategy_results_l(results_a, sorted_idex_v, top_n=5, bottom_n=5):
    # this function prints the best and worst scenarios
    def fmt(x): return f"{x: >7.2f}"
    def pct(x): return f"{x: >7.2%}"
    # setting decimals and what is shown in the output
    header = (
        f"{'Rank':<5} {'K':<5} {'BuyDTE':>7} {'Initial S':>10} {'Final S':>10} "
        f"{'Price Buy':>10} {'Price Sell':>11} {'Moneyness':>12} "
        f"{'Und. Return':>14} {'Opt. Return':>12}")

    print(f"\nTop {top_n} Scenarios (Highest Returns):")
    print(header)
    print("-" * len(header))
    for rank, idx in enumerate(sorted_idex_v[-top_n:][::-1], 1):
        r = results_a[idx]
        print(
            f"{rank:<5} {r['strike']:<5} {str(r['dte_buy'])+'d':>7} {fmt(r['initial_underlying'])} "
            f"{fmt(r['final_underlying'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
            f"{r['moneyness_label']:>12} {pct(r['underlying_return']):>14} {pct(r['option_return']):>12}")

    print(f"\nBottom {bottom_n} Scenarios (Lowest Returns):")
    print(header)
    print("-" * len(header))
    for rank, idx in enumerate(sorted_idex_v[:bottom_n], 1):
        r = results_a[idx]
        print(
            f"{rank:<5} {r['strike']:<5} {str(r['dte_buy'])+'d':>7} {fmt(r['initial_underlying'])} "
            f"{fmt(r['final_underlying'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
            f"{r['moneyness_label']:>12} {pct(r['underlying_return']):>14} {pct(r['option_return']):>12}")


'''
These where various visuals i made for the report and not related to intial project idea or code


'''


def save_strategy_results_l(results_l, sorted_idex_v, filename="strategy_results_l.txt", top_n=10, bottom_n=10):
    #saves the best and worst  scenarios"""
    def fmt(x): return f"{x: >7.2f}"
    def pct(x): return f"{x: >7.2%}"

    with open(filename, "w") as f:
        header = (
            f"{'Rank':<5} {'K':<5} {'BuyDTE':>7} {'Initial S':>10} {'Final S':>10} "
            f"{'Price Buy':>10} {'Price Sell':>11} {'Moneyness':>12} "
            f"{'Und. Return':>14} {'Opt. Return':>12}")
        f.write(f"\nTop {top_n} Scenarios (Highest Returns):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for rank, idx in enumerate(sorted_idex_v[-top_n:][::-1], 1):
            r = results_l[idx]
            f.write(
                f"{rank:<5} {r['strike']:<5} {str(r['dte_buy'])+'d':>7} {fmt(r['initial_underlying'])} "
                f"{fmt(r['final_underlying'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
                f"{r['moneyness_label']:>12} {pct(r['underlying_return']):>14} {pct(r['option_return']):>12}\n")

        f.write(f"\nBottom {bottom_n} Scenarios (Lowest Returns):\n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for rank, idx in enumerate(sorted_idex_v[:bottom_n], 1):
            r = results_l[idx]
            f.write(
                f"{rank:<5} {r['strike']:<5} {str(r['dte_buy'])+'d':>7} {fmt(r['initial_underlying'])} "
                f"{fmt(r['final_underlying'])} {fmt(r['price_buy'])} {fmt(r['price_sell'])} "
                f"{r['moneyness_label']:>12} {pct(r['underlying_return']):>14} {pct(r['option_return']):>12}\n"
            )
def plot_performance_analysis_histograms(results_l):
    #
    #Create  histograms that showcase  key findings from   report
    #- OTM vs ITM performance
    #- Impact of underlying price movements
    #- Premium cost vs returns
    #"""
    
    
    #  results_l to arrays for analysis
    option_returns_a = array([r['option_return'] for r in results_l])
    underlying_returns_a = array([r['underlying_return'] for r in results_l])
    moneyness_labels = [r['moneyness_label'] for r in results_l]
    initial_premiums_a = array([r['price_buy'] for r in results_l])
    
    # filter for reasonable display range (-200% to +500%)
    display_mask = (option_returns_a >= -2.0) & (option_returns_a <= 5.0)
    filtered_returns = option_returns_a[display_mask]
    filtered_underlying = array([underlying_returns_a[i] for i in range(len(underlying_returns_a)) if display_mask[i]])
    filtered_moneyness = [moneyness_labels[i] for i in range(len(moneyness_labels)) if display_mask[i]]
    filtered_premiums = array([initial_premiums_a[i] for i in range(len(initial_premiums_a)) if display_mask[i]])
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1  Returns by Moneyness (OTM vs ITM vs ATM) 
    otm_returns = [filtered_returns[i] for i in range(len(filtered_returns)) if filtered_moneyness[i] == 'OTM']
    itm_returns = [filtered_returns[i] for i in range(len(filtered_returns)) if filtered_moneyness[i] == 'ITM']
    atm_returns = [filtered_returns[i] for i in range(len(filtered_returns)) if filtered_moneyness[i] == 'ATM']
    
    bins = linspace(-2, 5, 50)
    ax1.hist(otm_returns, bins=bins, alpha=0.7, label=f'OTM (n={len(otm_returns)})', color='green', density=True)
    ax1.hist(itm_returns, bins=bins, alpha=0.7, label=f'ITM (n={len(itm_returns)})', color='red', density=True)
    ax1.hist(atm_returns, bins=bins, alpha=0.7, label=f'ATM (n={len(atm_returns)})', color='orange', density=True)
    ax1.axvline(0, color='black', linestyle='--', alpha=0.5)
    #  x-axis to percentage 
    ax1.set_xlabel('Option Return (%)')
    ax1.set_ylabel('Density')
    ax1.set_title('Return Distribution: OTM vs ITM vs ATM\n(OTM shows better upside potential)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # x-axis ticks to show percentages
    ax1_ticks = ax1.get_xticks()
    ax1.set_xticklabels([f'{int(x*100)}%' for x in ax1_ticks])
    
    # 2 Returns by Underlying Movement Direction 
    positive_underlying_mask = filtered_underlying > 0.05  # >5% underlying gain

    negative_underlying_mask = filtered_underlying < -0.05  # <-5% underlying loss
    neutral_underlying_mask = (filtered_underlying >= -0.05) & (filtered_underlying <= 0.05)
    
    positive_returns = filtered_returns[positive_underlying_mask]
    negative_returns = filtered_returns[negative_underlying_mask] 
    neutral_returns = filtered_returns[neutral_underlying_mask]
    
    ax2.hist(positive_returns, bins=bins, alpha=0.7, label=f'Underlying +5%+ (n={len(positive_returns)})', color='darkgreen', density=True)
    ax2.hist(negative_returns, bins=bins, alpha=0.7, label=f'Underlying -5%- (n={len(negative_returns)})', color='darkred', density=True)
    ax2.hist(neutral_returns, bins=bins, alpha=0.7, label=f'Underlying ±5% (n={len(neutral_returns)})', color='gray', density=True)
    ax2.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Option Return (%)')
    ax2.set_ylabel('Density')
    ax2.set_title('Return Distribution by Underlying Price Movement\n(Large favorable moves drive best returns)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # x-axis ticks to show percentages
    ax2_ticks = ax2.get_xticks()
    ax2.set_xticklabels([f'{int(x*100)}%' for x in ax2_ticks])
    
    # 3 returns by intial premium Cost -  
    low_premium_mask = filtered_premiums <= 2.0  # Low cost options
    high_premium_mask = filtered_premiums > 2.0   # Higher cost options
    
    low_premium_returns = filtered_returns[low_premium_mask]
    high_premium_returns = filtered_returns[high_premium_mask]
    
    ax3.hist(low_premium_returns, bins=bins, alpha=0.7, label=f'Low Premium ≤$2 (n={len(low_premium_returns)})', color='blue', density=True)
    ax3.hist(high_premium_returns, bins=bins, alpha=0.7, label=f'High Premium >$2 (n={len(high_premium_returns)})', color='purple', density=True)
    ax3.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Option Return (%)')
    ax3.set_ylabel('Density')
    ax3.set_title('Return Distribution by Initial Premium Cost\n(Low premium options offer better leverage)', fontsize=12)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # x-axis ticks to show percentages
    ax3_ticks = ax3.get_xticks()
    ax3.set_xticklabels([f'{int(x*100)}%' for x in ax3_ticks])
    
    # 4. 
    # Best is when OTM + positive underlying + low premium
    best_conditions_mask = (
        array([filtered_moneyness[i] == 'OTM' for i in range(len(filtered_moneyness))]) &
        (filtered_underlying > 0.10) &  # >10% underlying move
        (filtered_premiums <= 1.5))
    
    # Worst case: ITM + Negative underlying + High premium  
    worst_conditions_mask = (
        array([filtered_moneyness[i] == 'ITM' for i in range(len(filtered_moneyness))]) &
        (filtered_underlying < -0.05) &  # <-5% underlying move
        (filtered_premiums > 2.0))
    
    best_scenario_returns = filtered_returns[best_conditions_mask]
    worst_scenario_returns = filtered_returns[worst_conditions_mask]
    
    ax4.hist(best_scenario_returns, bins=bins, alpha=0.8, label=f'Best Conditions (n={len(best_scenario_returns)})', color='gold', density=True)
    
    
    ax4.hist(worst_scenario_returns, bins=bins, alpha=0.8, label=f'Worst Conditions (n={len(worst_scenario_returns)})', color='maroon', density=True)
    ax4.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Option Return (%)')
    ax4.set_ylabel('Density')
    ax4.set_title('Best vs Worst Scenario Comparison\n(OTM + Large Gains + Low Premium vs ITM + Losses + High Premium)', fontsize=12)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    #  x-axis ticks to show percentages
    ax4_ticks = ax4.get_xticks()
    ax4.set_xticklabels([f'{int(x*100)}%' for x in ax4_ticks])
    
    plt.tight_layout()
    plt.suptitle('Options Strategy Performance Analysis - Key Findings Visualization', fontsize=16, y=1.02)
    
    #  summary statistics
    print(f"\nHISTOGRAM ANALYSIS SUMMARY")
    print(f"OTM Average Return: {mean(otm_returns):.2%}, Std: {std(otm_returns):.2%}")
    print(f"ITM Average Return: {mean(itm_returns):.2%}, Std: {std(itm_returns):.2%}")
    print(f"Positive Underlying Avg Return: {mean(positive_returns):.2%}")
    print(f"Negative Underlying Avg Return: {mean(negative_returns):.2%}")
    print(f"Low Premium Avg Return: {mean(low_premium_returns):.2%}")
    print(f"High Premium Avg Return: {mean(high_premium_returns):.2%}")
    if len(best_scenario_returns) > 0:
        print(f"Best Scenario Avg Return: {mean(best_scenario_returns):.2%}")
    if len(worst_scenario_returns) > 0:
        print(f"Worst Scenario Avg Return: {mean(worst_scenario_returns):.2%}")
    
    plt.show()



def main():
    # these are parameters that can be changed to adjust to the experiment.
    # a further thing I would like to work on is adjusting to run these parms on the command line so one does not need
    # to open the code file and edit it
    initial_underlying = 556.22
    percent_change = 0.15
    max_days = 100
    window_size = 10
    strike_min = 400
    strike_max = 700
    strike_step = 5  

    results_l, results_a, sorted_idex_v = analyze_option_strategy(initial_underlying, percent_change, max_days, window_size, strike_min, strike_max, strike_step=strike_step)
    #print(len(results_l))

    
        # printed results_l  show the absolute best/worst trades
        #print_strategy_results_l(results_a, sorted_idex_v, top_n=7000, bottom_n=7000)
        #run_visualizations(results_a)
    save_strategy_results_l(results_l, sorted_idex_v, filename="strategy_results_l.txt", top_n=7000, bottom_n=7000)
    print_strategy_results_l(results_l, sorted_idex_v, top_n=20, bottom_n=20)
    #plot_histogram_returns_by_moneyness(results_a)
    #plot_histogram_returns_by_moneyness(results_a, filter_realistic=True)
    plot_performance_analysis_histograms(results_l)
    print(type(results_a))
    print(type(sorted_idex_v))
    

if __name__ == '__main__':
    main()


'''

How to use:
python variable_iv_analysis.py


This code analyzes the performance of an options trading strategy by calculating the best and worst returns achievable under a 
set of controlled conditions. In this case all components of Black-Scholes are variable except for IV.
'''