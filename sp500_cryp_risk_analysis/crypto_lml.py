# Cryptocurrency Risk Analysis: Volatility & Correlation vs Traditional Assets
# Author: Adam Kainikara
# Date: December 2025
# Goal: Quantify Bitcoin and Ethereum risk relative to S&P 500 and Gold,
#       with focus on market stress periods and institutional  narratives

from numpy import *
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 1. Configuration & Constants
tickers_l = ['BTC-USD', 'ETH-USD', 'SPY', 'GLD']
# List of tickers
start_date = '2022-12-01'
# 3 years of data
end_date = datetime.today().strftime('%Y-%m-%d')

window = 30
# Rolling window for vol & corr
annualization_factor = sqrt(252)

# In the last 3 years, these where several of the big market stress events that happend. Analysis
# will be done and these dates will be annotated
events_d = {
    '2022-11-09': 'FTX Collapse',
    '2023-03-10': 'SVB Crisis',
    '2024-01-10': 'BTC ETF Approval',
    '2024-11-05': 'US Election'
}

# Data Download from yahoo finance
print("Downloading price data")

# how yahoo finance works is that the parameter auto_adjust=True needs to be added to get the data as
# directly adjusted close prices (single level columns). For more see the yfin documentation

full_data = yf.download(tickers_l, start=start_date, end=end_date, progress=False, auto_adjust=True)

# With auto_adjust=True, the 'Close' column is already adjusted
prices_df = full_data['Close']

# Forward-fill/Interpolate missing values for days like weekends or holidays
prices_df = prices_df.ffill()

# Daily Returns & Rolling Volatility
returns_df = prices_df.pct_change().dropna()
# Annualized rolling 30-day volatility
volatility_df = returns_df.rolling(window).std() * annualization_factor


# 4. Rolling Window Correlation comparing BTC vs SPY (ETF that tracks SP500)
btc_spy_corr_s = returns_df['BTC-USD'].rolling(window).corr(returns_df['SPY'])

# Additional Metrics
# Average volatility (in percent)
avg_vol_s = volatility_df.mean() * 100
# BTC beta to SPY
cov_btc_spy = returns_df['BTC-USD'].cov(returns_df['SPY'])
var_spy = returns_df['SPY'].var()
btc_beta = cov_btc_spy / var_spy
# Volatility ratio BTC vs SPY
vol_ratio = avg_vol_s['BTC-USD'] / avg_vol_s['SPY']

# Plotting
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

# Volatility Comparison
fig, ax1 = plt.subplots()

ax1.plot(volatility_df['BTC-USD'], label='Bitcoin', color='#f7931a', linewidth=2)
ax1.plot(volatility_df['ETH-USD'], label='Ethereum', color='#627eea', linewidth=2)
ax1.plot(volatility_df['SPY'], label='S&P 500', color='#1f77b4', linewidth=2)
ax1.plot(volatility_df['GLD'], label='Gold', color='#d4af37', linewidth=2)

ax1.set_title('30-Day Rolling Annualized Volatility (2022–2025)', fontsize=16)
ax1.set_ylabel('Annualized Volatility (%)')
ax1.legend(loc='upper left')

# Annotate events
for date_str, label in events_d.items():
    date = pd.to_datetime(date_str)
    if date in volatility_df.index:
        ax1.axvline(x=date, color='gray', linestyle='--', alpha=0.6)
        ax1.text(date, ax1.get_ylim()[1]*0.95, label,
                 rotation=90, verticalalignment='top', fontsize=9)

plt.tight_layout()
plt.savefig('vol_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# BTC-SPY Correlation
fig, ax2 = plt.subplots()
ax2.plot(btc_spy_corr_s, color='#f7931a', linewidth=2.5)
ax2.set_title('30-Day Rolling Correlation: Bitcoin vs S&P 500', fontsize=16)
ax2.set_ylabel('Correlation')

# Event markers
for date_str, label in events_d.items():
    date = pd.to_datetime(date_str)
    if date in btc_spy_corr_s.index:
        ax2.axvline(x=date, color='gray', linestyle='--', alpha=0.6)
        ax2.text(date, ax2.get_ylim()[1]*0.95, label,
                 rotation=90, verticalalignment='top', fontsize=9)

plt.tight_layout()
plt.savefig('btc_spy_corr.png', dpi=300, bbox_inches='tight')
plt.show()

#
print("\nKey Findings")
print(f"Average Annualized Volatility:")
print(f"  BTC: {avg_vol_s['BTC-USD']:.1f}%")
print(f"  ETH: {avg_vol_s['ETH-USD']:.1f}%")
print(f"  SPY: {avg_vol_s['SPY']:.1f}%")
print(f"  GLD: {avg_vol_s['GLD']:.1f}%")
print(f"\nBTC Volatility Ratio vs SPY: {vol_ratio:.1f}x")
print(f"BTC Beta to SPY: {btc_beta:.2f}")
