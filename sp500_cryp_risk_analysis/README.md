# Cryptocurrency Risk Analysis: Volatility & Correlation Study

**Adam Kainikara** | December 23, 2025

## Overview
Independent quantitative analysis comparing the risk profiles of Bitcoin and Ethereum against traditional assets (S&P 500 and 
Gold) over 2022–2025.

This project extends my prior work in derivatives volatility modeling by applying similar time-series techniques to digital 
assets, examining whether Bitcoin behaves as "digital gold" or a risk-on equity-like asset.

## Key Findings
- Bitcoin average annualized volatility: **37.4%** (3.3× higher than S&P 500)
- Ethereum average volatility: **50.4%**
- S&P 500 volatility: **11.4%** | Gold volatility: **12.7%**
- Bitcoin beta to S&P 500: **0.89**
- Rolling BTC-S&P correlation ranged from ~0.2 to 0.7, spiking during stress events (FTX collapse, SVB crisis)

## Files
- `crypto_lml.py` – Main Python script (yfinance, pandas, matplotlib)
- `vol_comparison.png` – 30-day rolling annualized volatility across assets
- `btc_spy_corr.png` – Rolling correlation between Bitcoin and S&P 500
- `final_crypto_report.pdf` – Full technical report with methodology and analysis

## Usage
```bash
python crypto_lml.py


