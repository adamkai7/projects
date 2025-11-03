## AWS Lambda Projects

These are two AWS Lambda projects I worked on during the summer of 2025.

### Project 1: S&P 500 Earnings Downloader
Files: `lambda_function.py`, `sp500-queue-loader.py`

This Lambda downloads earnings data for S&P 500 tickers from Yahoo Finance.
It’s designed to run daily via AWS EventBridge.

### Project 2: ETF Holdings Downloader
File: `s3_link_downloader.py`

This Lambda takes a text file of ETF data source links and downloads the daily holdings data for each ETF, also via EventBridge.

