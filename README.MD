Portfolio Optimization Tool built with Claude

A Python-based portfolio optimization program using Modern Portfolio Theory (MPT) to help investors maximize returns while minimizing risk.

Features

- Interactive Menu System - Easy-to-use command-line interface
- Portfolio Analysis - Calculate optimal asset allocations using:
  - Maximum Sharpe Ratio (best risk-adjusted returns)
  - Minimum Variance (lowest risk)
  - Efficient Frontier visualization
- Purchase Tracking - Track multiple purchases per stock with fractional shares
- Real-Time Performance - View current portfolio value and gains/losses
- Flexible Date Ranges - Custom date ranges for individual stocks or portfolio-wide
- Visual Analytics - Generate efficient frontier plots with matplotlib


Quick Start

```
bash
Install dependencies
pip3 install numpy pandas matplotlib scipy yfinance

Run the program
python3 portfolio_optimizer.py
```

Usage

1. Enter your stock tickers (e.g., AAPL, MSFT, GOOGL)
2. Optionally track your actual purchases with dates and prices
3. View optimal portfolio recommendations
4. Compare your current holdings to optimal allocations
5. Visualize the efficient frontier

Perfect for both beginner and advanced investors looking to optimize their stock portfolios.
