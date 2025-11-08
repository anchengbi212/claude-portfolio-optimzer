import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None, custom_dates=None, purchases=None):
        """
        Initialize portfolio optimizer with stock tickers
        
        Parameters:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD)
        custom_dates (dict): Optional dict with ticker-specific dates
        purchases (dict): Optional dict tracking actual purchases
                         Format: {'AAPL': [{'date': '2024-01-01', 'shares': 10, 'price': 150}, ...]}
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.custom_dates = custom_dates or {}
        self.purchases = purchases or {}
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.price_data = None
        
    def fetch_data(self):
        """Download historical stock data and calculate returns"""
        print(f"\n📊 Fetching data for {', '.join(self.tickers)}...")
        
        all_data = {}
        
        # If custom dates are specified, fetch each ticker individually
        if self.custom_dates:
            print("📅 Using custom date ranges for individual stocks...")
            for ticker in self.tickers:
                if ticker in self.custom_dates:
                    start, end = self.custom_dates[ticker]
                    print(f"  {ticker}: {start} to {end}")
                else:
                    start, end = self.start_date, self.end_date
                    print(f"  {ticker}: {start} to {end} (default)")
                
                try:
                    ticker_data = yf.download(ticker, start=start, end=end, progress=False)
                    if not ticker_data.empty:
                        if 'Adj Close' in ticker_data.columns:
                            all_data[ticker] = ticker_data['Adj Close']
                        else:
                            all_data[ticker] = ticker_data['Close']
                except Exception as e:
                    print(f"  ⚠️ Warning: Could not fetch data for {ticker}: {e}")
            
            if not all_data:
                raise ValueError("No data downloaded for any ticker.")
            
            # Combine all data into a single DataFrame
            data = pd.DataFrame(all_data)
            
            # Find common date range (intersection of all dates)
            print(f"\n📊 Aligning data to common dates...")
            data = data.dropna()  # Keep only dates where all stocks have data
            
            if data.empty:
                raise ValueError("No overlapping dates found for all tickers. Try adjusting date ranges.")
            
            print(f"✅ Common date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        else:
            # Standard approach: fetch all tickers with same date range
            raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
            
            if raw_data.empty:
                raise ValueError("No data downloaded. Check your ticker symbols and internet connection.")
            
            # Handle different data structures based on number of tickers
            if len(self.tickers) == 1:
                # Single ticker case
                if isinstance(raw_data, pd.Series):
                    # Series with no column names
                    data = raw_data.to_frame(name=self.tickers[0])
                elif 'Adj Close' in raw_data.columns:
                    data = raw_data[['Adj Close']].copy()
                    data.columns = self.tickers
                elif 'Close' in raw_data.columns:
                    data = raw_data[['Close']].copy()
                    data.columns = self.tickers
                else:
                    # Assume first column is the price data
                    data = raw_data.iloc[:, [0]].copy()
                    data.columns = self.tickers
            else:
                # Multiple tickers
                try:
                    if isinstance(raw_data.columns, pd.MultiIndex):
                        # Multi-level columns
                        if 'Adj Close' in raw_data.columns.get_level_values(0):
                            data = raw_data.xs('Adj Close', axis=1, level=0)
                        elif 'Close' in raw_data.columns.get_level_values(0):
                            data = raw_data.xs('Close', axis=1, level=0)
                        else:
                            # Take first level
                            data = raw_data.xs(raw_data.columns.get_level_values(0)[0], axis=1, level=0)
                    else:
                        # Single level columns
                        if 'Adj Close' in raw_data.columns:
                            data = raw_data[['Adj Close']].copy()
                        elif 'Close' in raw_data.columns:
                            data = raw_data[['Close']].copy()
                        else:
                            data = raw_data.copy()
                except Exception as e:
                    print(f"Warning: Could not extract price data normally: {e}")
                    # Fallback: try to use raw_data as is
                    data = raw_data.copy()
            
            # Ensure data is a DataFrame
            if isinstance(data, pd.Series):
                data = data.to_frame()
            
            # Remove any columns with all NaN values
            data = data.dropna(axis=1, how='all')
        
        # Store price data for portfolio value calculations
        self.price_data = data.copy()
        
        # Ensure we have valid column names
        if hasattr(data.columns, 'tolist'):
            self.tickers = [str(col) for col in data.columns.tolist()]
        else:
            self.tickers = [str(col) for col in data.columns]
        
        # Remove any duplicate tickers
        self.tickers = list(dict.fromkeys(self.tickers))
        
        # Reindex data to match tickers
        if len(self.tickers) > 0:
            data = data[self.tickers]
        
        if len(self.tickers) == 0 or data.empty:
            raise ValueError("No valid data after cleaning. Check your ticker symbols and date range.")
        
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print(f"✅ Data fetched successfully! ({len(self.returns)} trading days)")
        print(f"📈 Tickers included: {', '.join(self.tickers)}")
        return self.returns
    
    def calculate_current_portfolio_value(self):
        """Calculate current value and performance of actual purchases"""
        if not self.purchases or self.price_data is None:
            return None
        
        results = {}
        total_invested = 0
        total_current_value = 0
        
        # Get latest price for each ticker
        latest_prices = self.price_data.iloc[-1]
        
        for ticker, purchase_list in self.purchases.items():
            if ticker not in latest_prices.index:
                continue
                
            ticker_invested = 0
            ticker_shares = 0
            
            for purchase in purchase_list:
                shares = purchase['shares']
                price = purchase['price']
                ticker_invested += shares * price
                ticker_shares += shares
            
            current_price = latest_prices[ticker]
            current_value = ticker_shares * current_price
            gain_loss = current_value - ticker_invested
            gain_loss_pct = (gain_loss / ticker_invested * 100) if ticker_invested > 0 else 0
            
            results[ticker] = {
                'shares': ticker_shares,
                'invested': ticker_invested,
                'current_value': current_value,
                'current_price': current_price,
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            }
            
            total_invested += ticker_invested
            total_current_value += current_value
        
        total_gain_loss = total_current_value - total_invested
        total_gain_loss_pct = (total_gain_loss / total_invested * 100) if total_invested > 0 else 0
        
        return {
            'holdings': results,
            'total_invested': total_invested,
            'total_current_value': total_current_value,
            'total_gain_loss': total_gain_loss,
            'total_gain_loss_pct': total_gain_loss_pct
        }
    
    def display_current_portfolio(self):
        """Display current portfolio performance"""
        portfolio_value = self.calculate_current_portfolio_value()
        
        if portfolio_value is None:
            print("\n⚠️ No purchase data available. Use Option 6 to enter your purchases.")
            return
        
        print("\n" + "="*70)
        print("YOUR CURRENT PORTFOLIO PERFORMANCE")
        print("="*70)
        
        for ticker, data in portfolio_value['holdings'].items():
            sign = "📈" if data['gain_loss'] >= 0 else "📉"
            print(f"\n{sign} {ticker}")
            print(f"  Shares: {data['shares']:.2f}")
            print(f"  Invested: ${data['invested']:,.2f}")
            print(f"  Current Value: ${data['current_value']:,.2f}")
            print(f"  Current Price: ${data['current_price']:.2f}")
            print(f"  Gain/Loss: ${data['gain_loss']:,.2f} ({data['gain_loss_pct']:+.2f}%)")
        
        print("\n" + "-"*70)
        total_sign = "📈" if portfolio_value['total_gain_loss'] >= 0 else "📉"
        print(f"{total_sign} TOTAL PORTFOLIO")
        print(f"  Total Invested: ${portfolio_value['total_invested']:,.2f}")
        print(f"  Current Value: ${portfolio_value['total_current_value']:,.2f}")
        print(f"  Total Gain/Loss: ${portfolio_value['total_gain_loss']:,.2f} ({portfolio_value['total_gain_loss_pct']:+.2f}%)")
        print("="*70)
    
    def portfolio_performance(self, weights):
        """Calculate portfolio return and volatility"""
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std
    
    def negative_sharpe(self, weights, risk_free_rate=0.02):
        """Calculate negative Sharpe ratio (for minimization)"""
        ret, std = self.portfolio_performance(weights)
        sharpe = (ret - risk_free_rate) / std
        return -sharpe
    
    def max_sharpe_ratio(self, risk_free_rate=0.02):
        """Find portfolio with maximum Sharpe ratio"""
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        result = minimize(self.negative_sharpe, init_guess, 
                         args=(risk_free_rate,),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def min_variance(self):
        """Find minimum variance portfolio"""
        num_assets = len(self.tickers)
        
        def portfolio_variance(weights):
            return self.portfolio_performance(weights)[1]
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        result = minimize(portfolio_variance, init_guess,
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def efficient_frontier(self, num_portfolios=50):
        """Generate efficient frontier"""
        num_assets = len(self.tickers)
        results = np.zeros((3, num_portfolios))
        weights_array = []
        
        min_vol_port = self.min_variance()
        max_sharpe_port = self.max_sharpe_ratio()
        
        min_ret = self.portfolio_performance(min_vol_port.x)[0]
        max_ret = self.portfolio_performance(max_sharpe_port.x)[0]
        
        target_returns = np.linspace(min_ret, max_ret, num_portfolios)
        
        for i, target_return in enumerate(target_returns):
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: self.portfolio_performance(x)[0] - target_return}
            )
            bounds = tuple((0, 1) for _ in range(num_assets))
            init_guess = num_assets * [1. / num_assets]
            
            result = minimize(lambda x: self.portfolio_performance(x)[1], init_guess,
                            method='SLSQP', bounds=bounds, constraints=constraints)
            
            results[0, i] = self.portfolio_performance(result.x)[0]
            results[1, i] = self.portfolio_performance(result.x)[1]
            results[2, i] = (results[0, i] - 0.02) / results[1, i]
            weights_array.append(result.x)
        
        return results, weights_array
    
    def plot_efficient_frontier(self, num_portfolios=50, show_random=True):
        """Plot efficient frontier with optimal portfolios"""
        print("\n📊 Generating efficient frontier plot...")
        
        frontier, weights = self.efficient_frontier(num_portfolios)
        
        max_sharpe = self.max_sharpe_ratio()
        max_sharpe_ret, max_sharpe_vol = self.portfolio_performance(max_sharpe.x)
        
        min_vol = self.min_variance()
        min_vol_ret, min_vol_vol = self.portfolio_performance(min_vol.x)
        
        plt.figure(figsize=(12, 8))
        
        if show_random:
            num_random = 5000
            random_results = np.zeros((2, num_random))
            for i in range(num_random):
                weights = np.random.random(len(self.tickers))
                weights /= np.sum(weights)
                random_results[0, i], random_results[1, i] = self.portfolio_performance(weights)
            
            plt.scatter(random_results[1, :], random_results[0, :], 
                       c=(random_results[0, :] - 0.02) / random_results[1, :],
                       cmap='viridis', marker='o', s=10, alpha=0.3, label='Random Portfolios')
            plt.colorbar(label='Sharpe Ratio')
        
        plt.plot(frontier[1, :], frontier[0, :], 'r--', linewidth=3, label='Efficient Frontier')
        plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='g', 
                   s=500, label=f'Max Sharpe Ratio', edgecolors='black', linewidth=2)
        plt.scatter(min_vol_vol, min_vol_ret, marker='*', color='b', 
                   s=500, label=f'Min Volatility', edgecolors='black', linewidth=2)
        
        plt.title('Efficient Frontier', fontsize=16, fontweight='bold')
        plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def display_optimal_portfolios(self):
        """Display optimal portfolio allocations"""
        max_sharpe = self.max_sharpe_ratio()
        max_sharpe_ret, max_sharpe_vol = self.portfolio_performance(max_sharpe.x)
        max_sharpe_ratio_value = (max_sharpe_ret - 0.02) / max_sharpe_vol
        
        min_vol = self.min_variance()
        min_vol_ret, min_vol_vol = self.portfolio_performance(min_vol.x)
        
        print("\n" + "="*60)
        print("OPTIMAL PORTFOLIOS")
        print("="*60)
        
        print("\n📈 MAXIMUM SHARPE RATIO PORTFOLIO")
        print("-" * 60)
        print(f"Expected Annual Return: {max_sharpe_ret:.2%}")
        print(f"Annual Volatility: {max_sharpe_vol:.2%}")
        print(f"Sharpe Ratio: {max_sharpe_ratio_value:.3f}")
        print("\nAllocations:")
        for ticker, weight in zip(self.tickers, max_sharpe.x):
            if weight > 0.001:
                print(f"  {ticker:6s}: {weight:6.2%}")
        
        print("\n📉 MINIMUM VARIANCE PORTFOLIO")
        print("-" * 60)
        print(f"Expected Annual Return: {min_vol_ret:.2%}")
        print(f"Annual Volatility: {min_vol_vol:.2%}")
        print(f"Sharpe Ratio: {(min_vol_ret - 0.02) / min_vol_vol:.3f}")
        print("\nAllocations:")
        for ticker, weight in zip(self.tickers, min_vol.x):
            if weight > 0.001:
                print(f"  {ticker:6s}: {weight:6.2%}")
        
        print("\n" + "="*60)


def print_banner():
    """Print welcome banner"""
    print("\n" + "="*60)
    print("  PORTFOLIO OPTIMIZATION TOOL")
    print("  Using Modern Portfolio Theory (MPT)")
    print("="*60)


def get_user_input():
    """Get portfolio details from user"""
    print("\n📝 Let's build your portfolio!")
    print("-" * 60)
    
    # Get tickers
    while True:
        tickers_input = input("\nEnter stock tickers separated by commas\n(e.g., AAPL,MSFT,GOOGL,TSLA):\n> ")
        tickers = [ticker.strip().upper() for ticker in tickers_input.split(',') if ticker.strip()]
        
        if len(tickers) < 2:
            print("⚠️  Please enter at least 2 tickers for portfolio optimization.")
            continue
        
        print(f"\n✅ You entered {len(tickers)} tickers: {', '.join(tickers)}")
        confirm = input("Is this correct? (y/n): ").lower()
        if confirm == 'y':
            break
    
    # Get date range options
    print("\n📅 Date Range Options")
    print("1. Same date range for all stocks (simple)")
    print("2. Custom date range for each stock (advanced)")
    
    date_option = input("Select option (1 or 2): ").strip()
    
    start_date = None
    end_date = None
    custom_dates = None
    
    if date_option == '2':
        print("\n📅 Custom Date Ranges")
        print("Enter date range for each stock (or press Enter to use default)")
        print("Format: YYYY-MM-DD YYYY-MM-DD (start and end)")
        print(f"Default range: Last 2 years")
        print("-" * 60)
        
        custom_dates = {}
        for ticker in tickers:
            date_input = input(f"{ticker}: ").strip()
            if date_input:
                try:
                    dates = date_input.split()
                    if len(dates) == 2:
                        custom_dates[ticker] = (dates[0], dates[1])
                        print(f"  ✅ {ticker}: {dates[0]} to {dates[1]}")
                    else:
                        print(f"  ⚠️ Invalid format for {ticker}, using default")
                except:
                    print(f"  ⚠️ Error parsing dates for {ticker}, using default")
            else:
                print(f"  ✅ {ticker}: Using default (last 2 years)")
        
        if not custom_dates:
            custom_dates = None
            print("\n✅ No custom dates entered, using default for all stocks")
    
    elif date_option == '1':
        print("\n📅 Global Date Range")
        use_custom = input("Use custom date range? (y/n, default is last 2 years): ").lower()
        
        if use_custom == 'y':
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            print(f"✅ Date range: {start_date} to {end_date}")
        else:
            print("✅ Using default: Last 2 years of data")
    else:
        print("⚠️ Invalid option, using default date range for all stocks")
    
    return tickers, start_date, end_date, custom_dates


def enter_purchases(tickers):
    """Enter purchase history for each stock"""
    print("\n💰 Enter Your Purchase History")
    print("-" * 60)
    print("Track your actual stock purchases with multiple buy dates")
    print("This will show your real portfolio performance!")
    print("-" * 60)
    
    purchases = {}
    
    for ticker in tickers:
        print(f"\n📊 {ticker}")
        track = input(f"  Do you own {ticker}? (y/n): ").lower()
        
        if track == 'y':
            purchases[ticker] = []
            
            while True:
                print(f"\n  Purchase #{len(purchases[ticker]) + 1} for {ticker}")
                date = input("    Purchase date (YYYY-MM-DD) or 'done': ").strip()
                
                if date.lower() == 'done':
                    break
                
                try:
                    shares = float(input("    Number of shares: ").strip())
                    price = float(input("    Price per share: $").strip())
                    
                    purchases[ticker].append({
                        'date': date,
                        'shares': shares,
                        'price': price
                    })
                    
                    total = shares * price
                    print(f"    ✅ Added: {shares} shares @ ${price:.2f} = ${total:.2f}")
                    
                    another = input("    Add another purchase for this stock? (y/n): ").lower()
                    if another != 'y':
                        break
                        
                except ValueError:
                    print("    ⚠️ Invalid input. Please enter valid numbers.")
    
    if purchases:
        print("\n✅ Purchase history recorded!")
        total_invested = sum(
            sum(p['shares'] * p['price'] for p in purchase_list)
            for purchase_list in purchases.values()
        )
        print(f"💵 Total invested: ${total_invested:,.2f}")
    else:
        print("\n⚠️ No purchases recorded.")
        purchases = None
    
    return purchases


def main_menu():
    """Main interactive menu"""
    print_banner()
    
    optimizer = None
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Enter/Change Portfolio Tickers")
        print("2. Enter/Update Purchase History")
        print("3. View Your Current Portfolio Performance")
        print("4. Show Optimal Portfolio Recommendations")
        print("5. Plot Efficient Frontier")
        print("6. Run Complete Analysis")
        print("7. Exit")
        print("-" * 60)
        
        choice = input("Select an option (1-7): ").strip()
        
        if choice == '1':
            tickers, start_date, end_date, custom_dates = get_user_input()
            
            try:
                purchases = optimizer.purchases if optimizer else None
                
                if custom_dates:
                    optimizer = PortfolioOptimizer(tickers, custom_dates=custom_dates, purchases=purchases)
                elif start_date and end_date:
                    optimizer = PortfolioOptimizer(tickers, start_date, end_date, purchases=purchases)
                else:
                    optimizer = PortfolioOptimizer(tickers, purchases=purchases)
                
                optimizer.fetch_data()
                print("\n✅ Portfolio loaded successfully!")
            except Exception as e:
                print(f"\n❌ Error loading portfolio: {e}")
                optimizer = None
        
        elif choice == '2':
            if optimizer is None:
                print("\n⚠️  Please enter portfolio tickers first (Option 1)")
            else:
                purchases = enter_purchases(optimizer.tickers)
                optimizer.purchases = purchases
        
        elif choice == '3':
            if optimizer is None:
                print("\n⚠️  Please enter a portfolio first (Option 1)")
            else:
                try:
                    optimizer.display_current_portfolio()
                except Exception as e:
                    print(f"\n❌ Error displaying portfolio: {e}")
        
        elif choice == '4':
            if optimizer is None:
                print("\n⚠️  Please enter a portfolio first (Option 1)")
            else:
                try:
                    optimizer.display_optimal_portfolios()
                except Exception as e:
                    print(f"\n❌ Error calculating portfolios: {e}")
        
        elif choice == '5':
            if optimizer is None:
                print("\n⚠️  Please enter a portfolio first (Option 1)")
            else:
                try:
                    optimizer.plot_efficient_frontier()
                except Exception as e:
                    print(f"\n❌ Error plotting frontier: {e}")
        
        elif choice == '6':
            if optimizer is None:
                print("\n⚠️  Please enter a portfolio first (Option 1)")
            else:
                try:
                    if optimizer.purchases:
                        optimizer.display_current_portfolio()
                    optimizer.display_optimal_portfolios()
                    optimizer.plot_efficient_frontier()
                except Exception as e:
                    print(f"\n❌ Error running analysis: {e}")
        
        elif choice == '7':
            print("\n👋 Thank you for using Portfolio Optimization Tool!")
            print("="*60)
            break
        
        else:
            print("\n⚠️  Invalid option. Please select 1-7.")


if __name__ == "__main__":
    main_menu()