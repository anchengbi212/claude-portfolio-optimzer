import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta

class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None):
        """
        Initialize portfolio optimizer with stock tickers
        
        Parameters:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date for historical data (YYYY-MM-DD)
        end_date (str): End date for historical data (YYYY-MM-DD)
        """
        self.tickers = tickers
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        
    def fetch_data(self):
        """Download historical stock data and calculate returns"""
        print(f"Fetching data for {', '.join(self.tickers)}...")
        
        # Download data with progress disabled for cleaner output
        raw_data = yf.download(self.tickers, start=self.start_date, end=self.end_date, progress=False)
        
        if raw_data.empty:
            raise ValueError("No data downloaded. Check your ticker symbols and internet connection.")
        
        # Handle different data structures based on number of tickers
        if len(self.tickers) == 1:
            # Single ticker - simpler structure
            if 'Adj Close' in raw_data.columns:
                data = raw_data[['Adj Close']].copy()
                data.columns = self.tickers
            else:
                data = raw_data.copy()
        else:
            # Multiple tickers - need to handle multi-level columns
            try:
                # Try to access Adj Close for multiple tickers
                if isinstance(raw_data.columns, pd.MultiIndex):
                    # Multi-level columns: ('Adj Close', 'AAPL'), ('Adj Close', 'MSFT'), etc.
                    data = raw_data.xs('Adj Close', axis=1, level=0)
                else:
                    # Single level columns
                    data = raw_data[['Adj Close']].copy() if 'Adj Close' in raw_data.columns else raw_data.copy()
            except Exception as e:
                print(f"Warning: Could not extract Adj Close, using available data: {e}")
                data = raw_data.copy()
        
        # Remove any columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Ensure we have valid column names
        if hasattr(data.columns, 'tolist'):
            self.tickers = [str(col) for col in data.columns.tolist()]
        else:
            self.tickers = [str(col) for col in data.columns]
        
        if len(self.tickers) == 0:
            raise ValueError("No valid data after cleaning. Check your ticker symbols and date range.")
        
        # Calculate daily returns
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        print(f"Data fetched successfully! ({len(self.returns)} trading days)")
        print(f"Tickers included: {', '.join(self.tickers)}")
        return self.returns
    
    def portfolio_performance(self, weights):
        """
        Calculate portfolio return and volatility
        
        Parameters:
        weights (array): Portfolio weights
        
        Returns:
        tuple: (annual_return, annual_volatility)
        """
        # Annualized return (252 trading days)
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        
        # Annualized volatility
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
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (long-only)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        init_guess = num_assets * [1. / num_assets]
        
        # Optimize
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
        """
        Generate efficient frontier by varying target returns
        
        Parameters:
        num_portfolios (int): Number of portfolios to generate
        
        Returns:
        tuple: (returns, volatilities, weights_array)
        """
        num_assets = len(self.tickers)
        results = np.zeros((3, num_portfolios))
        weights_array = []
        
        # Get min and max possible returns
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
            results[2, i] = (results[0, i] - 0.02) / results[1, i]  # Sharpe ratio
            weights_array.append(result.x)
        
        return results, weights_array
    
    def plot_efficient_frontier(self, num_portfolios=50, show_random=True):
        """Plot efficient frontier with optimal portfolios"""
        # Generate efficient frontier
        frontier, weights = self.efficient_frontier(num_portfolios)
        
        # Calculate optimal portfolios
        max_sharpe = self.max_sharpe_ratio()
        max_sharpe_ret, max_sharpe_vol = self.portfolio_performance(max_sharpe.x)
        
        min_vol = self.min_variance()
        min_vol_ret, min_vol_vol = self.portfolio_performance(min_vol.x)
        
        # Plot
        plt.figure(figsize=(12, 8))
        
        # Random portfolios for comparison
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
        
        # Efficient frontier
        plt.plot(frontier[1, :], frontier[0, :], 'r--', linewidth=3, label='Efficient Frontier')
        
        # Maximum Sharpe ratio portfolio
        plt.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='g', 
                   s=500, label=f'Max Sharpe Ratio', edgecolors='black', linewidth=2)
        
        # Minimum variance portfolio
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
        # Maximum Sharpe Ratio Portfolio
        max_sharpe = self.max_sharpe_ratio()
        max_sharpe_ret, max_sharpe_vol = self.portfolio_performance(max_sharpe.x)
        max_sharpe_ratio_value = (max_sharpe_ret - 0.02) / max_sharpe_vol
        
        # Minimum Variance Portfolio
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
            if weight > 0.001:  # Only show non-negligible weights
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


# Example usage
if __name__ == "__main__":
    # Define portfolio tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'V', 'PG']
    
    # Create optimizer instance
    optimizer = PortfolioOptimizer(tickers)
    
    # Fetch historical data
    optimizer.fetch_data()
    
    # Display optimal portfolios
    optimizer.display_optimal_portfolios()
    
    # Plot efficient frontier
    optimizer.plot_efficient_frontier(num_portfolios=50, show_random=True)