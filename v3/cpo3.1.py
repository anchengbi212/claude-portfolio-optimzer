import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import minimize
import yfinance as yf
from datetime import datetime, timedelta
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading

class PortfolioOptimizer:
    def __init__(self, tickers, start_date=None, end_date=None, custom_dates=None, purchases=None):
        self.tickers = tickers
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.start_date = start_date or (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        # Validate dates
        today = datetime.now().date()
        if end_date:
            end_dt = datetime.strptime(end_date, '%Y-%m-%d').date()
            if end_dt > today:
                self.end_date = today.strftime('%Y-%m-%d')
        
        if start_date:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d').date()
            if start_dt > today:
                self.start_date = (today - timedelta(days=365*2)).strftime('%Y-%m-%d')
        
        self.custom_dates = custom_dates or {}
        self.purchases = purchases or {}
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None
        self.price_data = None
        self.market_data = None
        self.market_returns = None
        
    def fetch_data(self):
        all_data = {}
        failed_tickers = []
        
        for ticker in self.tickers:
            try:
                ticker_data = yf.download(ticker, start=self.start_date, end=self.end_date, 
                                        progress=False, auto_adjust=False)
                
                if not ticker_data.empty:
                    if 'Adj Close' in ticker_data.columns:
                        price_series = ticker_data['Adj Close']
                    elif 'Close' in ticker_data.columns:
                        price_series = ticker_data['Close']
                    else:
                        price_series = ticker_data.iloc[:, 0]
                    
                    if isinstance(price_series, pd.DataFrame):
                        price_series = price_series.squeeze()
                    
                    if not isinstance(price_series.index, pd.DatetimeIndex):
                        price_series.index = pd.to_datetime(price_series.index)
                    
                    all_data[ticker] = price_series
                else:
                    failed_tickers.append(ticker)
            except Exception as e:
                failed_tickers.append(ticker)
        
        if not all_data:
            raise ValueError("No data downloaded for any ticker.")
        
        data = pd.DataFrame(all_data)
        data = data.dropna()
        
        self.price_data = data.copy()
        self.tickers = list(data.columns)
        
        if len(self.tickers) < 2:
            raise ValueError(f"Need at least 2 valid tickers.")
        
        self.returns = data.pct_change().dropna()
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()
        
        self._fetch_market_data()
        
        return self.returns, failed_tickers
    
    def _fetch_market_data(self):
        try:
            market_data = yf.download('^GSPC', start=self.start_date, end=self.end_date, 
                                     progress=False, auto_adjust=False)
            
            if not market_data.empty:
                if 'Adj Close' in market_data.columns:
                    self.market_data = market_data['Adj Close']
                elif 'Close' in market_data.columns:
                    self.market_data = market_data['Close']
                else:
                    self.market_data = market_data.iloc[:, 0]
                
                if isinstance(self.market_data, pd.DataFrame):
                    self.market_data = self.market_data.squeeze()
                
                self.market_returns = self.market_data.pct_change().dropna()
        except Exception as e:
            self.market_data = None
            self.market_returns = None
    
    def portfolio_performance(self, weights):
        portfolio_return = np.sum(self.mean_returns * weights) * 252
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix * 252, weights)))
        return portfolio_return, portfolio_std
    
    def negative_sharpe(self, weights, risk_free_rate=0.02):
        ret, std = self.portfolio_performance(weights)
        sharpe = (ret - risk_free_rate) / std
        return -sharpe
    
    def max_sharpe_ratio(self, risk_free_rate=0.02):
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        init_guess = num_assets * [1. / num_assets]
        
        result = minimize(self.negative_sharpe, init_guess, 
                         args=(risk_free_rate,),
                         method='SLSQP', bounds=bounds, constraints=constraints)
        return result
    
    def min_variance(self):
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
    
    def calculate_capm_metrics(self, risk_free_rate=0.02):
        if self.market_returns is None:
            return None
        
        capm_results = {}
        
        for ticker in self.tickers:
            common_dates = self.returns.index.intersection(self.market_returns.index)
            stock_returns = self.returns[ticker].loc[common_dates]
            market_returns_aligned = self.market_returns.loc[common_dates]
            
            if len(common_dates) < 30:
                continue
            
            covariance = stock_returns.cov(market_returns_aligned)
            market_variance = market_returns_aligned.var()
            beta = covariance / market_variance
            
            stock_return = stock_returns.mean() * 252
            market_return = market_returns_aligned.mean() * 252
            expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
            alpha = stock_return - expected_return
            
            correlation = stock_returns.corr(market_returns_aligned)
            r_squared = correlation ** 2
            
            total_variance = stock_returns.var() * 252
            systematic_variance = (beta ** 2) * (market_returns_aligned.var() * 252)
            unsystematic_variance = total_variance - systematic_variance
            
            capm_results[ticker] = {
                'beta': beta,
                'alpha': alpha,
                'expected_return': expected_return,
                'actual_return': stock_return,
                'r_squared': r_squared,
                'systematic_risk': np.sqrt(systematic_variance),
                'unsystematic_risk': np.sqrt(unsystematic_variance),
                'total_risk': np.sqrt(total_variance)
            }
        
        return capm_results
    
    def calculate_portfolio_beta(self, weights):
        if self.market_returns is None:
            return None
        
        capm_results = self.calculate_capm_metrics()
        if capm_results is None:
            return None
        
        portfolio_beta = 0
        for ticker, weight in zip(self.tickers, weights):
            if ticker in capm_results:
                portfolio_beta += weight * capm_results[ticker]['beta']
        
        return portfolio_beta
    
    def calculate_current_portfolio_value(self):
        if not self.purchases or self.price_data is None:
            return None
        
        results = {}
        total_invested = 0
        total_current_value = 0
        
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


class PortfolioGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Portfolio Optimization Tool")
        self.root.geometry("1200x800")
        
        self.optimizer = None
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.create_input_tab()
        self.create_results_tab()
        self.create_capm_tab()
        self.create_performance_tab()
        self.create_chart_tab()
        
    def create_input_tab(self):
        input_frame = ttk.Frame(self.notebook)
        self.notebook.add(input_frame, text="📝 Input Portfolio")
        
        # Tickers input
        ttk.Label(input_frame, text="Stock Tickers (comma-separated):", 
                 font=('Arial', 12, 'bold')).pack(pady=10)
        self.tickers_entry = ttk.Entry(input_frame, width=50, font=('Arial', 11))
        self.tickers_entry.pack(pady=5)
        self.tickers_entry.insert(0, "AAPL,MSFT,GOOGL,TSLA")
        
        # Date range
        date_frame = ttk.Frame(input_frame)
        date_frame.pack(pady=10)
        
        ttk.Label(date_frame, text="Start Date:", font=('Arial', 10)).grid(row=0, column=0, padx=5)
        self.start_date_entry = ttk.Entry(date_frame, width=15)
        self.start_date_entry.grid(row=0, column=1, padx=5)
        default_start = (datetime.now() - timedelta(days=365*2)).strftime('%Y-%m-%d')
        self.start_date_entry.insert(0, default_start)
        
        ttk.Label(date_frame, text="End Date:", font=('Arial', 10)).grid(row=0, column=2, padx=5)
        self.end_date_entry = ttk.Entry(date_frame, width=15)
        self.end_date_entry.grid(row=0, column=3, padx=5)
        self.end_date_entry.insert(0, datetime.now().strftime('%Y-%m-%d'))
        
        # Buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Load Portfolio Data", 
                  command=self.load_portfolio).pack(side='left', padx=5)
        ttk.Button(button_frame, text="Run Analysis", 
                  command=self.run_analysis).pack(side='left', padx=5)
        
        # Status
        self.status_label = ttk.Label(input_frame, text="Ready", 
                                     font=('Arial', 10), foreground='blue')
        self.status_label.pack(pady=10)
        
        # Progress
        self.progress = ttk.Progressbar(input_frame, mode='indeterminate', length=400)
        self.progress.pack(pady=10)
        
    def create_results_tab(self):
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="📊 Optimal Portfolios")
        
        # Create scrolled text for results
        self.results_text = scrolledtext.ScrolledText(results_frame, width=140, height=40,
                                                      font=('Courier', 10))
        self.results_text.pack(fill='both', expand=True, padx=10, pady=10)
        
    def create_capm_tab(self):
        capm_frame = ttk.Frame(self.notebook)
        self.notebook.add(capm_frame, text="📈 CAPM Analysis")
        
        # Create treeview for CAPM data
        columns = ('Ticker', 'Beta', 'Alpha', 'Expected Return', 'Actual Return', 
                  'R²', 'Total Risk', 'Systematic Risk', 'Unsystematic Risk')
        
        self.capm_tree = ttk.Treeview(capm_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.capm_tree.heading(col, text=col)
            self.capm_tree.column(col, width=120, anchor='center')
        
        scrollbar = ttk.Scrollbar(capm_frame, orient='vertical', command=self.capm_tree.yview)
        self.capm_tree.configure(yscrollcommand=scrollbar.set)
        
        self.capm_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
        
    def create_performance_tab(self):
        perf_frame = ttk.Frame(self.notebook)
        self.notebook.add(perf_frame, text="💰 Current Holdings")
        
        columns = ('Ticker', 'Shares', 'Invested', 'Current Value', 
                  'Current Price', 'Gain/Loss $', 'Gain/Loss %')
        
        self.perf_tree = ttk.Treeview(perf_frame, columns=columns, show='headings', height=20)
        
        for col in columns:
            self.perf_tree.heading(col, text=col)
            self.perf_tree.column(col, width=140, anchor='center')
        
        scrollbar = ttk.Scrollbar(perf_frame, orient='vertical', command=self.perf_tree.yview)
        self.perf_tree.configure(yscrollcommand=scrollbar.set)
        
        self.perf_tree.pack(side='left', fill='both', expand=True, padx=10, pady=10)
        scrollbar.pack(side='right', fill='y')
        
        # Summary label
        self.perf_summary = ttk.Label(perf_frame, text="", font=('Arial', 12, 'bold'))
        self.perf_summary.pack(pady=10)
        
    def create_chart_tab(self):
        chart_frame = ttk.Frame(self.notebook)
        self.notebook.add(chart_frame, text="📉 Efficient Frontier")
        
        self.chart_canvas = None
        self.chart_frame = chart_frame
        
    def load_portfolio(self):
        tickers_str = self.tickers_entry.get().strip()
        if not tickers_str:
            messagebox.showerror("Error", "Please enter at least 2 tickers")
            return
        
        tickers = [t.strip().upper() for t in tickers_str.split(',')]
        
        if len(tickers) < 2:
            messagebox.showerror("Error", "Please enter at least 2 tickers")
            return
        
        start_date = self.start_date_entry.get().strip()
        end_date = self.end_date_entry.get().strip()
        
        self.status_label.config(text="Loading data...", foreground='orange')
        self.progress.start()
        
        def load_thread():
            try:
                self.optimizer = PortfolioOptimizer(tickers, start_date, end_date)
                returns, failed = self.optimizer.fetch_data()
                
                self.root.after(0, lambda: self.progress.stop())
                
                msg = f"✅ Loaded {len(self.optimizer.tickers)} tickers successfully!"
                if failed:
                    msg += f"\n⚠️ Failed: {', '.join(failed)}"
                
                self.root.after(0, lambda: self.status_label.config(text=msg, foreground='green'))
                self.root.after(0, lambda: messagebox.showinfo("Success", msg))
                
            except Exception as e:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error: {str(e)}", foreground='red'))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=load_thread, daemon=True).start()
    
    def run_analysis(self):
        if self.optimizer is None:
            messagebox.showerror("Error", "Please load portfolio data first")
            return
        
        self.status_label.config(text="Running analysis...", foreground='orange')
        self.progress.start()
        
        def analyze_thread():
            try:
                # Calculate optimal portfolios
                max_sharpe = self.optimizer.max_sharpe_ratio()
                max_sharpe_ret, max_sharpe_vol = self.optimizer.portfolio_performance(max_sharpe.x)
                max_sharpe_ratio_value = (max_sharpe_ret - 0.02) / max_sharpe_vol
                
                min_vol = self.optimizer.min_variance()
                min_vol_ret, min_vol_vol = self.optimizer.portfolio_performance(min_vol.x)
                
                portfolio_beta_max = self.optimizer.calculate_portfolio_beta(max_sharpe.x)
                portfolio_beta_min = self.optimizer.calculate_portfolio_beta(min_vol.x)
                
                # Display results
                results_text = "="*80 + "\n"
                results_text += "OPTIMAL PORTFOLIOS\n"
                results_text += "="*80 + "\n\n"
                
                results_text += "📈 MAXIMUM SHARPE RATIO PORTFOLIO\n"
                results_text += "-"*80 + "\n"
                results_text += f"Expected Annual Return: {max_sharpe_ret:.2%}\n"
                results_text += f"Annual Volatility: {max_sharpe_vol:.2%}\n"
                results_text += f"Sharpe Ratio: {max_sharpe_ratio_value:.3f}\n"
                if portfolio_beta_max:
                    results_text += f"Portfolio Beta: {portfolio_beta_max:.3f}\n"
                results_text += "\nAllocations:\n"
                for ticker, weight in zip(self.optimizer.tickers, max_sharpe.x):
                    if weight > 0.001:
                        results_text += f"  {ticker:6s}: {weight:6.2%}\n"
                
                results_text += "\n📉 MINIMUM VARIANCE PORTFOLIO\n"
                results_text += "-"*80 + "\n"
                results_text += f"Expected Annual Return: {min_vol_ret:.2%}\n"
                results_text += f"Annual Volatility: {min_vol_vol:.2%}\n"
                results_text += f"Sharpe Ratio: {(min_vol_ret - 0.02) / min_vol_vol:.3f}\n"
                if portfolio_beta_min:
                    results_text += f"Portfolio Beta: {portfolio_beta_min:.3f}\n"
                results_text += "\nAllocations:\n"
                for ticker, weight in zip(self.optimizer.tickers, min_vol.x):
                    if weight > 0.001:
                        results_text += f"  {ticker:6s}: {weight:6.2%}\n"
                
                results_text += "\n" + "="*80
                
                self.root.after(0, lambda: self.results_text.delete(1.0, tk.END))
                self.root.after(0, lambda: self.results_text.insert(1.0, results_text))
                
                # CAPM Analysis
                capm_results = self.optimizer.calculate_capm_metrics()
                if capm_results:
                    self.root.after(0, lambda: self.update_capm_table(capm_results))
                
                # Plot efficient frontier
                self.root.after(0, self.plot_efficient_frontier)
                
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.config(
                    text="✅ Analysis complete!", foreground='green'))
                
            except Exception as e:
                self.root.after(0, lambda: self.progress.stop())
                self.root.after(0, lambda: self.status_label.config(
                    text=f"Error: {str(e)}", foreground='red'))
                self.root.after(0, lambda: messagebox.showerror("Error", str(e)))
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def update_capm_table(self, capm_results):
        # Clear existing data
        for item in self.capm_tree.get_children():
            self.capm_tree.delete(item)
        
        # Add data
        for ticker, metrics in capm_results.items():
            values = (
                ticker,
                f"{metrics['beta']:.3f}",
                f"{metrics['alpha']:.2%}",
                f"{metrics['expected_return']:.2%}",
                f"{metrics['actual_return']:.2%}",
                f"{metrics['r_squared']:.3f}",
                f"{metrics['total_risk']:.2%}",
                f"{metrics['systematic_risk']:.2%}",
                f"{metrics['unsystematic_risk']:.2%}"
            )
            self.capm_tree.insert('', 'end', values=values)
    
    def plot_efficient_frontier(self):
        # Clear previous chart
        for widget in self.chart_frame.winfo_children():
            widget.destroy()
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Generate efficient frontier
        frontier, weights = self.optimizer.efficient_frontier(50)
        
        # Calculate optimal portfolios
        max_sharpe = self.optimizer.max_sharpe_ratio()
        max_sharpe_ret, max_sharpe_vol = self.optimizer.portfolio_performance(max_sharpe.x)
        
        min_vol = self.optimizer.min_variance()
        min_vol_ret, min_vol_vol = self.optimizer.portfolio_performance(min_vol.x)
        
        # Random portfolios
        num_random = 5000
        random_results = np.zeros((2, num_random))
        for i in range(num_random):
            weights_rand = np.random.random(len(self.optimizer.tickers))
            weights_rand /= np.sum(weights_rand)
            random_results[0, i], random_results[1, i] = self.optimizer.portfolio_performance(weights_rand)
        
        scatter = ax.scatter(random_results[1, :], random_results[0, :], 
                   c=(random_results[0, :] - 0.02) / random_results[1, :],
                   cmap='viridis', marker='o', s=10, alpha=0.3, label='Random Portfolios')
        plt.colorbar(scatter, ax=ax, label='Sharpe Ratio')
        
        # Efficient frontier
        ax.plot(frontier[1, :], frontier[0, :], 'r--', linewidth=3, label='Efficient Frontier')
        
        # Optimal portfolios
        ax.scatter(max_sharpe_vol, max_sharpe_ret, marker='*', color='g', 
                   s=500, label='Max Sharpe Ratio', edgecolors='black', linewidth=2, zorder=5)
        ax.scatter(min_vol_vol, min_vol_ret, marker='*', color='b', 
                   s=500, label='Min Volatility', edgecolors='black', linewidth=2, zorder=5)
        
        ax.set_title('Efficient Frontier', fontsize=16, fontweight='bold')
        ax.set_xlabel('Volatility (Standard Deviation)', fontsize=12)
        ax.set_ylabel('Expected Return', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioGUI(root)
    root.mainloop()