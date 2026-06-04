# stocksimpy

## Purpose

stocksimpy is created to be a simple and intuitive backtesting library for stock trading strategies. It is designed to be beginner-friendly, allowing users to quickly test and visualize their trading ideas without needing to write complex code. The library provides easy-to-use functions for loading stock data, running backtests, evaluating performance, and visualizing results.

(Disclaimer: This is an early alpha, API might change in the future updates)

---

## 🎯 Quick Features

- Load stock prices from `yfinance` or your own database.
- Run fixed or dynamic backtests in just a few lines of code.
- Built-in example strategies like SMA/EMA crossover, RSI, and price action.
- Clear and simple design — no hidden magic, everything is easy to read.
- Beginner-friendly: start experimenting immediately.

---

## ⚡ [Quick Start](https://stocksimpy.readthedocs.io/en/latest/quick_start.html)

### 1. Install `stocksimpy` (and `yfinance` for convenient data input)

```bash
pip install stocksimpy yfinance
```

### 2. Imports

```python
from stocksimpy import StockData, Backtester, Visualize, Performance
```

### 3. Load Data

```python
# If you want to load your own data from .csv, .sqlite or similar, use appropriate functions built into `StockData()`
data = StockData.from_yfinance([your_stock_symbol], your_starting_date, your_end_date)
```

### 4. Define Your Strategy

```python
def sma_crossover(prices):

    # Don't do any trades until long range
    if(len(prices) < 50):
        return "hold"
    
    short = prices["Close"].rolling(20).mean()
    long = prices["Close"].rolling(50).mean()

    # Buy when short crosses above long, sell when it crosses below
    if short > long:
        return "buy"
    else:
        return "sell"
```

### 5. Run Backtest

```python
# You can use your own strategy or one of the built-in ones for testing (e.g. Strategy.rsi_momentum_fixed())
bt = Backtester('AAPL', data, sma_crossover)

# Depending on your strategy you may need to run bt.run_backtest_dynamic(), see more on the documentation
bt.run_backtest_fixed()
```

### 6. Evaluate the Results

```python
# View performance metrics like max drawdown and sharpe ratio
perf = Performance(bt)
print(perf.generate_risk_report())
```

### 7. Visualize

```python
# Visualize the graph
graph = Visualize(bt)
graph.visualize_backtest().show() # You can use .savefig(...) instead of .show() to record the graph in disk
```

And thats it, you just ran your first backtest 🚀

---

## 🤝 Contributions

Ideas, strategies, or improvements? We welcome all contributions! Check out [CONTRIBUTING.md](CONTRIBUTING.md) for a more detailed information about contributing

## 📃 Docs

More in depth explanation on how to use this library can be found in [ReadTheDocs](https://stocksimpy.readthedocs.io/en/latest/).
