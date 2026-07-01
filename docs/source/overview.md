# Getting Started with stocksimpy

Welcome to **stocksimpy**! This guide will walk you through the core concepts and help you run your first backtest in minutes.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Loading Data](#loading-data)
5. [Adding Indicators](#adding-indicators)
6. [Running a Backtest](#running-a-backtest)
7. [Analyzing Results](#analyzing-results)
8. [Using Built-in Strategies](#using-built-in-strategies)
9. [Creating Custom Strategies](#creating-custom-strategies)
10. [Next Steps](#next-steps)

---

## Installation

### Prerequisites

- Python 3.7+
- pip

### Install stocksimpy

```bash
pip install stocksimpy
```

### Verify installation

```python
from stocksimpy import StockData, Backtester, Strategy, Performance, Visualize
print("stocksimpy is ready!")
```

---

## Quick Start

Here's a 30-second backtest:

```python
from stocksimpy import *

# 1. Load data
data = StockData.from_yfinance(["AAPL"], days_before=365)

# 2. Create backtester with a built-in strategy
bt = Backtester("AAPL", data, Strategy.rsi_momentum_fixed())

# 3. Run backtest
bt.run_backtest_fixed()

# 4. Analyze results
perf = Performance(bt)
print(perf.generate_risk_report())

# 5. Visualize
viz = Visualize(bt)
viz.visualize_backtest().show()
```

---

## Core Concepts

### StockData

**What it does:** Loads, validates, and manages OHLCV (Open, High, Low, Close, Volume) and indicators stock price data.

**Key features:**

- Supports multiple data sources: yfinance, CSV, Excel, SQL, JSON, or pandas DataFrames
- Validates data integrity (no missing values, duplicate dates, or negative volumes)
- Handles multi-ticker data automatically
- Can implement indicators with built-in methods

**Example:**

```python
from stocksimpy import StockData

# Directly from yfinance
data = StockData.from_yfinance(["AAPL", "GOOGL"], days_before=365)

# From CSV file
data = StockData.from_csv("stock_prices.csv")

# From pandas DataFrame
import pandas as pd
df = pd.read_csv("data.csv", index_col="Date", parse_dates=True)
data = StockData(df)
```

### Indicators

**What it does:** Contains calculations for built-in indicators, such as SMA, RSI or MACD line, that can be directly implemented into the ``StockData`` throught ``.add_indicator``

**Key features:**

- Contains a variety of indicator calculations
    - SMA, WMA, EMA, wilders_smoothing, DEMA, TEMA, HMA, RSI, macd line, wilders_macd, tema_macd, hma_macd
- Each method returns a dictionary of the calculated values

**Examples:**

```Python
from stocksimpy import *

# Load Data
data = StockData.from_yfinance(["AAPL", "GOOGL"], days_before=365)

# Calculates SMA with a window of 20, and adds it as "sma_20" into the data
data.add_indicator(
    calculate_sma, "close" 20
)

# The functions can also be used stand alone
sma = calculate_sma(data["close", "AAPL"], 20)
```

### Backtester

**What it does:** Executes a trading strategy on historical data and tracks the portfolio state (cash, holdings, trades, value over time).

**Key features:**

- Two execution modes: **fixed-size** (same shares per trade) or **dynamic-size** (variable shares)
- Simulates realistic trading: applies transaction fees, tracks all trades
- Multi-ticker support: automatically filters to a single symbol
- Portfolio tracking: records every trade and portfolio value at each timestep

**Example:**

```python
from stocksimpy import Backtester, StockData

data = StockData.from_yfinance(["AAPL"], days_before=365)

bt = Backtester(
    symbol="AAPL",
    data=data,
    strategy=my_strategy,
    initial_cap=100000,        # Starting cash
    transaction_fee=10,        # Fee per trade
    trade_amount=5000          # Used in fixed-size mode
)

# Run fixed-size backtest
bt.run_backtest_fixed()

# Or run dynamic-size backtest
bt.run_backtest_dynamic()
```

### Strategy

**What it does:** Encapsulates trading logic. Strategies receive historical data and return a signal ('buy', 'sell', or 'hold'). Remember you can use ``Strategy`` class for built-in strategies.

**Two types:**

1. **Fixed-size strategy**
   - Signature: `strategy(data: DataFrame) -> str`
   - Always trades the same number of shares (determined by `trade_amount`)
   - Simple and easy to reason about

2. **Dynamic-size strategy**
   - Signature: `strategy(data: DataFrame, holdings: float) -> (str, int)`
   - Can vary the number of shares per trade
   - More flexibility for advanced trading logic

**Example:**

```python
def my_fixed_strategy(data):
    """Buy when price is below its 20-day average."""
    close = data['Close']
    sma_20 = close.rolling(20).mean()
    if close.iloc[-1] < sma_20.iloc[-1]:
        return 'buy'
    return 'hold'

def my_dynamic_strategy(data, holdings):
    """Buy more when oversold (RSI < 30)."""
    # Compute RSI
    delta = data['Close'].diff()
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    avg_gain = gains.rolling(14).mean()
    avg_loss = losses.rolling(14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    if rsi.iloc[-1] < 30:
        shares = int(10000 / data['Close'].iloc[-1])  # Allocate $10k
        return 'buy', shares
    return 'hold', 0
```

### Portfolio

**What it does:** Tracks the state of cash and holdings during a backtest.

**Key attributes:**

- `cash`: Current available cash
- `holdings`: Current shares owned
- `value_history`: Time series of total portfolio value (cash + holdings value)
- `trade_log`: DataFrame with all executed trades

**Example:**

```python
bt = Backtester("AAPL", data, my_strategy)
bt.run_backtest_fixed()

# Access portfolio results
print(f"Final cash: ${bt.portfolio.cash:.2f}")
print(f"Final holdings: {bt.portfolio.holdings} shares")
print(f"Total value over time:\n{bt.portfolio.value_history}")
print(f"All executed trades:\n{bt.portfolio.trade_log}")
```

### Performance

**What it does:** Computes performance metrics from a completed backtest.

**Available metrics:**

- **Daily returns:** Percentage change in portfolio value each day
- **Total return:** Overall gain/loss percentage
- **Annualized return:** Scaled to annual percentage
- **Maximum drawdown:** Largest peak-to-trough decline
- **Sharpe ratio:** Risk-adjusted return (assumes 2% risk-free rate)
- **Volatility:** Standard deviation of daily returns

**Example:**

```python
from stocksimpy import Performance

bt = Backtester("AAPL", data, Strategy.rsi_momentum_fixed())
bt.run_backtest_fixed()

perf = Performance(bt)
print(perf.generate_risk_report())  # Pretty-printed metrics
print(f"Sharpe Ratio: {perf.calc_sharpe_ratio():.2f}")
print(f"Max Drawdown: {perf.calc_max_drawdown():.2%}")
```

### Visualize

**What it does:** Plots backtest results with price, portfolio value, and trade markers.

**Example:**

```python
from stocksimpy import Visualize

viz = Visualize(bt)
plt = viz.visualize_backtest()

# Display or save
plt.show()
# Or: plt.savefig("backtest_results.png")
```

---

## Loading Data

### From Yahoo Finance (yfinance)

Fastest way to get started:

```python
from stocksimpy import StockData

# Load last 365 days
data = StockData.from_yfinance(["AAPL"], days_before=365)

# Load multiple symbols
data = StockData.from_yfinance(["AAPL", "GOOGL", "MSFT"], days_before=730)

# Load between specific dates
import datetime
start = datetime.date(2022, 1, 1)
end = datetime.date(2023, 12, 31)
data = StockData.from_yfinance(["AAPL"], start_date=start, end_date=end)
```

### From CSV File

```python
from stocksimpy import StockData

data = StockData.from_csv(
    "historical_prices.csv",
    index_col="Date",  # Column name for dates
    parse_dates=True   # Parse as datetime
)
```

**Expected CSV format:**

```
Date,Open,High,Low,Close,Volume
2023-01-01,100.0,105.0,99.0,102.0,1000000
2023-01-02,102.0,106.0,101.0,104.0,1200000
...
```

### From Pandas DataFrame

```python
import pandas as pd
from stocksimpy import StockData

df = pd.read_csv("prices.csv", index_col="Date", parse_dates=True)
data = StockData(df)
```

### From SQL Database

```python
from stocksimpy import StockData
import sqlite3

conn = sqlite3.connect("stock_data.db")
df = pd.read_sql("SELECT * FROM prices", conn, index_col="Date", parse_dates=True)
data = StockData(df)
```

---

## Adding Indicators

You can add technical indicators to your data for use in strategies:

```python
from stocksimpy import StockData, Indicators

data = StockData.from_yfinance(["AAPL"], days_before=365)

# Add 20-day SMA
data.add_indicator(Indicators.sma, base_col = "Close", window=20)
# Add 14-day RSI
data.add_indicator(Indicators.rsi, base_col = "Close", window=14)
```

Indicators module offers many built-in indicators like SMA, EMA, RSI, MACD, etc. You can also add custom indicators by passing a function that takes the data (as a pandas Series) and returns a dictionary of Series, with the keys being the indicator names and the values being the Series.

Here is an example of adding a custom indicator:

```python
def my_custom_indicator(data: pd.Series) -> dict:
    # Example: 10-day volatility (standard deviation of returns)
    returns = data.pct_change()
    volatility = returns.rolling(10).std()
    return {'volatility_10d': volatility}

data.add_indicator(my_custom_indicator, base_col="Close")
``` 

---

## Running a Backtest

### Fixed-Size Backtest

Use when you want to trade the same dollar amount each time:

```python
from stocksimpy import Backtester, StockData, Strategy

data = StockData.from_yfinance(["AAPL"], days_before=365)

bt = Backtester(
    symbol="AAPL",
    data=data,
    strategy=Strategy.sma_ema_crossover_fixed(fast=12, slow=26),
    initial_cap=100000,      # Start with $100k
    trade_amount=5000,       # Trade $5k per signal
    transaction_fee=10       # $10 per trade
)

# Run the backtest
bt.run_backtest_fixed()

# Check results
report = bt.generate_report()
print(report)
```

### Dynamic-Size Backtest

Use when you want the strategy to decide position size:

```python
from stocksimpy import Backtester, StockData, Strategy

data = StockData.from_yfinance(["AAPL"], days_before=365)

bt = Backtester(
    symbol="AAPL",
    data=data,
    strategy=Strategy.price_action_dynamic(),
    initial_cap=100000,
    transaction_fee=10
)

# Run the backtest (strategy determines shares per trade)
bt.run_backtest_dynamic()

# Check results
report = bt.generate_report()
print(report)
```

---

## Analyzing Results

### View Performance Metrics

```python
from stocksimpy import Performance

perf = Performance(bt)

# Print all metrics
print(perf.generate_risk_report())

# Access individual metrics
print(f"Total Return: {perf.calc_total_return():.2%}")
print(f"Annualized Return: {perf.get_annualized_return():.2%}")
print(f"Max Drawdown: {perf.calc_max_drawdown():.2%}")
print(f"Sharpe Ratio: {perf.calc_sharpe_ratio():.2f}")
print(f"Volatility: {perf.calc_volatility():.2%}")
```

### Access Trade History

```python
# Get all trades
trades = bt.portfolio.trade_log
print(trades)

# Filter by type
buy_trades = trades[trades['Type'] == 'buy']
sell_trades = trades[trades['Type'] == 'sell']

# Summary statistics
print(f"Total trades: {len(trades)}")
print(f"Buy trades: {len(buy_trades)}")
print(f"Sell trades: {len(sell_trades)}")
print(f"Win rate: {(trades['Profit'] > 0).sum() / len(trades):.1%}")
```

### Access Portfolio Value History

```python
# Time series of total portfolio value
value_history = bt.portfolio.value_history

# Final value
final_value = value_history.iloc[-1]
profit = final_value - bt.initial_cap
print(f"Starting capital: ${bt.initial_cap:,.2f}")
print(f"Final value: ${final_value:,.2f}")
print(f"Profit/Loss: ${profit:,.2f}")
```

### Visualize Results

```python
from stocksimpy import Visualize

viz = Visualize(bt)
plt = viz.visualize_backtest()

# Show on screen
plt.show()

# Or save to file
plt.savefig("my_backtest.png", dpi=150)
```

---

## Using Built-in Strategies

stocksimpy includes several ready-to-use strategies for quick testing.

### Fixed-Size Strategies

#### 1. Buy All (`buy_all_fixed`)

Always buy. Useful for buy-and-hold baseline comparison:

```python
from stocksimpy import Strategy

strategy = Strategy.buy_all_fixed()
# Returns 'buy' at every step
```

#### 2. RSI Momentum (`rsi_momentum_fixed`)

Buy on upward RSI crossover (30 → 50), sell on downward crossover (50 → 30):

```python
strategy = Strategy.rsi_momentum_fixed(rsi_period=14)
```

**Parameters:**

- `rsi_period`: RSI lookback window (default: 14)

#### 3. SMA/EMA Crossover (`sma_ema_crossover_fixed`)

Buy when fast SMA crosses above slow SMA, sell on crossunder:

```python
strategy = Strategy.sma_ema_crossover_fixed(fast=12, slow=26)
```

**Parameters:**

- `fast`: Fast moving average window (default: 12)
- `slow`: Slow moving average window (default: 26)

#### 4. RSI Reversion (`rsi_reversion_fixed`)

Buy when RSI is oversold, sell when overbought:

```python
strategy = Strategy.rsi_reversion_fixed(
    rsi_period=14,
    low_th=30,      # Oversold threshold
    high_th=70      # Overbought threshold
)
```

#### 5. Multi-Indicator (`multi_indicator_fixed`)

Buy when RSI < 60 AND price > 200-day SMA, sell when RSI > 70 OR price < SMA:

```python
strategy = Strategy.multi_indicator_fixed(rsi_period=14, sma_long=200)
```

### Dynamic-Size Strategies

#### 1. Price Action (`price_action_dynamic`)

Buy when price drops ≥ 14% from 30 days ago, allocates $20k per trade:

```python
strategy = Strategy.price_action_dynamic()
```

---

## Creating Custom Strategies

### Template: Fixed-Size Strategy

A fixed-size strategy receives historical data and returns a signal string:

```python
def my_fixed_strategy(data):
    """
    Custom fixed-size strategy.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Historical OHLCV data indexed by date.
    
    Returns
    -------
    str
        One of 'buy', 'sell', or 'hold'.
    """
    # Extract the close price
    close = data['Close']
    
    # Your logic here
    if close.iloc[-1] > close.iloc[-20]:
        return 'buy'
    elif close.iloc[-1] < close.iloc[-5]:
        return 'sell'
    else:
        return 'hold'
```

### Template: Dynamic-Size Strategy

A dynamic-size strategy receives historical data and current holdings, returns signal and share count:

```python
def my_dynamic_strategy(data, holdings):
    """
    Custom dynamic-size strategy.
    
    Parameters
    ----------
    data : pandas.DataFrame
        Historical OHLCV data indexed by date.
    holdings : float
        Current shares owned.
    
    Returns
    -------
    tuple
        (signal, shares) where signal is 'buy', 'sell', or 'hold'
        and shares is an integer number of shares to trade.
    """
    close = data['Close']
    
    if close.iloc[-1] > close.rolling(50).mean().iloc[-1]:
        # Buy: allocate $10,000
        shares = int(10000 / close.iloc[-1])
        return 'buy', shares
    elif holdings > 0 and close.iloc[-1] < close.rolling(10).mean().iloc[-1]:
        # Sell all
        return 'sell', holdings
    else:
        return 'hold', 0
```

### Using Your Strategy

```python
from stocksimpy import Backtester, StockData

data = StockData.from_yfinance(["AAPL"], days_before=365)

# Fixed-size
bt_fixed = Backtester("AAPL", data, my_fixed_strategy)
bt_fixed.run_backtest_fixed()

# Dynamic-size
bt_dynamic = Backtester("AAPL", data, my_dynamic_strategy)
bt_dynamic.run_backtest_dynamic()
```

### Pro Tips

1. **Handle MultiIndex data:** The data might have MultiIndex columns for multi-ticker data. Extract close carefully:

   ```python
   try:
       close = data.loc[:, ('Close', symbol)]  # MultiIndex
   except:
       close = data['Close']  # Single-level
   ```

2. **Avoid lookahead bias:** Only use data up to the current index. Don't use `data.iloc[-1]` then apply shifts; design your calculations properly.

3. **Handle edge cases:** Check if you have enough data before using rolling windows:

   ```python
   if len(data) < 50:
       return 'hold'
   ```

4. **Keep it readable:** Clear variable names and comments make backtests easier to debug:

   ```python
   rsi = compute_rsi(data['Close'], period=14)
   is_oversold = rsi.iloc[-1] < 30
   if is_oversold:
       return 'buy'
   ```

---

## Next Steps

1. **Explore the examples:** Check the `examples/` folder for complete working notebooks.

2. **Read the API docs:** Visit the [full API documentation](api) for detailed method signatures and parameters.

3. **Test multiple strategies:** Compare different strategies on the same data:

   ```python
   from stocksimpy import Backtester, Performance, StockData, Strategy
   
   data = StockData.from_yfinance(["AAPL"], days_before=365)
   strategies = [
       Strategy.buy_all_fixed(),
       Strategy.rsi_momentum_fixed(),
       Strategy.sma_ema_crossover_fixed(),
   ]
   
   for strat in strategies:
       bt = Backtester("AAPL", data, strat)
       bt.run_backtest_fixed()
       perf = Performance(bt)
       print(perf.generate_risk_report())
   ```

4. **Optimize parameters:** Use grid search or random search to find optimal parameter combinations.

5. **Handle real-world data:** Load your own CSV, database, or yfinance data and test on it.

6. **Contribute:** Found a cool strategy? Have an idea? Check [CONTRIBUTING](https://github.com/SuleymanSade/stocksimpy/blob/main/CONTRIBUTING.md) to contribute back!

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'yfinance'"

Install yfinance: `pip install yfinance`

### "ValueError: Input DataFrame must have a 'Date' column or a DatetimeIndex"

Ensure your data has a properly indexed date column. If using a CSV:

```python
df = pd.read_csv("file.csv", index_col="Date", parse_dates=True)
data = StockData(df)
```

### "ValueError: DataFrame contains NaN values"

Your data has missing values. Fill them:

```python
df = df.fillna(method='ffill')  # Forward fill
```

### Strategy returns no trades

Check:

- Your signal logic (is it ever returning 'buy' or 'sell'?)
- Your data has enough rows (some strategies need lookback periods)
- Transaction fees aren't eating all capital (try lowering fees)

### Performance metrics are NaN

This usually means:

- The backtest didn't execute any trades (no portfolio value change)
- Not enough data to compute metrics
- Ensure `run_backtest_fixed()` or `run_backtest_dynamic()` was called

---

## Additional Resources

- **GitHub:** [github.com/SuleymanSade/stocksimpy](https://github.com/SuleymanSade/stocksimpy)
- **Contributing:** See [CONTRIBUTING](https://github.com/SuleymanSade/stocksimpy/blob/main/CONTRIBUTING.md)
- **Code of Conduct:** See [CODE_OF_CONDUCT](https://github.com/SuleymanSade/stocksimpy/blob/main/CODE_OF_CONDUCT.md)

Happy backtesting! 🚀
