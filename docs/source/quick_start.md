# Quick Start

This guide shows the minimum to load data, create strategy, run backtesting, and inspect the results.

## Install

```bash
pip install stocksimpy
```

## Minimal working example (copy & run)

```python
from stocksimpy import StockData, Backtester, Strategy, Performance, Visualize

# Load ~1 year of data (requires yfinance)
data = StockData.from_yfinance(["AAPL"], days_before=365)

# Use a built-in fixed-size strategy (RSI momentum)
strategy = Strategy.rsi_momentum_fixed()

# Create a backtester instance
bt = Backtester(symbol="AAPL", data=data, strategy=strategy)

# Run the backtest (fixed-size strategies use run_backtest_fixed)
bt.run_backtest_fixed()

# Performance summary
perf = Performance(bt)
print(perf.generate_risk_report())

# Visualization
viz = Visualize(bt)
figure = viz.visualize_backtest()
figure.show()
```

This gives you real trades, P/L values, risk stats, and a basic equity curve in seconds.

## Writing your own strategy

Stocksimpy supports two strategy styles

### Fixed strategies (simplest)

- Same trade size on every buy/sell
- Strategy sees past data only
- Must return one of: "buy", "sell", "hold"

Minimal custom fixed strategy example:

```Python
def sma_crossover_fixed(data):
    close = data["Close"]
    short = close.rolling(20).mean()
    long = close.rolling(50).mean()

    if short.iloc[-1] > long.iloc[-1]:
        return "buy"
    elif short.iloc[-1] < long.iloc[-1]:
        return "sell"
    else:
        return "hold"
```

Usage:

```Python
bt = Backtester("AAPL", data, sma_crossover_fixed)
bt.run_backtest_fixed()
```

### Dynamic strategies (more control)

- Trade size can change per trade
- Strategy receives past data + current holdings
- Must return (signal, trade_amount)
  - signal in "buy" | "sell" | "hold"
  - trade_amount is an integer (shares)
- Minimal dynamic strategy example

Minimal custom dynamic strategy

```Python
def rsi_dynamic(data, holdings):
    rsi = Strategy.rsi(data["Close"], period=14)  # or your own RSI logic
    latest = rsi.iloc[-1]

    if latest < 30:
        return "buy", 10      # accumulate
    elif latest > 70 and holdings > 0:
        return "sell", holdings  # exit position
    else:
        return "hold", 0
```

Usage:

```Python
bt = Backtester("AAPL", data, rsi_dynamic)
bt.run_backtest_dynamic()
```

## Useful shortcuts

- Built-in strategies live in Strategy
- Fixed: run_backtest_fixed()
- Dynamic: run_backtest_dynamic()
- Visuals: Visualize(bt).visualize_backtest()
- Reports: Performance(bt).generate_risk_report()

## When to use fixed vs dynamic

| Feature                    | Fixed           | Dynamic                         |
| -------------------------- | --------------- | ------------------------------- |
| Trade size                 | constant        | variable                        |
| Access to current holdings | no              | yes                             |
| Return value               | "buy/sell/hold" | ("buy/sell/hold", amount)       |
| Best for                   | simple rules    | scaling in/out, position sizing |

If you are unsure start with fixed.

## See the docs for deeper guide and more advanced features
