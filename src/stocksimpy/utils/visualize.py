import matplotlib.pyplot as plt


class Visualize:
  """
  Visualization utilities for backtest results.

  This helper provides convenience plotting for single-symbol backtests.
  It expects the provided `portfolio` to expose the following attributes:
    
  Parameters
  ----------
  portfolio : Portfolio
      The portfolio instance from the backtester_result containing value history

  Attributes.
  ----------
  portfolio : Portfolio
      The portfolio instance from the backtester_result containing value history
      and trade log.

  Notes
  -----
  - The plotting functions return the ``matplotlib.pyplot`` module (``plt``)
    so callers can call ``plt.show()``, ``plt.savefig(...)``, or continue
    customizing the figure.
  - The helper intentionally performs minimal validation on the input data
    to stay lightweight; callers should ensure the backtester and portfolio
    conform to the expected structure.
  """
  
  def __init__(self, portfolio):
    self.portfolio = portfolio

  def run(self, stock_data=None) -> plt:
    """
    Plot the portfolio value and executed trades, optionally with stock data.
    This method chooses between plotting with or without stock data based
    on the presence of the `stock_data` parameter.
    
    Parameters
    ----------
    stock_data : DataFrame, optional
        Historical stock data for the symbol. If provided, the plot will
        include the stock's close price alongside portfolio value and trades.
        Default is ``None``.
        
    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module used to create the figure. Call
        ``plt.show()`` to display the plot or ``plt.savefig(...)`` to save
        it to disk.
        
    Examples
    --------
    >>> viz = Visualize(portfolio)
    >>> plt = viz.run(stock_data)
    >>> plt.show()
    """
    
    if stock_data is not None:
      self.visualize_portfolio_stock_data(self.portfolio, stock_data)
    else:
      self.visualize_portfolio(self.portfolio)

  def visualize_portfolio_stock_data(self, stock_data) -> plt:
    """
    Plot the backtest price, portfolio value, and executed trades.

    This method renders a two-axis plot: the symbol's close price on the
    left y-axis and the portfolio's total value on the right y-axis. Buy
    and sell executions are annotated on the price axis as upward and
    downward markers respectively.
    
    Parameters
    ----------
    portfolio : Portfolio
        Portfolio instance containing the value history and trade log.
    stock_data : DataFrame
        Historical stock data for the symbol.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module used to create the figure. Call
        ``plt.show()`` to display the plot or ``plt.savefig(...)`` to save
        it to disk.

    Notes
    -----
    - The method uses ``backtester.symbol`` to choose the symbol to plot.
    - For MultiIndex DataFrames produced by e.g. yfinance this method
      selects the close series using ``('Close', symbol)``. If your data
      is flat (no MultiIndex), ensure the close prices are available at
      the column ``'Close'``.
    - The trade log is expected to contain at least the columns ``Date``,
      ``Price``, and ``Type`` where ``Type`` is ``'buy'``/``'sell'``.

    Examples
    --------
    >>> viz = Visualize()
    >>> plt = viz.visualize_portfolio_stock_data(portfolio, stock_data)
    >>> plt.show()
    """
    portfolio = self.portfolio
    stock_symbol = portfolio.symbol
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot stock price
    # Support MultiIndex (('Close', symbol)) or flat ('Close') DataFrames.
    try:
      close_prices = stock_data.to_dataframe().loc[:, ("Close", stock_symbol)]
    except Exception:
      close_prices = stock_data.to_dataframe().loc[:, "Close"]

    ax1.plot(
      close_prices.index,
      close_prices,
      label="Close Price",
      color="gray",
      alpha=0.5,
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Price ($)", color="gray")
    ax1.tick_params(axis="y", labelcolor="gray")

    # Plot portfolio value on second y-axis
    ax2 = ax1.twinx()
    ax2.plot(
      portfolio.value_history.index,
      portfolio.value_history,
      label="Portfolio Value",
      color="blue",
    )
    ax2.set_ylabel("Portfolio Value ($)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Plot trades if available
    trade_log = portfolio.trade_log
    if not trade_log.empty:
      buy_trades = trade_log[trade_log["Type"] == "buy"]
      sell_trades = trade_log[trade_log["Type"] == "sell"]

      if not buy_trades.empty:
        ax1.scatter(
          buy_trades["Date"],
          buy_trades["Price"],
          color="green",
          marker="^",
          s=100,
          label="Buy",
        )
      if not sell_trades.empty:
        ax1.scatter(
          sell_trades["Date"],
          sell_trades["Price"],
          color="red",
          marker="v",
          s=100,
          label="Sell",
        )

    plt.title(f"{stock_symbol} Backtest Performance with Trades")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.9))
    plt.tight_layout()

    return plt

  def visualize(self) -> plt:
    """
    Plot the portfolio value and executed trades.

    This method renders a plot of portfolio total value over time. Buy and sell
    executions are annotated on the price axis as upward and downward markers
    respectively.

    Returns
    -------
    matplotlib.pyplot
        The ``matplotlib.pyplot`` module used to create the figure. Call
        ``plt.show()`` to display the plot or ``plt.savefig(...)`` to save
        it to disk.

    Notes
    -----
    - The trade log is expected to contain at least the columns ``Date``,
      ``Price``, and ``Type`` where ``Type`` is ``'buy'``/``'sell'``.

    Examples
    --------
    >>> viz = Visualize()
    >>> plt = viz.visualize_portfolio(portfolio)
    >>> plt.show()
    
    >>> # Alternatively, directly from a backtester_result
    >>> btres = backtester.run()
    >>> plt = btres.get_visualization().visualize_portfolio(btres.portfolio)
    >>> plt.show()
    """
    portfolio = self.portfolio
    
    stock_symbol = portfolio.symbol
    fig, ax1 = plt.subplots(figsize=(12, 6))
    trade_log = portfolio.trade_log

    ax1.plot(
      portfolio.value_history.index,
      portfolio.value_history,
      label="Portfolio Value",
      color="blue",
    )
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Portfolio Value ($)", color="blue")
    ax1.tick_params(axis="y", labelcolor="blue")

    if not trade_log.empty:
      buy_trades = trade_log[trade_log["Type"] == "buy"]
      sell_trades = trade_log[trade_log["Type"] == "sell"]

      if not buy_trades.empty:
        ax1.scatter(
          buy_trades["Date"],
          buy_trades["Price"],
          color="green",
          marker="^",
          s=100,
          label="Buy",
        )
      if not sell_trades.empty:
        ax1.scatter(
          sell_trades["Date"],
          sell_trades["Price"],
          color="red",
          marker="v",
          s=100,
          label="Sell",
        )

    plt.title(f"{stock_symbol} Portfolio Performance")
    plt.tight_layout()

    return plt
