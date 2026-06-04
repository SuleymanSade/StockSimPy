# src/stocksimpy/utils/visualize.py
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt


class Visualize:
    """
    Visualization utilities for backtest results.

    This helper provides convenience plotting for single-symbol backtests.
    It expects the provided `backtester` to expose the following attributes:

    - ``backtester.data``: a DataFrame with OHLCV columns. For MultiIndex
      data the implementation selects ``('Close', symbol)`` for the symbol's
      close prices.
    - ``backtester.portfolio.value_history``: a pandas Series of portfolio
      total values indexed by timestamps.
    - ``backtester.portfolio.trade_log``: a pandas DataFrame with executed
      trades and at minimum the columns ``['Date', 'Price', 'Type']`` where
      ``Type`` is ``'buy'`` or ``'sell'``.

    Parameters
    ----------
    backtester : Backtester
        Backtester instance containing historical data, the portfolio, and the
        trade log. The instance is stored on the Visualize object and used by
        the plotting methods.

    Notes
    -----
    - The plotting functions return the ``matplotlib.pyplot`` module (``plt``)
      so callers can call ``plt.show()``, ``plt.savefig(...)``, or continue
      customizing the figure.
    - The helper intentionally performs minimal validation on the input data
      to stay lightweight; callers should ensure the backtester and portfolio
      conform to the expected structure.
    """

    def __init__(self, backtester: Any) -> None:
        self.backtester = backtester

    def visualize_backtest(self) -> Any:
        """
        Plot the backtest price, portfolio value, and executed trades.

        This method renders a two-axis plot: the symbol's close price on the
        left y-axis and the portfolio's total value on the right y-axis. Buy
        and sell executions are annotated on the price axis as upward and
        downward markers respectively.

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
        >>> viz = Visualize(bt)
        >>> plt = viz.visualize_backtest()
        >>> plt.show()
        """

        backtester = self.backtester
        stock_symbol = backtester.symbol
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Plot stock price
        # Support MultiIndex (('Close', symbol)) or flat ('Close') DataFrames.
        try:
            close_prices = backtester.data.loc[:, ("Close", stock_symbol)]
        except Exception:
            close_prices = backtester.data.loc[:, "Close"]

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
            backtester.portfolio.value_history.index,
            backtester.portfolio.value_history,
            label="Portfolio Value",
            color="blue",
        )
        ax2.set_ylabel("Portfolio Value ($)", color="blue")
        ax2.tick_params(axis="y", labelcolor="blue")

        # Plot trades if available
        trade_log = backtester.portfolio.trade_log
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
