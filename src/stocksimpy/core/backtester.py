# src/stocksimpy/backtester.py

from typing import Literal
import pandas as pd
import inspect

from .portfolio import Portfolio
from .stock_data import StockData
from stocksimpy import Visualize, Performance



class BacktestResult:
    """
    Store the results of a backtest execution.
    This class encapsulates the portfolio resulting from a backtest run,
    providing methods to access performance metrics and visualization tools.
    
    Parameters
    ----------
    portfolio : Portfolio, optional
        The Portfolio instance resulting from the backtest execution.
        
    Attributes
    ----------
    visualize : Visualize
        Visualization helper initialized with the backtest's portfolio.
    performance : Performance
        Performance helper initialized with the backtest's portfolio.
    portfolio : Portfolio
        The Portfolio instance resulting from the backtest execution.
        
    Notes
    -----
    - The `get_performance()` method returns the Performance instance for
      calculating metrics like Sharpe ratio, max drawdown, etc.
    - The `get_visualize()` method returns the Visualize instance for
      plotting portfolio value and trades.
    """

    def __init__(self, portfolio: Portfolio):
        self.visualize = Visualize(portfolio)
        self.performance = Performance(portfolio)
        self.portfolio = portfolio

    # def performance(self) -> Performance:
    #     """
    #     Get the Performance instance for this backtest result.

    #     Returns
    #     -------
    #     Performance
    #         The Performance instance initialized with the backtest's portfolio.

    #     Examples
    #     --------
    #     >>> btres = bt.run()
    #     >>> perf = btres.performance()
    #     >>> sharpe = perf.calc_sharpe_ratio()
    #     >>> print(f"Sharpe Ratio: {sharpe:.2f}")
    #     """
    #     return self.performance
    
    # def visualize(self) -> Visualize:
    #     """
    #     Get the Visualize instance for this backtest result.

    #     Returns
    #     -------
    #     Visualize
    #         The Visualize instance initialized with the backtest's portfolio.

    #     Examples
    #     --------
    #     >>> btres = bt.run()
    #     >>> viz = btres.visualize()
    #     >>> plt = viz.visualize_portfolio(btres.portfolio)
    #     >>> plt.show()
    #     """
    #     return self.visualize

    def generate_report(self) -> dict:
        """
        Generate a summary report of the backtest execution.

        Computes final portfolio value, total return percentage, and the number
        of trades executed. Returns an empty-portfolio default if no backtest
        has been run or the portfolio is empty.

        Returns
        -------
        dict
            A dictionary with keys:

            - ``'final_value'`` (float): Final portfolio value (cash + stock value).
            - ``'total_return_percent'`` (float): Percentage return from initial
            capital (e.g., 15.5 for 15.5% return).
            - ``'number_of_trades'`` (int): Total number of trades executed
            during the backtest.

        Examples
        --------
        >>> report = btres.generate_report()
        >>> print(f"Final Value: ${report['final_value']:.2f}")
        >>> print(f"Return: {report['total_return_percent']:.2f}%")
        >>> print(f"Trades: {report['number_of_trades']}")
        """
        if self.portfolio.value_history.empty:
            return {
                "final_value": self.portfolio.initial_cap,
                "total_return_percent": 0.0,
                "number_of_trades": 0,
            }

        final_value = self.portfolio.value_history.iloc[-1]
        total_return = (
            final_value - self.portfolio.initial_cap
        ) / self.portfolio.initial_cap

        return {
            "final_value": final_value,
            "total_return_percent": total_return * 100,
            "number_of_trades": len(self.portfolio.trade_log),
        }


class Backtester:
    """
    Backtester for running simple single-symbol strategies on historical data.

    This class executes either fixed-size or dynamic-size trading strategies
    against OHLCV time-series data. At each timestep, the strategy is invoked
    with historical data up to that point. The internal Portfolio tracks cash,
    holdings, executed trades, and value history.

    Parameters
    ----------
    symbol : str
        Ticker symbol to backtest. For multi-ticker DataFrames, only this
        symbol's data will be used.
    data : StockData
        StockData instance containing OHLCV data. The underlying DataFrame
        will be extracted and filtered to the selected symbol if necessary.
    strategy : callable
        Strategy function to be executed at each timestep.
        In fixed mode: ``strategy(historic_df) -> signal``
        In dynamic mode: ``strategy(historic_df, holdings) -> (signal, shares)``
        where `signal` is one of {'buy', 'sell', 'hold'}.
    initial_cap : float, optional
        Starting cash for the portfolio. Default is 100000.
    transaction_fee : float, optional
        Flat fee applied to every executed trade. Default is 0.0.
    trade_amount : float, optional
        Dollar amount allocated per trade when using fixed-size backtesting.
        Ignored in dynamic mode. Default is 10000.

    Attributes
    ----------
    symbol : str
        Ticker symbol used for this backtest.
    data : pandas.DataFrame
        Historical OHLCV data for `symbol`, indexed by datetime.
    strategy : callable
        Strategy function executed at each timestep.
    initial_cap : float
        Initial portfolio cash.
    transaction_fee : float
        Fee applied to each trade.
    trade_amount : float
        Dollar amount allocated per trade during fixed-size backtests.
    portfolio : Portfolio
        Tracks cash, holdings, executed trades, and historical total value.

    Notes
    -----
    **Fixed-size mode**
        - Strategy receives only historical OHLCV data: ``strategy(df) -> signal``.
        - Number of shares is computed as:
          ``shares = int(trade_amount / close_price)``.
        - If close price is zero or NaN, shares will be zero and no trade occurs.

    **Dynamic-size mode**
        - Strategy receives historical data and current holdings:
          ``strategy(df, holdings) -> (signal, shares)``.
        - Strategy must return ``(signal, shares)``.
        - If the signal is not 'buy' or 'sell', no trade is executed.

    **Multi-ticker data**
        If the input `StockData` contains multi-level columns (e.g., the output
        of `from_yfinance`), only the subcolumns for `symbol` are passed to
        the strategy. All other symbols are ignored.

    **State Mutation**
        Calling a backtest method mutates the internal portfolio state.
        Run `generate_report()` for results or inspect:
        - `portfolio.value_history`
        - `portfolio.trade_log`

    Examples
    --------
    >>> def strat(df):
    ...     return 'buy' if df['Close'].iloc[-1] > df['Close'].rolling(10).mean().iloc[-1] else 'hold'
    >>> bt = Backtester("AAPL", stock_data, strat, trade_amount=500)
    >>> bt.run_backtest_fixed()   # doctest: +SKIP
    >>> bt.generate_report()      # doctest: +SKIP
    """

    def __init__(
        self,
        data: StockData,
        initial_cap: float = 100_000,
        transaction_fee: float = 0.000,
        trade_amount: float = 10_000,
    ):
        self.data = data.to_dataframe()
        self.initial_cap = initial_cap
        self.transaction_fee = transaction_fee

        self.trade_amount = trade_amount

        # Set initial portfolio values
        self.portfolio.date_length = (
            self.data.index.max() - self.data.index.min()
        ).days

    def _process_trade(self, portfolio: Portfolio, symbol : str, signal: str, shares: int, price: float, date):
        """
        Execute a single trade and update the portfolio value.

        This internal helper method wraps the low-level Portfolio methods to
        ensure consistent state management. For buy/sell signals with positive
        share counts, a trade is executed via ``Portfolio.exec_trade``. Then,
        the portfolio value is updated to reflect the current price.

        Parameters
        ----------
        signal : str
            Trade action. Expected values are ``'buy'`` or ``'sell'``; other
            values result in no trade (only a value update).
        shares : int
            Number of shares to trade. Must be non-negative. If zero, no trade
            is executed (only the portfolio value is updated).
        price : float
            Per-share execution price for the trade.
        date : datetime-like
            Timestamp for recording the trade. Used in the trade log and to
            update the portfolio value history.

        Notes
        -----
        - This is an internal helper; it is called by ``run_backtest_fixed``
        and ``run_backtest_dynamic`` and is not intended for direct external
        use.
        - The portfolio value is updated even if no trade is executed, to
        ensure accurate daily/timestep valuation.
        """

        if signal in ["buy", "sell"]:
            portfolio.exec_trade(
                symbol=symbol,
                trade_type=signal,
                price=price,
                shares=shares,
                date=date,
                transaction_fee=self.transaction_fee,
            )

        self.portfolio.update_value(date, {symbol: price})

    def run(self,  strategy : function, symbol : str, mode: Literal["fixed", "dynamic", "auto"] = "auto") -> BacktestResult:
        """
        Execute a backtest using fixed trade amounts for each signal.

        Iterates through historical data and calls the strategy at each timestep.
        The strategy receives a DataFrame containing all available historical
        values up to the current timestamp. The number of shares to trade is
        computed as:

            shares = int(self.trade_amount / close_price)

        For multi-ticker DataFrames (e.g., from yfinance), only the columns
        corresponding to ``self.symbol`` are passed to the strategy. For flat
        DataFrames, the entire DataFrame slice is passed unchanged.
        
        Parameters
        ----------
        mode : {'fixed', 'dynamic', 'precomputed'}, optional
            Backtest execution mode. Options are:
            - ``'fixed'``: Use fixed trade amounts per signal.
            - ``'dynamic'``: Use dynamic share counts returned by the strategy.
            - ``'precomputed'``: Skip execution; assume portfolio is pre-filled.
            - ``'auto'`` (default): Choose ``'fixed'`` if ``trade_amount`` is
              set, else ``'dynamic'``.

        Notes
        -----
        - If the close price is zero or missing at a timestep, the computed
        share count will be zero and no trade will be executed.
        - This method mutates the internal portfolio state directly and does
        not return anything. After execution, use ``generate_report()`` or
        inspect ``portfolio.value_history`` and ``portfolio.trade_log`` to
        access results.

        Examples
        --------
        >>> def strat(df):
        ...     return 'buy' if df['Close'].iloc[-1] > df['Close'].rolling(10).mean().iloc[-1] else 'hold'
        >>> bt = Backtester('AAPL', stock_data, strat, trade_amount=500)
        >>> bt.run_backtest_fixed()
        >>> bt.generate_report()

        Raises
        ------
        Exception
            Any exception raised inside the strategy will propagate to the
            caller.
        """
        
        if len(inspect.signature(function).parameters) > 2:
            pass
            #TODO: throw error
        
        portfolio = Portfolio(self.initial_cap)
        
        if mode == "auto":
            mode = "dynamic" if len(inspect.signature(function).parameters) == 2 else "fixed"

        for i in range(1, len(self.data)):
            current_date = self.data.index[i]

            historic_vals = self.data.iloc[: i + 1]
            # If MultiIndex (e.g. yfinance data) only get the data for a specific symbol
            if isinstance(self.data.columns, pd.MultiIndex):
                historic_vals = historic_vals.xs(symbol, axis=1, level=-1)

            if mode == "dynamic":
                signal, shares_to_trade = strategy(
                    historic_vals, portfolio.holdings[symbol]
                )
            elif mode == "fixed":
                signal = strategy(historic_vals)

            price = self.data.loc[current_date, ("Close", symbol)]

            if price > 0:  
                if mode == "fixed":
                    shares_to_trade = int(self.trade_amount / price)
            else:
                shares_to_trade = 0

            self._process_trade(signal, portfolio, symbol, shares_to_trade, price, current_date)

        return BacktestResult(portfolio)
    
    # For the following code, migrated to different approaches

    # def run_backtest_dynamic(self):
    #     """
    #     Execute a backtest using dynamic trade sizes.

    #     At each timestep, the strategy is called with two arguments:
    #     ``(df, holdings)`` where:

    #     - ``df`` is the historical DataFrame up to the current timestamp
    #     (filtered to ``self.symbol`` for MultiIndex data).
    #     - ``holdings`` is the current number of shares held.

    #     The strategy must return a tuple ``(signal, shares)``, where ``shares``
    #     is an integer specifying the number of shares to buy or sell.

    #     Notes
    #     -----
    #     - The portfolio is updated in place. This method does not return a value.
    #     - Any strategy that returns a non-tuple or a tuple of incorrect length
    #     will raise a TypeError.
    #     - All exceptions raised inside the strategy propagate directly to the
    #     caller, allowing debugging of strategy logic.
    #     - The DataFrame slice passed to the strategy includes *all* history up
    #     to the current timestamp, enabling rolling-window or stateful logic.

    #     Returns
    #     -------
    #     None

    #     Raises
    #     ------
    #     TypeError
    #         If the strategy does not return a two-item tuple.
    #     Exception
    #         Any other error raised inside the strategy or during data access.

    #     Examples
    #     --------
    #     >>> def dyn(df, holdings):
    #     ...     return ('buy', 5) if df['Close'].iloc[-1] < df['Close'].rolling(20).mean().iloc[-1] else ('sell', 5)
    #     >>> bt = Backtester('AAPL', stock_data, dyn)
    #     >>> bt.run_backtest_dynamic()
    #     >>> bt.generate_report()
    #     """

    #     for i in range(1, len(self.data)):
    #         current_date = self.data.index[i]
    #         historic_vals = self.data.iloc[: i + 1]

    #         try:
    #             signal, shares_to_trade = self.strategy(
    #                 historic_vals, self.portfolio.holdings[self.symbol]
    #             )
    #         except ValueError:
    #             raise TypeError(
    #                 "strategy function should return a tuple (signal, shares) for dynamic sizing."
    #             )

    #         historic_vals = self.data.iloc[: i + 1]
    #         # If MultiIndex (e.g. yfinance data) only get the data for a specific symbol
    #         if isinstance(self.data.columns, pd.MultiIndex):
    #             historic_vals = historic_vals.xs(self.symbol, axis=1, level=-1)

    #         price = self.data.loc[current_date, ("Close", self.symbol)]

    #         self._process_trade(signal, shares_to_trade, price, current_date)

    #     return BacktestResult(self.portfolio)

    # @DeprecationWarning
    # def generate_report(self) -> dict:
    #     """
    #     # This function is deprecated.
    #     # Please use the resulting `BacktestResult.generate_report()` instead.

    #     Generate a summary report of the backtest execution.

    #     Computes final portfolio value, total return percentage, and the number
    #     of trades executed. Returns an empty-portfolio default if no backtest
    #     has been run or the portfolio is empty.

    #     Returns
    #     -------
    #     dict
    #         A dictionary with keys:

    #         - ``'final_value'`` (float): Final portfolio value (cash + stock value).
    #         - ``'total_return_percent'`` (float): Percentage return from initial
    #         capital (e.g., 15.5 for 15.5% return).
    #         - ``'number_of_trades'`` (int): Total number of trades executed
    #         during the backtest.

    #     Examples
    #     --------
    #     >>> report = bt.generate_report()
    #     >>> print(f"Final Value: ${report['final_value']:.2f}")
    #     >>> print(f"Return: {report['total_return_percent']:.2f}%")
    #     >>> print(f"Trades: {report['number_of_trades']}")
    #     """
    #     if self.portfolio.value_history.empty:
    #         return {
    #             "final_value": self.portfolio.initial_cap,
    #             "total_return_percent": 0.0,
    #             "number_of_trades": 0,
    #         }

    #     final_value = self.portfolio.value_history.iloc[-1]
    #     total_return = (
    #         final_value - self.portfolio.initial_cap
    #     ) / self.portfolio.initial_cap

    #     return {
    #         "final_value": final_value,
    #         "total_return_percent": total_return * 100,
    #         "number_of_trades": len(self.portfolio.trade_log),
    #     }
