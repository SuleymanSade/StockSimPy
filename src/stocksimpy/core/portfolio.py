# src/stocksimpy/core/portfolio.py

from collections import defaultdict
from typing import DefaultDict

import pandas as pd


class Portfolio:
    """Manage a trading portfolio recording cash, holdings, trades and value history.

    Parameters
    ----------
    initial_cap : float, optional
        Starting capital for the portfolio. Default is ``100_000``.

    Notes
    -----
    This class is used by the backtester to execute trades via ``exec_trade``,
    update portfolio value with ``update_value``, and expose ``value_history`` and
    ``trade_log`` for analysis and plotting (see ``stocksimpy.utils.visualize``).
    """

    def __init__(self, initial_cap: float = 100_000):
        self.initial_cap = initial_cap
        self.cash = initial_cap
        self.holdings: DefaultDict[str, int] = defaultdict(
            int
        )  # To prevent code from crashing if a non-initialized symbol is tried to accessed

        self.risk_free_rate = 0.0
        self.value_history = pd.Series(dtype="float64")
        self.date_length = 0
        self.trade_log = pd.DataFrame(
            columns=[
                "Date",
                "Symbol",
                "Type",
                "Price",
                "Shares",
                "TransactionFee",
                "TotalAmount",
            ]
        ).astype(
            {
                "Date": "datetime64[ns]",
                "Symbol": "object",
                "Type": "object",
                "Price": "float64",
                "Shares": "float64",
                "TransactionFee": "float64",
                "TotalAmount": "float64",
            }
        )

    def _log_trade(
        self,
        symbol: str,
        trade_type: str,
        price: float,
        shares: float,
        transaction_fee: float,
        total_amount: float,
        date: pd.Timestamp,
    ):
        """Logs a trade in the trade_log DataFrame.

        Parameters:
        -----------
            symbol (str): The stock symbol.
            trade_type (str): 'buy' or 'sell'.
            price (float): Price per share.
            shares (float): Number of shares traded.
            transaction_fee (float): Transaction fee for the trade.
            total_amount (float): Total amount spent or received in the trade.
            date (pd.Timestamp): Date of the trade.
        """

        add = pd.DataFrame(
            [
                {
                    "Date": date,
                    "Symbol": symbol,
                    "Type": trade_type,
                    "Price": price,
                    "Shares": shares,
                    "TransactionFee": transaction_fee,
                    "TotalAmount": total_amount,
                }
            ]
        )
        self.trade_log = pd.concat([self.trade_log, add], ignore_index=True)

    def exec_trade(
        self,
        symbol: str,
        trade_type: str,
        price: float,
        shares: float,
        date: pd.Timestamp,
        transaction_fee: float = 0.000,
    ):
        """Execute a buy or sell trade, updating cash, holdings, and the trade log.

        Parameters
        ----------
        symbol : str
            Stock symbol to trade.
        trade_type : str
            'buy' or 'sell' (case-insensitive).
        price : float
            Price per share (must be > 0).
        shares : float
            Number of shares to trade (non-negative). Will be converted to int.
        date : pd.Timestamp
            Timestamp of the trade.
        transaction_fee : float, optional
            Transaction fee applied to the trade (default 0.0).

        Notes
        -----
        - For buys, if available cash is insufficient the method reduces the
          number of shares to the maximum affordable (after fee). If the
          transaction fee exceeds available cash, the trade is ignored.
        - For sells, if holdings are less than requested shares the method sells
          all available shares.
        - Each executed trade is logged via ``_log_trade`` and updates
          ``cash`` and ``holdings`` accordingly.

        Raises
        ------
        ValueError
            If ``price <= 0``, ``shares < 0``, ``trade_type`` not in
            {'buy','sell'}, or ``transaction_fee < 0``.
        """
        trade_type = trade_type.lower().rstrip()
        if price <= 0:
            raise ValueError("price cannot be 0 or less than 0")
        if shares < 0:
            raise ValueError("shares cannot be negative")
        if trade_type not in ["buy", "sell"]:
            raise ValueError("trade_type has to be one of the following: 'buy', 'sell'")
        if transaction_fee < 0:
            raise ValueError("transaction_fee cannot be negative")

        shares = int(shares)

        if trade_type == "buy":
            total_cost = price * shares + transaction_fee

            if self.cash < total_cost:
                max_shares_possible = int((self.cash - transaction_fee) / price)

                # Edge case of transaction_fee > self.cash
                if max_shares_possible <= 0:
                    return

                shares = max_shares_possible
                total_cost = price * shares + transaction_fee

            self.cash -= total_cost
            self.holdings[symbol] += shares

            self._log_trade(
                symbol, trade_type, price, shares, transaction_fee, total_cost, date
            )

        else:
            if self.holdings[symbol] < shares:
                shares = self.holdings[symbol]

            total_rev = (shares * price) - transaction_fee
            self.cash += total_rev
            self.holdings[symbol] -= shares

            self._log_trade(
                symbol, trade_type, price, shares, transaction_fee, total_rev, date
            )

    def update_value(self, current_date, current_prices: dict):
        """
        Updates the total value of the portfolio and appends it to value_history.
        This is a corrected version that calculates the total value (cash + holdings).
        """
        holdings_value = 0
        for symbol, num_shares in self.holdings.items():
            if symbol in current_prices:
                holdings_value += num_shares * current_prices[symbol]

        total_value = self.cash + holdings_value
        self.value_history.loc[current_date] = total_value
