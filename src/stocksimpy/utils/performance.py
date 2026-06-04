# src/stocksimpy/utils/performance.py

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from stocksimpy.core.backtester import Backtester


class Performance:
    """
    Compute performance and risk metrics for a backtest execution.

    This class wraps a completed Backtester instance to calculate risk-adjusted
    returns, volatility, drawdown, and other common performance statistics. All
    metrics are computed from the portfolio's value history and trade log.

    Parameters
    ----------
    backtester : Backtester
        A completed Backtester instance with an executed portfolio. The portfolio's
        value_history and trade_log will be used to compute metrics.
    risk_free_rate : float, optional
        Annual risk-free rate used for Sharpe ratio and other risk-adjusted
        calculations. Expressed as a decimal (e.g., 0.02 for 2%). Default is 0.02.

    Attributes
    ----------
    backtester : Backtester
        Reference to the underlying Backtester instance.
    portfolio : Portfolio
        The portfolio object from the backtester.
    symbol : str
        Ticker symbol being analyzed.
    risk_free_rate : float
        Annual risk-free rate for calculations.

    Notes
    -----
    - All return metrics are expressed as decimals (e.g., 0.15 for 15%).
    - Volatility and Sharpe ratio are annualized using the observed trading days
      per year in the portfolio's value history.
    - Drawdown is expressed as a negative decimal (e.g., -0.25 for -25%).
    - Metrics assume that the backtester has already been executed; results may
      be incorrect or zero if the backtest produced no trades or value history.

    Examples
    --------
    >>> bt = Backtester('AAPL', stock_data, strategy)
    >>> bt.run_backtest_fixed()
    >>> perf = Performance(bt, risk_free_rate=0.02)
    >>> report = perf.generate_risk_report()
    >>> print(f"Sharpe Ratio: {report['Sharpe Ratio']:.2f}")
    """

    def __init__(self, backtester: "Backtester", risk_free_rate: float = 0.02) -> None:
        self.backtester = backtester
        self.portfolio = backtester.portfolio
        self.symbol = backtester.symbol
        self.risk_free_rate = risk_free_rate

    def calc_daily_returns(self) -> pd.Series:
        """
        Calculate daily percentage returns from portfolio value history.

        Returns
        -------
        pandas.Series
            Daily returns as decimals (e.g., 0.01 for 1% daily return).
            Index is aligned with the portfolio value history dates.
            Missing values are dropped.
        """
        return self.portfolio.value_history.pct_change().dropna()

    def calc_total_return(self) -> float:
        """
        Calculate total return over the backtest period.

        Computes the percentage return from the initial capital to the final
        portfolio value, expressed as a decimal.

        Returns
        -------
        float
            Total return as a decimal. For example, 0.15 represents a 15% return.
            Returns 0.0 if the portfolio value history is empty.

        Notes
        -----
        Total return is calculated as:

            (final_value - initial_capital) / initial_capital
        """
        value_history = self.portfolio.value_history
        initial_cap = self.portfolio.initial_cap

        if value_history.empty:
            return 0.0

        final_val = value_history.iloc[-1]
        return (final_val - initial_cap) / initial_cap

    def get_annualized_return(self) -> float:
        """
        Calculate annualized return over the backtest period.

        Converts the total return to a compound annual growth rate (CAGR) using
        the date span of the portfolio (start to end). This gives a consistent
        annual rate for comparisons across different time horizons.

        Returns
        -------
        float
            Annualized return as a decimal. For example, 0.12 represents a 12%
            annualized return. Returns 0.0 if date span is zero.

        Notes
        -----
        - The calculation assumes a 365.25-day year to account for leap years.
        - The date span includes all calendar days (weekends, holidays, etc.),
          not just trading days.
        - Formula: ``(1 + total_return) ** (365.25 / days) - 1``
        """
        total_return = self.calc_total_return()
        # For some reason days include all the days from start to end, including weekend official holidays, etc.
        days = self.portfolio.date_length
        if days == 0:
            return 0.0

        annualized_return = (1 + total_return) ** (365.25 / days) - 1

        return annualized_return

    def _get_annualized_trading_days(self) -> float:
        """
        Compute the average trading days per year from the portfolio history.

        This internal method dynamically calculates the number of trading days
        per year observed in the simulation period, ensuring that annualized
        metrics (Sharpe ratio, volatility) are accurate for any time range
        (intraday, weeks, months, or years).

        Returns
        -------
        float
            Average trading days per year. Returns 252 (standard for US equities)
            as a fallback if insufficient data is available.

        Notes
        -----
        - If the portfolio has fewer than 2 entries or spans zero years,
          the method returns 252.
        - Formula: ``total_trading_days / total_years``
        - This ensures volatility and Sharpe ratios are correctly annualized
          even for partial-year or multi-year backtests.
        """
        value_history = self.portfolio.value_history
        if len(value_history) < 2:
            return 252  # Fallback to standard 252 trading days per year

        start_date = value_history.index.min()
        end_date = value_history.index.max()

        # Total time span in years
        total_years = (end_date - start_date).days / 365.25

        # Total number of trading days
        total_trading_days = len(value_history)

        if total_years == 0 or total_trading_days < 2:
            return 252  # Avoid division by zero

        # Average trading days per year
        return total_trading_days / total_years

    def calc_max_drawdown(self) -> float:
        """
        Calculate the maximum drawdown during the backtest period.

        Maximum drawdown is the peak-to-trough decline from the highest
        portfolio value to the lowest subsequent value. It measures the worst
        cumulative loss experienced by the strategy.

        Returns
        -------
        float
            Maximum drawdown as a negative decimal. For example, -0.25 represents
            a 25% drawdown from peak to trough. Returns 0.0 if the portfolio
            value history is empty.

        Notes
        -----
        - Formula: ``min((value - cummax(value)) / cummax(value))``
        - A drawdown of -0.2 means the portfolio fell 20% from its peak.
        - All drawdown values are negative or zero.
        """
        value_history = self.portfolio.value_history
        if value_history.empty:
            return 0.0

        rolling_max = value_history.cummax()
        drawdowns = (value_history - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()

        return max_drawdown

    def calc_sharpe_ratio(self) -> float:
        """
        Calculate the annualized Sharpe ratio.

        The Sharpe ratio measures risk-adjusted return by comparing the excess
        daily return (above the risk-free rate) to the standard deviation of
        daily returns. Higher Sharpe ratios indicate better risk-adjusted
        performance.

        Returns
        -------
        float
            Annualized Sharpe ratio. A typical range is 0 to 3, with values
            above 1.0 generally considered good and above 2.0 excellent.
            Returns 0.0 if daily returns are empty or have zero volatility.

        Notes
        -----
        - The calculation uses the dynamic annualization factor from
          ``_get_annualized_trading_days()`` to adapt to the backtest period.
        - Formula:

            (mean_daily_excess_return / daily_volatility) * sqrt(annualization_factor)

        - The risk-free rate is converted to a daily rate before computation.
        - Sharpe ratios above 1.0 typically indicate strong risk-adjusted returns.

        Examples
        --------
        >>> perf = Performance(bt)
        >>> sharpe = perf.calc_sharpe_ratio()
        >>> if sharpe > 1.0:
        ...     print("Good risk-adjusted performance")
        """
        daily_returns = self.calc_daily_returns()

        if daily_returns.empty or daily_returns.std() == 0:
            return 0.0

        # 1. Get the dynamic annualization factor
        annualization_factor = self._get_annualized_trading_days()

        # 2. Adjust annual risk-free rate to daily rate
        # Daily risk-free rate = (1 + R)^(1/T) - 1
        daily_risk_free_rate = (1 + self.risk_free_rate) ** (
            1 / annualization_factor
        ) - 1

        # 3. Calculate the daily Sharpe Ratio
        sharpe_ratio = (
            daily_returns.mean() - daily_risk_free_rate
        ) / daily_returns.std()

        # 4. Annualize the ratio using the square root of the annualization factor
        return sharpe_ratio * np.sqrt(annualization_factor)

    # TODO: for future implementation, the current version is not correct
    """
    def calc_sortino_ratio(self) -> float:
        daily_returns = self.calc_daily_returns()
        
        if daily_returns.empty: return 0.0

        annualization_factor = self._get_annualized_trading_days()
        # Daily Minimum Acceptable Return (MAR) is set to the daily risk-free rate
        daily_mar = (1 + self.risk_free_rate)**(1/annualization_factor) - 1
        
        # 1. Calculate Downside Deviation (Downside Risk)
        # Identify returns below the MAR (daily risk-free rate)
        downside_returns = daily_returns[daily_returns < daily_mar]
        
        # If there are no downside returns, the deviation is 0. 
        if downside_returns.empty:
            # If no downside, the ratio is effectively infinite, but we safely return 0.0 or the annualized excess return.
            # Returning 0.0 ensures safety in case of division by zero later.
            return (daily_returns.mean() - daily_mar) * annualization_factor

        # Calculate the sum of squared differences only for returns below MAR.
        # The divisor must be the total number of periods (len(daily_returns)) for the population downside deviation.
        sum_of_squares = ((downside_returns - daily_mar)**2).sum()
        downside_deviation = np.sqrt(sum_of_squares / len(daily_returns))
        
        if downside_deviation == 0:
            return 0.0

        # 2. Calculate Daily Sortino Ratio
        daily_excess_return = daily_returns.mean() - daily_mar
        sortino_ratio = daily_excess_return / downside_deviation
        
        # 3. Annualize the ratio
        return sortino_ratio * np.sqrt(annualization_factor)
    """

    # TODO: for future implementation
    """
    def calculate_calmar_ratio(self) -> float:
        pass
    """

    def generate_risk_report(self) -> pd.Series:
        """
        Generate a comprehensive performance and risk report.

        Computes and aggregates the most important performance metrics into a
        single pandas Series for easy inspection and comparison. All metrics are
        computed lazily at the time of the call.

        Returns
        -------
        pandas.Series
            A labeled one-dimensional array with keys:

            - ``'Total Return'`` (float): Decimal total return from initial
              capital to final portfolio value.
            - ``'Annualized Return'`` (float): Geometric annualized return over
              the backtest period.
            - ``'Max Drawdown'`` (float): Maximum peak-to-trough decline as a
              negative decimal (e.g., -0.25 for -25%).
            - ``'Sharpe Ratio'`` (float): Risk-adjusted return metric annualized
              using dynamic trading days. Generally, values > 1.0 are good.
            - ``'Volatility'`` (float): Annualized standard deviation of daily
              returns, expressed as a decimal (e.g., 0.15 for 15%).

        Notes
        -----
        - All return metrics are expressed as decimals (0.15 = 15%).
        - All metrics return 0.0 if insufficient data is available.
        - Sortino Ratio is deliberately omitted pending implementation verification;
          uncomment the corresponding line in the code once ready.
        - This is the recommended entry point for quick portfolio evaluation.

        Examples
        --------
        >>> report = perf.generate_risk_report()
        >>> print(report)
        Total Return           0.150000
        Annualized Return      0.120000
        Max Drawdown          -0.250000
        Sharpe Ratio           1.450000
        Volatility             0.180000
        dtype: float64
        """
        return pd.Series(
            {
                "Total Return": self.calc_total_return(),
                "Annualized Return": self.get_annualized_return(),
                "Max Drawdown": self.calc_max_drawdown(),
                "Sharpe Ratio": self.calc_sharpe_ratio(),
                # 'Sortino Ratio': self.calc_sortino_ratio(),  # Uncomment when implemented
                "Volatility": self.calc_volatility(),
            }
        )

    def calc_volatility(self) -> float:
        """
        Calculate annualized portfolio volatility.

        Volatility is the standard deviation of daily returns, annualized using
        the observed trading days per year. Higher volatility indicates more
        erratic price swings; lower volatility indicates more stable returns.

        Returns
        -------
        float
            Annualized volatility as a decimal. For example, 0.20 represents 20%
            annualized volatility. Returns 0.0 if daily returns are empty.

        Notes
        -----
        - Formula: ``daily_returns.std() * sqrt(annualization_factor)``
        - Annualization factor is dynamically computed from trading days observed
          in the portfolio history.
        - Volatility is a key input to the Sharpe ratio calculation.

        Examples
        --------
        >>> vol = perf.calc_volatility()
        >>> print(f"Annualized volatility: {vol*100:.1f}%")
        Annualized volatility: 18.5%
        """
        daily_returns = self.calc_daily_returns()
        if daily_returns.empty:
            return 0.0
        annualization_factor = self._get_annualized_trading_days()
        return daily_returns.std() * np.sqrt(annualization_factor)
