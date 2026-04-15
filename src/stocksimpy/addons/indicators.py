# src/stocksimpy/addons/indicators.py

import math

import numpy as np
import pandas as pd


class Indicators:
    """This module provides basic functions for calculating various technical indicators.

    Summary
    -------
    Provides implementations of commonly used technical indicators such as SMA, RSI,
    MACD, DEMA, TEMA and related helper smoothing functions. These functions operate on
    pandas Series and return Series or DataFrame results suitable for usage in
    strategy code and performance analysis.
    """

    def _validate_indicator_inputs(
        data_series: pd.Series, window: int, min_data_length: int = 1
    ) -> None:
        """Validate common inputs for indicator calculations.

        Summary
        -------
        Checks that `data_series` is a pandas Series, non-empty, numeric,
        that `window` is a positive integer, and that the series length is
        sufficient for the requested window and `min_data_length`.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series to validate.
        window : int
            The window period to validate.
        min_data_length : int, optional
            The minimum required length of the data_series after accounting for the window.
            Defaults to 1, meaning data_series length must be at least ``window``.
            Set to 0 if the indicator can handle ``window > len(data_series)`` by returning NaNs.

        Raises
        ------
        TypeError
            If ``data_series`` is not a pandas Series or is non-numeric.
        ValueError
            If ``data_series`` is empty, ``window`` is not a positive integer,
            or if ``data_series`` is too short for the ``window`` (and ``min_data_length`` check).
        """
        if type(window) is not int:
            raise TypeError("Window has to be an integer")
        if not isinstance(data_series, pd.Series):
            raise TypeError("Input 'data_series' must be a pandas Series.")
        if data_series.empty:
            raise ValueError("Input 'data_series' cannot be empty.")
        if not pd.api.types.is_numeric_dtype(data_series.dtype):
            raise TypeError("Input 'data_series' must be a numerical pandas Series")
        if not isinstance(window, int) or window <= 0:
            raise ValueError("Window must be a positive integer.")
        if len(data_series) < (
            window + (min_data_length - 1)
        ):  # Adjust min_data_length logic
            raise ValueError(
                f"Input data series length ({len(data_series)}) is too short for the specified window ({window}) "
                f"and required minimum data length (at least {window + min_data_length -1} entries needed). "
                "Please provide more data or a smaller window."
            )

    # -----------------------------
    # DIFFERENT TYPES OF EMA

    def calculate_sma(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate the Simple Moving Average (SMA) of a given data series.

        Summary
        -------
        Computes the rolling mean over the specified window. The first
        ``window - 1`` entries will be NaN.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series for which to calculate the SMA.
        window : int
            The window size (number of periods) to use for the SMA calculation.

        Returns
        -------
        dict
            A dictionary with key ``sma_{window}`` containing a pandas Series with the SMA values.
            The initial ``window - 1`` values will be NaN.
        """
        Indicators._validate_indicator_inputs(data_series=data_series, window=window)

        return {f"sma_{window}": data_series.rolling(window=window).mean()}

    def calculate_wma(data_series: pd.Series, window: int = 14) -> dict:
        """Calculates the Weighted Moving Average (WMA) for a pandas Series.

        Summary
        -------
        Computes a linearly-weighted average over the specified window where
        more recent observations receive greater weight.

        Parameters
        ----------
        data_series : pd.Series
            The input pandas Series for which to calculate the WMA.
        window : int
            The size of the moving window. This must be a positive integer.

        Returns
        -------

            dict
            A dictionary with key ``wma_{window}`` containing a pandas Series with the Weighted Moving Average.
            The first ``window - 1`` values will be NaN.
        """
        Indicators._validate_indicator_inputs(data_series, window)

        weights = pd.Series(np.arange(1, window + 1))
        weights_total = weights.sum()

        wma_series = pd.Series(index=data_series.index, dtype=float)

        for i in range(window - 1, len(data_series)):
            current_window_data_values = data_series.iloc[i - window + 1 : i + 1].values

            weighted_sum = (current_window_data_values * weights).sum()

            wma = weighted_sum / weights_total
            wma_series.iloc[i] = wma

        return {f"wma_{window}": wma_series}

    def calculate_ema(data_series: pd.Series, window: int = 14) -> dict:
        """Calculates the Exponential Moving Average (EMA) of a data series.

        Summary
        -------
        Computes the EMA using pandas' ewm implementation with ``span=window``.

        Parameters
        ----------
        data_series : pd.Series
            The input data series (e.g., closing prices).
        window : int
            The period for the EMA calculation.

        Returns
        -------
        dict
            A dictionary with key ``ema_{window}`` containing a pandas Series with the EMA values.
        """

        Indicators._validate_indicator_inputs(data_series=data_series, window=window)

        return {
            f"ema_{window}": data_series.ewm(
                span=window, adjust=False, min_periods=window
            ).mean()
        }

    def wilders_smoothing(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate Wilder's Smoothing for a given data series.

        Summary
        -------
        Implements Wilder's smoothing (an EMA variant) commonly used for RSI/ATR.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series to smooth.
        window : int
            The window size (number of periods) for the smoothing calculation.

        Returns
        -------
        dict
            A dictionary with key ``wilders_smoothing_{window}`` containing a pandas Series with the smoothed values.
            The initial ``window - 1`` values will be NaN.
        """

        Indicators._validate_indicator_inputs(data_series, window)

        return {
            f"wilders_smoothing_{window}": data_series.ewm(
                com=window - 1, adjust=False, min_periods=window
            ).mean()
        }

    def calculate_dema(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate the Double Exponential Moving Average (DEMA) of a data series.

        Summary
        -------
        DEMA reduces lag by using a double-smoothed EMA: ``DEMA = (2 * EMA1) - EMA2``.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        window : int
            The number of periods to use for the DEMA calculation.

        Returns
        -------
        dict
            A dictionary with key ``dema_{window}`` containing a pandas Series with the DEMA values.
            The initial ``2*window - 1`` values will be NaN.

        Notes
        -----
        The formula for DEMA is:
        ``DEMA = (2 * EMA1) - EMA2`` where ``EMA1`` is the EMA of the original series
        and ``EMA2`` is the EMA of ``EMA1``.
        """

        Indicators._validate_indicator_inputs(data_series, window)

        if len(data_series) < (2 * window - 1):
            raise ValueError("Input data series length")

        ema1 = Indicators.calculate_ema(data_series, window)[f"ema_{window}"]
        ema2 = Indicators.calculate_ema(ema1, window)[f"ema_{window}"]

        return {f"dema_{window}": (2 * ema1) - ema2}

    def calculate_tema(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate the Triple Exponential Moving Average (TEMA) of a data series.

        Summary
        -------
        TEMA reduces lag by applying triple EMA smoothing: ``TEMA = 3*EMA1 - 3*EMA2 + EMA3``.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        window : int
            The number of periods to use for the TEMA calculation.

        Returns
        -------
        dict
            A dictionary with key ``tema_{window}`` containing a pandas Series with the TEMA values.
            The initial ``3*window - 2`` values will be NaN.

        Notes
        -----
        The formula for TEMA is: ``TEMA = (3 * EMA1) - (3 * EMA2) + EMA3`` where
        ``EMA1``, ``EMA2``, ``EMA3`` are successive EMAs of the series.
        """

        Indicators._validate_indicator_inputs(data_series, window)

        if len(data_series) < (3 * window - 2):
            raise ValueError("Input data series length")

        ema1_dict = Indicators.calculate_ema(data_series, window)
        ema1_series = ema1_dict[f"ema_{window}"]
        ema2_dict = Indicators.calculate_ema(ema1_series, window)
        ema2_series = ema2_dict[f"ema_{window}"]
        ema3_dict = Indicators.calculate_ema(ema2_series, window)
        ema3_series = ema3_dict[f"ema_{window}"]

        return {f"tema_{window}": (3 * ema1_series) - (3 * ema2_series) + ema3_series}

    def calculate_hma(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate the Hull Moving Average (HMA) of a data series.

        Summary
        -------
        The HMA minimizes lag while keeping smoothness by combining weighted
        averages. This implementation approximates WMA using EMA.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        window : int
            The number of periods to use for the HMA calculation.

        Returns
        -------
        dict
            A dictionary with key ``hma_{window}`` containing a pandas Series with the HMA values.
            The initial values will be NaN due to the nested EMA calculations.

        Raises
        ------
        ValueError
            If the ``window`` is less than 2, as HMA calculation requires at least two periods.

        Notes
        -----
        This implementation uses the following equation to calculate HMA:
        ``HMA(n) = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)`` and approximates WMA via EMA.
        """

        Indicators._validate_indicator_inputs(data_series, window)

        if window < 2:
            raise ValueError("Window for HMA calculation cannot be less than 2")

        half_window_ema = Indicators.calculate_ema(data_series, window // 2)[
            f"ema_{window // 2}"
        ]
        full_window_ema = Indicators.calculate_ema(data_series, window)[f"ema_{window}"]
        hma_series = Indicators.calculate_ema(
            (2 * half_window_ema) - full_window_ema,
            int(math.sqrt(window)),
        )[f"ema_{int(math.sqrt(window))}"]
        return {f"hma_{window}": hma_series}

    # ----------------

    def calculate_rsi(data_series: pd.Series, window: int = 14) -> dict:
        """Calculate the Relative Strength Index (RSI) of a given data series.

        Summary
        -------
        RSI is a momentum oscillator ranging 0-100 used to identify overbought
        or oversold conditions. This implementation uses Wilder's smoothing.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series for which to calculate the RSI.
        window : int, optional
            The number of periods to use for the RSI calculation (default is 14).

        Returns
        -------
        dict
            A dictionary with key ``rsi_{window}`` containing a pandas Series with the RSI values.
            The initial ``window - 1`` values will be NaN.
        """

        Indicators._validate_indicator_inputs(
            data_series=data_series, window=window, min_data_length=2
        )

        delta = data_series.diff()

        gain = delta.copy()
        loss = delta.copy()

        # Set all the NaN values to 0 for proper RSI calc
        gain[gain < 0] = 0
        loss[loss > 0] = 0
        loss = loss.abs()

        # Use Wilder's smoothing
        avg_gain = Indicators.wilders_smoothing(gain, window)[
            f"wilders_smoothing_{window}"
        ]
        avg_loss = Indicators.wilders_smoothing(loss, window)[
            f"wilders_smoothing_{window}"
        ]

        rs = avg_gain / avg_loss

        return {f"rsi_{window}": 100 - (100 / (1 + rs))}

    # -----------------------------
    # DIFFERENT TYPES OF MACD
    def _validate_macd_inputs(
        data_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        min_data_lenght: int = 1,
    ):
        Indicators._validate_indicator_inputs(
            data_series,
            window=max(slow_period, signal_period),
            min_data_length=min_data_lenght,
        )

        if not isinstance(fast_period, int) or fast_period <= 0:
            raise ValueError("fast_period must be a positive integer.")
        if not isinstance(slow_period, int) or slow_period <= 0:
            raise ValueError("slow_period must be a positive integer.")
        if not isinstance(signal_period, int) or signal_period <= 0:
            raise ValueError("signal_period must be a positive integer.")
        if fast_period >= slow_period:
            raise ValueError("fast_period must be less than slow_period.")

    def calculate_macd(
        data_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """Calculates the Moving Average Convergence Divergence (MACD) indicator.

        Summary
        -------
        MACD is a trend-following momentum indicator showing the relationship
        between two EMAs. This function returns the MACD line, its signal line,
        and the histogram.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        fast_period : int, optional
            The period for the fast EMA (default is 12).
        slow_period : int, optional
            The period for the slow EMA (default is 26).
        signal_period : int, optional
            The period for the signal line EMA (default is 9).

        Returns
        -------
        dict
            A dictionary containing three Series: 'macd_line', 'macd_signal', and 'macd_histogram'.
        """

        Indicators._validate_macd_inputs(
            data_series, fast_period, slow_period, signal_period, min_data_lenght=1
        )

        ema_fast = Indicators.calculate_ema(data_series, fast_period)[
            f"ema_{fast_period}"
        ]
        ema_slow = Indicators.calculate_ema(data_series, slow_period)[
            f"ema_{slow_period}"
        ]

        macd_line = ema_fast - ema_slow
        signal_line = Indicators.calculate_ema(macd_line, signal_period)[
            f"ema_{signal_period}"
        ]
        macd_histogram = macd_line - signal_line
        return {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": macd_histogram,
        }

    def calculate_wilders_macd(
        data_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """Calculates the Moving Average Convergence Divergence (MACD) indicator using Wilder's smoothing.

        Summary
        -------
        Variant of MACD where the signal line is smoothed with Wilder's method
        instead of a standard EMA. Returns MACD line, Wilder-smoothed signal,
        and the histogram.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        fast_period : int, optional
            The period for the fast EMA (default is 12).
        slow_period : int, optional
            The period for the slow EMA (default is 26).
        signal_period : int, optional
            The period for the signal line (Wilder's smoothing) (default is 9).

        Returns
        -------
        dict
            A dictionary containing three Series: 'wilders_macd_line', 'wilders_macd_signal', and 'wilders_macd_histogram'.
        """

        Indicators._validate_macd_inputs(
            data_series, fast_period, slow_period, signal_period, min_data_lenght=1
        )

        ema_fast = Indicators.calculate_ema(data_series, fast_period)[
            f"ema_{fast_period}"
        ]
        ema_slow = Indicators.calculate_ema(data_series, slow_period)[
            f"ema_{slow_period}"
        ]

        macd_line = ema_fast - ema_slow
        signal_line = Indicators.wilders_smoothing(macd_line, window=signal_period)[
            f"wilders_smoothing_{signal_period}"
        ]

        macd_histogram = macd_line - signal_line
        return {
            "wilders_macd_line": macd_line,
            "wilders_macd_signal": signal_line,
            "wilders_macd_histogram": macd_histogram,
        }

    def calculate_tema_macd(
        data_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """Calculate the Triple Exponential Moving Average (TEMA) MACD indicator.

        Summary
        -------
        Variant of MACD that smooths the signal line with a TEMA for increased
        responsiveness. Returns the MACD line, TEMA-smoothed signal, and histogram.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        fast_period : int, optional
            The period for the fast EMA used in the MACD line (default 12).
        slow_period : int, optional
            The period for the slow EMA used in the MACD line (default 26).
        signal_period : int, optional
            The period for the TEMA used to smooth the signal line (default 9).

        Returns
        -------
        dict
            A dictionary containing three Series: 'tema_macd_line', 'tema_macd_signal', and 'tema_macd_histogram'.

        Notes
        -----
        Calculation steps:
        1. Calculate fast EMA and slow EMA.
        2. MACD Line = fast EMA - slow EMA.
        3. Signal Line = TEMA(MACD Line, signal_period).
        4. Histogram = MACD Line - Signal Line.
        """

        Indicators._validate_macd_inputs(
            data_series, fast_period, slow_period, signal_period, min_data_lenght=1
        )
        ema_fast = Indicators.calculate_ema(data_series, fast_period)[
            f"ema_{fast_period}"
        ]
        ema_slow = Indicators.calculate_ema(data_series, slow_period)[
            f"ema_{slow_period}"
        ]

        macd_line = ema_fast - ema_slow
        signal_line = Indicators.calculate_tema(macd_line, window=signal_period)[
            f"tema_{signal_period}"
        ]

        macd_histogram = macd_line - signal_line
        return {
            "tema_macd_line": macd_line,
            "tema_macd_signal": signal_line,
            "tema_macd_histogram": macd_histogram,
        }

    def calculate_hma_macd(
        data_series: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
    ) -> dict:
        """Calculate the Hull Moving Average (HMA) MACD indicator.

        Summary
        -------
        Variant of MACD that smooths the signal line using HMA for reduced lag
        and improved responsiveness. Returns MACD line, HMA-smoothed signal, and histogram.

        Parameters
        ----------
        data_series : pandas.Series
            The input data series (e.g., closing prices).
        fast_period : int, optional
            The period for the fast EMA used in the MACD line (default 12).
        slow_period : int, optional
            The period for the slow EMA used in the MACD line (default 26).
        signal_period : int, optional
            The period for the HMA used to smooth the signal line (default 9).

        Returns
        -------
        dict
            A dictionary containing three Series: 'hma_macd_line', 'hma_macd_signal', and 'hma_macd_histogram'.

        Notes
        -----
        Calculation steps:
        1. Calculate fast EMA and slow EMA.
        2. MACD Line = fast EMA - slow EMA.
        3. Signal Line = HMA(MACD Line, signal_period).
        4. Histogram = MACD Line - Signal Line.
        """

        Indicators._validate_macd_inputs(
            data_series, fast_period, slow_period, signal_period, min_data_lenght=1
        )
        ema_fast = Indicators.calculate_ema(data_series, fast_period)[
            f"ema_{fast_period}"
        ]
        ema_slow = Indicators.calculate_ema(data_series, slow_period)[
            f"ema_{slow_period}"
        ]

        macd_line = ema_fast - ema_slow
        signal_line = Indicators.calculate_hma(macd_line, window=signal_period)[
            f"hma_{signal_period}"
        ]

        macd_histogram = macd_line - signal_line
        return {
            "hma_macd_line": macd_line,
            "hma_macd_signal": signal_line,
            "hma_macd_histogram": macd_histogram,
        }

    def get_name_func() -> dict:
        """
        Get a dictionary mapping indicator names to their corresponding calculation functions.

        Returns
        -------
        dict
            A dictionary where keys are indicator names (e.g., 'sma', 'ema', 'rsi') and values are the corresponding
            static methods of the Indicators class that perform the calculations. All lowercased for consistency.
        """
        name_func = {
            "sma": Indicators.calculate_sma,
            "dema": Indicators.calculate_dema,
            "ema": Indicators.calculate_ema,
            "tema": Indicators.calculate_tema,
            "hma": Indicators.calculate_hma,
            "rsi": Indicators.calculate_rsi,
            "wma": Indicators.calculate_wma,
            "macd": Indicators.calculate_macd,
            "wilders_macd": Indicators.calculate_wilders_macd,
            "tema_macd": Indicators.calculate_tema_macd,
            "hma_macd": Indicators.calculate_hma_macd,
            "wilders_smoothing": Indicators.wilders_smoothing,
        }

        return name_func
