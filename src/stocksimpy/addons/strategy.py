# src/stocksimpy/strategy.py

import pandas as pd


class Strategy:
    """
    Collection of ready-to-use example strategies.

    This class provides simple, self-contained trading strategies in both
    fixed-size and dynamic-size formats. Each strategy is independent and
    returned as a callable function that conforms to the Backtester API.

    Fixed strategies return:
        strategy(data) -> signal

    Notes
    -----
    These strategies are designed as reference implementations and examples
    demonstrating how to write custom strategies. They can also be used
    directly in Backtester instances for quick experimentation.
    """

    def buy_all():
        """
        Always-return-buy fixed-size strategy.

        Returns
        -------
        callable
            A function ``strategy(data) -> signal`` that always returns ``'buy'``.

        Notes
        -----
        Used mainly for testing, examples, or verifying the Backtester pipeline.
        """

        def strategy(data: pd.DataFrame) -> str:
            return "buy"

        return strategy

    
    def price_action_dynamic():
        """
        Create a simple price-action dynamic strategy.

        Returns
        -------
        callable
            A function with signature ``strategy(data, holdings) -> (signal, shares)``.

        Strategy Logic
        --------------
        - Compares current price to price 30 days ago.
        - If price has dropped ≥ 14%, buys using a fixed $20,000 allocation.
        - Otherwise returns ``('hold', 0)``.

        Notes
        -----
        - Automatically handles both MultiIndex and flat OHLCV DataFrames.
        - If insufficient data (< 30 points), always returns ``('hold', 0)``.
        """

        def strategy(data: pd.DataFrame, holdings: float) -> tuple:
            # Extract close prices robustly (supports MultiIndex or single-level)
            close_prices = None
            if isinstance(data.columns, pd.MultiIndex):
                if "Close" in data.columns.get_level_values(0):
                    close_prices = data["Close"]
                    if isinstance(close_prices, pd.DataFrame):
                        if close_prices.shape[1] == 1:
                            close_prices = close_prices.squeeze()
                        else:
                            close_prices = close_prices.iloc[:, 0]
                elif "Close" in data.columns.get_level_values(-1):
                    symbol = None
                    if hasattr(data, "symbol"):
                        symbol = data.symbol
                    elif hasattr(data, "name"):
                        symbol = data.name
                    else:
                        symbol = data.columns.get_level_values(0)[0]
                    close_prices = data.xs(symbol, axis=1)["Close"]
            else:
                close_prices = data["Close"]

            if close_prices is None or len(close_prices) < 30:
                return "hold", 0

            current_price = float(close_prices.iloc[-1])
            lookback_price = float(close_prices.iloc[-30])

            pct_change = (current_price - lookback_price) / lookback_price

            if pct_change <= -0.14:
                shares = int(20000 / current_price)
                return "buy", shares

            return "hold", 0

        return strategy

    def rsi_momentum(rsi_period: int = 14):
        """
        RSI momentum crossover fixed-size strategy.

        Parameters
        ----------
        rsi_period : int, default 14
            Period used in RSI computation.

        Returns
        -------
        callable
            Function ``strategy(data) -> signal``.

        Strategy Logic
        --------------
        - Computes RSI.
        - A buy signal occurs when RSI crosses above 50.
        - A sell signal occurs when RSI crosses below 50.
        - Otherwise returns ``'hold'``.

        Notes
        -----
        Requires at least ``rsi_period + 1`` data points.
        """

        def strategy(data: pd.DataFrame) -> str:
            # Extract close prices (MultiIndex or single-level)
            if isinstance(data.columns, pd.MultiIndex):
                close_prices = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
            else:
                close_prices = data["Close"]

            if len(close_prices) < rsi_period + 1:
                return "hold"

            delta = close_prices.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            avg_gain = gains.rolling(window=rsi_period).mean()
            avg_loss = losses.rolling(window=rsi_period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            current = rsi.iloc[-1]
            previous = rsi.iloc[-2]

            if previous < 50 and current > 50:
                return "buy"
            if previous > 50 and current < 50:
                return "sell"

            return "hold"

        return strategy

    def sma_ema_crossover(fast: int = 12, slow: int = 26):
        """
        SMA crossover strategy.

        Parameters
        ----------
        fast : int
            Fast moving average window.
        slow : int
            Slow moving average window.

        Returns
        -------
        callable
            Function ``strategy(data) -> signal``.

        Strategy Logic
        --------------
        - Computes two simple moving averages.
        - Buy when fast SMA crosses above slow SMA.
        - Sell when fast SMA crosses below slow SMA.
        - Otherwise hold.

        Notes
        -----
        Requires at least ``slow + 2`` data points.
        """

        def strategy(data: pd.DataFrame) -> str:
            # Extract close prices
            if isinstance(data.columns, pd.MultiIndex):
                close = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
            else:
                close = data["Close"]

            if len(close) < slow + 2:
                return "hold"

            sma_fast = close.rolling(fast).mean()
            sma_slow = close.rolling(slow).mean()

            prev_fast, prev_slow = sma_fast.iloc[-2], sma_slow.iloc[-2]
            curr_fast, curr_slow = sma_fast.iloc[-1], sma_slow.iloc[-1]

            if prev_fast <= prev_slow and curr_fast > curr_slow:
                return "buy"
            if prev_fast >= prev_slow and curr_fast < curr_slow:
                return "sell"

            return "hold"

        return strategy

    def rsi_reversion(
        rsi_period: int = 14, low_th: float = 30, high_th: float = 70
    ):
        """
        RSI mean-reversion strategy.
        Parameters
        ----------
        rsi_period : int
            RSI period length.
        low_th : float
            Oversold threshold.
        high_th : float
            Overbought threshold.

        Returns
        -------
        callable
            Function ``strategy(data) -> signal``.

        Strategy Logic
        --------------
        - Buy when RSI < low_th (oversold).
        - Sell when RSI > high_th (overbought).
        - Otherwise hold.
        """

        def strategy(data: pd.DataFrame) -> str:
            if isinstance(data.columns, pd.MultiIndex):
                close = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
            else:
                close = data["Close"]

            if len(close) < rsi_period + 1:
                return "hold"

            delta = close.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)

            avg_gain = gains.rolling(rsi_period).mean()
            avg_loss = losses.rolling(rsi_period).mean()

            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            current = rsi.iloc[-1]

            if current < low_th:
                return "buy"
            if current > high_th:
                return "sell"

            return "hold"

        return strategy

    def multi_indicator(rsi_period: int = 14, sma_long: int = 200):
        """
        Multi-indicator confirmation strategy.

        Parameters
        ----------
        rsi_period : int
            Period for RSI calculation.
        sma_long : int
            Long-term SMA window for trend confirmation.

        Returns
        -------
        callable
            Function ``strategy(data) -> signal``.

        Strategy Logic
        --------------
        Buy when:
            - RSI < 60
            - Price > SMA
        Sell when:
            - RSI > 70
            - OR Price < SMA
        """

        def strategy(data: pd.DataFrame) -> str:
            # Extract
            if isinstance(data.columns, pd.MultiIndex):
                close = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
            else:
                close = data["Close"]

            if len(close) < max(rsi_period, sma_long) + 2:
                return "hold"

            # RSI
            delta = close.diff()
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            avg_gain = gains.rolling(rsi_period).mean()
            avg_loss = losses.rolling(rsi_period).mean()
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))

            # SMA
            sma = close.rolling(sma_long).mean()

            curr_rsi = rsi.iloc[-1]
            curr_price = close.iloc[-1]
            curr_sma = sma.iloc[-1]

            if curr_rsi < 60 and curr_price > curr_sma:
                return "buy"
            if curr_rsi > 70 or curr_price < curr_sma:
                return "sell"

            return "hold"

        return strategy

    
    def breakout_dynamic(lookback: int = 20, allocation: float = 20000):
        """
        Breakout dynamic strategy.

        Parameters
        ----------
        lookback : int
            Window size used to compute breakout high/low.
        allocation : float
            Dollar amount to convert into shares on a breakout.

        Returns
        -------
        callable
            Function ``strategy(data, holdings) -> (signal, shares)``.

        Strategy Logic
        --------------
        - Buy when price breaks above the highest high in the lookback window.
        - Sell when price breaks below the lowest low.
        - Otherwise hold.

        Notes
        -----
        Shares are computed as ``int(allocation / price)``.
        """

        def strategy(data: pd.DataFrame, holdings: float) -> tuple:
            if isinstance(data.columns, pd.MultiIndex):
                close = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
            else:
                close = data["Close"]

            if len(close) < lookback + 1:
                return "hold", 0

            price = float(close.iloc[-1])
            high = float(close.iloc[-lookback:].max())
            low = float(close.iloc[-lookback:].min())

            if price > high:
                shares = int(allocation / price)
                return "buy", shares

            if price < low:
                return "sell", 0

            return "hold", 0

        return strategy

    def atr_trend_dynamic(
        ma_period: int = 20,
        atr_period: int = 14,
        k: float = 1.5,
        allocation: float = 20000,
    ):
        """
        ATR-based dynamic trend-following strategy.

        Parameters
        ----------
        ma_period : int
            Moving average period.
        atr_period : int
            ATR smoothing period.
        k : float
            Volatility multiplier.
        allocation : float
            Dollar amount converted to shares when buying.

        Returns
        -------
        callable
            Function ``strategy(data, holdings) -> (signal, shares)``.

        Strategy Logic
        --------------
        - Computes ATR and SMA bands:
            upper = SMA + k * ATR
            lower = SMA - k * ATR
        - Buy when price exceeds upper band.
        - Sell when price falls below lower band.
        - Otherwise hold.

        Notes
        -----
        Requires ``max(ma_period, atr_period) + 2`` data points.
        """

        def strategy(data: pd.DataFrame, holdings: float) -> tuple:
            if isinstance(data.columns, pd.MultiIndex):
                close = data.iloc[
                    :, data.columns.get_level_values(0) == "Close"
                ].squeeze()
                high = data.iloc[
                    :, data.columns.get_level_values(0) == "High"
                ].squeeze()
                low = data.iloc[:, data.columns.get_level_values(0) == "Low"].squeeze()
            else:
                close = data["Close"]
                high = data["High"]
                low = data["Low"]

            needed = max(ma_period, atr_period) + 2
            if len(close) < needed:
                return "hold", 0

            prev_close = close.shift(1)
            tr = pd.concat(
                [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1,
            ).max(axis=1)

            atr = tr.rolling(atr_period).mean()
            sma = close.rolling(ma_period).mean()

            price = float(close.iloc[-1])
            upper = float((sma + k * atr).iloc[-1])
            lower = float((sma - k * atr).iloc[-1])

            if price > upper:
                shares = int(allocation / price)
                return "buy", shares

            if price < lower:
                return "sell", 0

            return "hold", 0

        return strategy
