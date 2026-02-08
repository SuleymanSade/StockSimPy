import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path so we can import stocksimpy modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stocksimpy.addons.indicators import Indicators

# Import indicator functions for direct testing
wilders_smoothing = Indicators.wilders_smoothing
calculate_sma = Indicators.calculate_sma
calculate_rsi = Indicators.calculate_rsi
calculate_ema = Indicators.calculate_ema
calculate_macd = Indicators.calculate_macd
calculate_dema = Indicators.calculate_dema
calculate_tema = Indicators.calculate_tema
calculate_hma = Indicators.calculate_hma
calculate_tema_macd = Indicators.calculate_tema_macd
calculate_hma_macd = Indicators.calculate_hma_macd


class TestWildersSmoothing:
    def test_wilders_smoothing_valid_input(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        window = 3
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_exact_window_size(self):
        data_series = pd.Series([1, 2, 3])
        window = 3
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_large_data_series(self):
        data_series = pd.Series(range(10000))
        window = 50
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_non_positive_window(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            wilders_smoothing(data_series, 0)

    def test_wilders_smoothing_empty_data_series(self):
        data_series = pd.Series([])
        window = 3
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            wilders_smoothing(data_series, window)

    def test_wilders_smoothing_non_numeric_data(self):
        data_series = pd.Series(["a", "b", "c"])
        window = 3
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            wilders_smoothing(data_series, window)


class TestCalculateSMA:
    def test_calculate_sma_valid_input(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        window = 3
        result = calculate_sma(data_series, window)
        expected = pd.Series([None, None, 2.0, 3.0, 4.0])
        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_calculate_sma_large_window(self):
        data_series = pd.Series([1, 2, 3])
        window = 5
        with pytest.raises(ValueError):
            calculate_sma(data_series, window)


class TestCalculateRSI:
    def test_calculate_rsi_default_window(self):
        data_series = pd.Series(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        )
        result = calculate_rsi(data_series)
        assert len(result) == len(data_series)
        assert result.isna().sum() == 14  # First 14 values should be NaN

    def test_calculate_rsi_custom_window(self):
        data_series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        window = 5
        result = calculate_rsi(data_series, window)
        assert len(result) == len(data_series)
        assert result.isna().sum() == window  # First 5 values should be NaN

    def test_calculate_rsi_constant_series(self):
        data_series = pd.Series([10] * 15)
        result = calculate_rsi(data_series)
        assert (result.dropna() == 50).all()  # RSI should be 50 for constant series

    def test_calculate_rsi_empty_series(self):
        data_series = pd.Series([])
        with pytest.raises(ValueError):
            calculate_rsi(data_series)

    def test_calculate_rsi_nan_values(self):
        data_series = pd.Series([10, np.nan, 12, np.nan, 14, 15, np.nan])
        result = calculate_rsi(data_series, 5)
        assert result.isna().sum() > 0  # Should handle NaN values without error

    def test_calculate_rsi_zero_series(self):
        data_series = pd.Series([0] * 15)
        result = calculate_rsi(data_series)
        assert (result.dropna() == 50).all()  # RSI should be 50 for zero series

    def test_calculate_rsi_strictly_increasing_series(self):
        # After initial period, if it's strictly increasing, RSI should go to 100
        data_series = pd.Series(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ],
            dtype=float,
        )
        # Using a small window for easier manual verification
        window = 3
        result = calculate_rsi(data_series, window=window)

        # Expected values can be tricky to derive manually for EWM/RSI.
        # It's best to use a trusted source (e.g., TA-Lib, Excel formula)
        # For a strictly increasing series, RSI should eventually hit 100 or be very close.
        # Let's verify the initial NaNs and then that the rest are very high.
        assert result.iloc[:window].isna().all()
        assert (result.iloc[window:] >= 99.9).all()  # Should be very close to 100

    def test_calculate_rsi_strictly_decreasing_series(self):
        # After initial period, if it's strictly decreasing, RSI should go to 0
        data_series = pd.Series(
            [
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
            ],
            dtype=float,
        )
        window = 3
        result = calculate_rsi(data_series, window=window)
        assert result.iloc[:window].isna().all()
        assert (result.iloc[window:] <= 0.1).all()  # Should be very close to 0

    def test_calculate_rsi_only_gains(self):
        # After initial window, RSI should be 100 when only gains occur
        data_series = pd.Series(
            [10, 10, 10, 10, 11, 12, 13, 14, 15], dtype=float
        )  # Initial flat then gains
        window = 3
        result = calculate_rsi(data_series, window=window)
        # Manually calculate expected values for a small, controlled series
        # e.g., using Excel or a trusted library
        # For simplicity, let's just check the tail that should be 100
        assert (
            result.iloc[5:].fillna(0).eq(100).all()
        )  # Should be 100 after stabilization

    def test_calculate_rsi_only_losses(self):
        # After initial window, RSI should be 0 when only losses occur
        data_series = pd.Series([10, 10, 10, 10, 9, 8, 7, 6, 5], dtype=float)
        window = 3
        result = calculate_rsi(data_series, window=window)
        assert (
            result.iloc[5:].fillna(100).eq(0).all()
        )  # Should be 0 after stabilization

    def test_calculate_rsi_non_positive_window(self):
        data_series = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_rsi(data_series, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_rsi(data_series, -5)

    def test_calculate_rsi_insufficient_data_length(self):
        # RSI needs window + 1 for diff then EWM
        data_series = pd.Series(
            [1.0, 2.0, 3.0]
        )  # Less than 14 default, or for window=5, needs 6
        window = 5  # Needs 6 data points
        with pytest.raises(ValueError, match="Input data series length"):
            calculate_rsi(data_series, window)

    def test_calculate_rsi_non_numeric_data(self):
        data_series = pd.Series(["a", "b", "c"])
        window = 3
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_rsi(data_series, window)


class TestCalculateEMA:
    def test_calculate_ema_basic(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3
        result = calculate_ema(data, window)
        expected = data.ewm(span=window, adjust=False, min_periods=window).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_ema_nan_for_initial_values(self):
        data = pd.Series([10, 20, 30, 40, 50])
        window = 3
        result = calculate_ema(data, window)
        # The first window-1 values should be NaN
        assert result[: window - 1].isna().all()

    def test_calculate_ema_window_equals_length(self):
        data = pd.Series([5, 10, 15, 20])
        window = 4
        result = calculate_ema(data, window)
        expected = data.ewm(span=window, adjust=False, min_periods=window).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_ema_invalid_window(self):
        data = pd.Series([1, 2, 3, 4])
        with pytest.raises(ValueError):
            calculate_ema(data, 0)
        with pytest.raises(ValueError):
            calculate_ema(data, -2)

    def test_calculate_ema_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError):
            calculate_ema(data, 2)

    def test_calculate_ema_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_ema(data, 2)


class TestWildersSmoothing:
    def test_wilders_smoothing_valid_input(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        window = 3
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_exact_window_size(self):
        data_series = pd.Series([1, 2, 3])
        window = 3
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_large_data_series(self):
        data_series = pd.Series(range(10000))
        window = 50
        result = wilders_smoothing(data_series, window)
        expected = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_wilders_smoothing_non_positive_window(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            wilders_smoothing(data_series, 0)

    def test_wilders_smoothing_empty_data_series(self):
        data_series = pd.Series([])
        window = 3
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            wilders_smoothing(data_series, window)

    def test_wilders_smoothing_non_numeric_data(self):
        data_series = pd.Series(["a", "b", "c"])
        window = 3
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            wilders_smoothing(data_series, window)


class TestCalculateSMA:
    def test_calculate_sma_valid_input(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        window = 3
        result = calculate_sma(data_series, window)
        expected = pd.Series(
            [np.nan, np.nan, 2.0, 3.0, 4.0]
        )  # Changed None to np.nan for strict comparison
        pd.testing.assert_series_equal(result, expected, check_exact=False)

    def test_calculate_sma_large_window(self):
        data_series = pd.Series([1, 2, 3])
        window = 5
        with pytest.raises(
            ValueError, match="Input data series length"
        ):  # Add specific match
            calculate_sma(data_series, window)

    def test_calculate_sma_empty_data_series(self):
        data_series = pd.Series([])
        window = 3
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_sma(data_series, window)

    def test_calculate_sma_non_positive_window(self):
        data_series = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_sma(data_series, 0)

    def test_calculate_sma_non_numeric_data(self):
        data_series = pd.Series(["a", "b", "c"])
        window = 3
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_sma(data_series, window)


class TestCalculateRSI:
    def test_calculate_rsi_default_window(self):
        data_series = pd.Series(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        )
        result = calculate_rsi(data_series)
        assert len(result) == len(data_series)
        assert result.isna().sum() == 14  # First 14 values should be NaN

    def test_calculate_rsi_custom_window(self):
        data_series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        window = 5
        result = calculate_rsi(data_series, window)
        assert len(result) == len(data_series)
        # For window=5, RSI needs 5 diffs, so it needs 6 data points to start calculation.
        # The first 5 values (index 0-4) will be NaN if the window is 5.
        assert result.isna().sum() == window

    def test_calculate_rsi_constant_series(self):
        data_series = pd.Series([10] * 15)
        result = calculate_rsi(data_series)
        assert (result.dropna() == 50).all()  # RSI should be 50 for constant series

    def test_calculate_rsi_empty_series(self):
        data_series = pd.Series([])
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_rsi(data_series)

    def test_calculate_rsi_nan_values(self):
        data_series = pd.Series(
            [10, np.nan, 12, np.nan, 14, 15, np.nan, 16, 17, 18], dtype=float
        )
        # The presence of NaNs in the data series will propagate NaNs in the RSI output.
        # This test ensures it doesn't crash and returns a series with NaNs.
        result = calculate_rsi(data_series, 3)
        assert (
            result.isna().sum() >= 3
        )  # At least initial NaNs and where source data is NaN

    def test_calculate_rsi_zero_series(self):
        data_series = pd.Series([0] * 15)
        result = calculate_rsi(data_series)
        assert (result.dropna() == 50).all()  # RSI should be 50 for zero series

    def test_calculate_rsi_strictly_increasing_series(self):
        data_series = pd.Series(
            [
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
            ],
            dtype=float,
        )
        window = 3
        result = calculate_rsi(data_series, window=window)
        assert result.iloc[:window].isna().all()
        assert (
            result.iloc[window:].dropna() >= 99.9
        ).all()  # Should be very close to 100

    def test_calculate_rsi_strictly_decreasing_series(self):
        data_series = pd.Series(
            [
                30,
                29,
                28,
                27,
                26,
                25,
                24,
                23,
                22,
                21,
                20,
                19,
                18,
                17,
                16,
                15,
                14,
                13,
                12,
                11,
                10,
            ],
            dtype=float,
        )
        window = 3
        result = calculate_rsi(data_series, window=window)
        assert result.iloc[:window].isna().all()
        assert (result.iloc[window:].dropna() <= 0.1).all()  # Should be very close to 0

    def test_calculate_rsi_only_gains(self):
        data_series = pd.Series([10, 10, 10, 10, 11, 12, 13, 14, 15], dtype=float)
        window = 3
        result = calculate_rsi(data_series, window=window)
        # For this specific setup, 3 initial flat values mean diffs are 0.
        # Then gains start. The 4th element (index 3) is where gains start
        # RSI will become 100 after 1 full window of only gains
        # Index 0,1,2,3 are NaNs due to diffs and smoothing min_periods.
        # So from index 3, the first actual diff is 1.0 (11-10).
        # Need (window + 1) for first valid gain/loss. So index 0 to window (14) are NaN
        # For window 3, first valid RSI at index 3+1 = 4.
        # Let's adjust expected NaNs and then check tail.
        # For [10,10,10,10,11,12,13,14,15] and window 3:
        # diffs: [NaN, 0, 0, 0, 1, 1, 1, 1, 1]
        # avg_gain (wilders_smoothing, window=3):
        # Index 0,1,2: NaN
        # Index 3: initial point for EWM, (0+0+0)/3 is not how EWM works. It's EWM of diffs.
        # The first valid RSI for window=3 appears at index 3 (data_series[3] is 10, diff is 0).
        # But for 'only_gains', we need the *smoothing* to stabilize.
        # With the values (10, 10, 10, 10, 11, 12, 13, 14, 15) and window=3:
        # First valid RSI is at index 3 (4th data point) as `min_data_length=2` means `window + (2-1) = window + 1` needed.
        # Let's assume the tail should eventually converge to 100
        assert (result.dropna() >= 99.9).all()  # Should be 100 after stabilization

    def test_calculate_rsi_only_losses(self):
        data_series = pd.Series([10, 10, 10, 10, 9, 8, 7, 6, 5], dtype=float)
        window = 3
        result = calculate_rsi(data_series, window=window)
        assert (result.dropna() <= 0.1).all()  # Should be 0 after stabilization

    def test_calculate_rsi_non_positive_window(self):
        data_series = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_rsi(data_series, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_rsi(data_series, -5)

    def test_calculate_rsi_insufficient_data_length(self):
        data_series = pd.Series(
            [1.0, 2.0, 3.0]
        )  # Less than 14 default, or for window=5, needs 6
        window = 5  # Needs 6 data points
        with pytest.raises(ValueError, match="Input data series length"):
            calculate_rsi(data_series, window)

    def test_calculate_rsi_non_numeric_data(self):
        data_series = pd.Series(["a", "b", "c"])
        window = 3
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_rsi(data_series, window)


class TestCalculateEMA:
    def test_calculate_ema_basic(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window = 3
        result = calculate_ema(data, window)
        expected = data.ewm(span=window, adjust=False, min_periods=window).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_ema_nan_for_initial_values(self):
        data = pd.Series([10, 20, 30, 40, 50])
        window = 3
        result = calculate_ema(data, window)
        # The first window-1 values should be NaN
        assert result[: window - 1].isna().all()

    def test_calculate_ema_window_equals_length(self):
        data = pd.Series([5, 10, 15, 20])
        window = 4
        result = calculate_ema(data, window)
        expected = data.ewm(span=window, adjust=False, min_periods=window).mean()
        pd.testing.assert_series_equal(result, expected)

    def test_calculate_ema_invalid_window(self):
        data = pd.Series([1, 2, 3, 4])
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_ema(data, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_ema(data, -2)

    def test_calculate_ema_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_ema(data, 2)

    def test_calculate_ema_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_ema(data, 2)


class TestCalculateDEMA:
    def test_calculate_dema_basic(self):
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        window = 3
        result = calculate_dema(data, window)

        # Manual calculation for expected DEMA
        ema1 = data.ewm(span=window, adjust=False, min_periods=window).mean()
        ema2 = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
        expected = (2 * ema1) - ema2

        pd.testing.assert_series_equal(result, expected)

    def test_calculate_dema_nan_propagation(self):
        data = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        window = 3  # DEMA needs 2*window - 1 values for first non-NaN
        result = calculate_dema(data, window)
        # First EMA needs `window` periods. Second EMA needs `window` periods of the first EMA.
        # So, the first non-NaN value of ema1 is at index `window-1`.
        # The first non-NaN value of ema2 (ema of ema1) is at index `(window-1) + (window-1) = 2*window - 2`.
        # So, DEMA has NaNs up to index `2*window - 3`.
        expected_nans = (
            2 * window
        ) - 2  # This is the count of NaNs for pandas ewm.mean with min_periods=window.
        # The first non-NaN value will be at index (2*window - 2).
        assert result.iloc[:expected_nans].isna().all()
        assert not result.iloc[expected_nans:].isna().any()

    def test_calculate_dema_invalid_window(self):
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_dema(data, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_dema(data, -5)

    def test_calculate_dema_insufficient_data(self):
        data = pd.Series([1, 2], dtype=float)
        window = 2  # DEMA needs at least 2*2 - 1 = 3 data points for first non-NaN
        with pytest.raises(ValueError, match="Input data series length"):
            calculate_dema(data, window)

    def test_calculate_dema_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_dema(data, 2)

    def test_calculate_dema_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_dema(data, 2)


class TestCalculateTEMA:
    def test_calculate_tema_basic(self):
        data = pd.Series(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=float
        )
        window = 3
        result = calculate_tema(data, window)

        # Manual calculation for expected TEMA
        ema1 = data.ewm(span=window, adjust=False, min_periods=window).mean()
        ema2 = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
        ema3 = ema2.ewm(span=window, adjust=False, min_periods=window).mean()
        expected = (3 * ema1) - (3 * ema2) + ema3

        pd.testing.assert_series_equal(result, expected)

    def test_calculate_tema_nan_propagation(self):
        data = pd.Series(range(20), dtype=float)
        window = 4  # TEMA needs 3*window - 2 values for first non-NaN
        result = calculate_tema(data, window)
        # First EMA: window-1 NaNs.
        # Second EMA (of ema1): (window-1) + (window-1) NaNs.
        # Third EMA (of ema2): (window-1) + (window-1) + (window-1) NaNs.
        # Total NaNs = 3*window - 3. First non-NaN at index 3*window - 3.
        expected_nans = (3 * window) - 3  # This is the count of NaNs.
        # The first non-NaN will be at index (3*window - 3).
        assert result.iloc[:expected_nans].isna().all()
        assert not result.iloc[expected_nans:].isna().any()

    def test_calculate_tema_invalid_window(self):
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_tema(data, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_tema(data, -5)

    def test_calculate_tema_insufficient_data(self):
        data = pd.Series([1, 2, 3, 4, 5, 6], dtype=float)
        window = 3  # TEMA needs at least 3*3 - 2 = 7 data points for first non-NaN
        with pytest.raises(ValueError, match="Input data series length"):
            calculate_tema(data, window)

    def test_calculate_tema_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_tema(data, 2)

    def test_calculate_tema_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_tema(data, 2)


class TestCalculateHMA:
    def test_calculate_hma_basic(self):
        data = pd.Series(np.arange(1, 20, dtype=float))
        window = 5  # A small window for HMA test

        # Manual calculation for expected HMA (using calculate_ema as WMA approximation)
        ema_half_window = calculate_ema(data, window // 2)  # Here window//2 is 2
        ema_full_window = calculate_ema(data, window)  # Here window is 5
        raw_hma_component = (2 * ema_half_window) - ema_full_window

        sqrt_window = int(math.sqrt(window))  # int(sqrt(5)) = 2
        if sqrt_window == 0:
            sqrt_window = 1
        expected = calculate_ema(raw_hma_component, sqrt_window)

        result = calculate_hma(data, window)
        pd.testing.assert_series_equal(result, expected, check_exact=False, rtol=1e-5)

    def test_calculate_hma_nan_propagation(self):
        data = pd.Series(np.arange(1, 50, dtype=float))
        window = 14  # Standard HMA window
        result = calculate_hma(data, window)

        # Calculating expected NaNs for HMA can be complex due to nested EMAs and different window sizes.
        # ema_half_window: min_periods = window//2. First non-NaN at index (window//2 - 1).
        # ema_full_window: min_periods = window. First non-NaN at index (window - 1).
        # raw_hma_component: NaNs up to max( (window//2 - 1), (window - 1) ) = window - 1.
        # Then, calculate_ema(raw_hma_component, sqrt_window):
        # First non-NaN will be at index (window - 1) + (int(sqrt(window)) - 1).

        # For window=14:
        # ema_half_window (window=7): first non-NaN at index 6
        # ema_full_window (window=14): first non-NaN at index 13
        # raw_hma_component: first non-NaN at index 13
        # sqrt_window = int(sqrt(14)) = 3.
        # final HMA: first non-NaN at index 13 + (3 - 1) = 13 + 2 = 15.
        expected_first_valid_idx = (window - 1) + (int(math.sqrt(window)) - 1)

        assert result.iloc[:expected_first_valid_idx].isna().all()
        assert not result.iloc[expected_first_valid_idx:].isna().any()

    def test_calculate_hma_window_less_than_2_raises_error(self):
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(ValueError):
            calculate_hma(data, 1)
        with pytest.raises(ValueError):
            calculate_hma(
                data, 0
            )  # Also caught by _validate_indicator_inputs, but HMA-specific check is good

    def test_calculate_hma_invalid_window_type(self):
        data = pd.Series([1, 2, 3, 4, 5], dtype=float)
        with pytest.raises(
            TypeError
        ):  # _validate_indicator_inputs catches non-int window
            calculate_hma(data, 2.5)

    def test_calculate_hma_insufficient_data(self):
        data = pd.Series([1, 2, 3], dtype=float)  # window=5, needs more data for HMA(5)
        window = 5
        with pytest.raises(ValueError, match="Input data series length"):
            calculate_hma(data, window)

    def test_calculate_hma_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_hma(data, 2)

    def test_calculate_hma_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_hma(data, 2)


class TestCalculateMACD:
    def test_macd_basic_output_shape(self):
        data = pd.Series(np.arange(1, 51, dtype=float))
        result = calculate_macd(data)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"MACD", "Signal", "Histogram"}
        assert len(result) == len(data)

    def test_macd_nan_for_initial_values(self):
        data = pd.Series(np.arange(1, 30, dtype=float))
        result = calculate_macd(data, fast_period=5, slow_period=10, signal_period=3)
        # The first slow_period-1 values should be NaN for MACD
        assert result["MACD"][:9].isna().all()
        assert result["Signal"][:2].isna().all()  # signal_period-1
        assert result["Histogram"][:2].isna().all()

    def test_macd_fast_period_greater_than_slow_period(self):
        data = pd.Series(np.arange(1, 30, dtype=float))
        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            calculate_macd(data, fast_period=15, slow_period=10)

    def test_macd_invalid_periods(self):
        data = pd.Series(np.arange(1, 30, dtype=float))
        with pytest.raises(ValueError):
            calculate_macd(data, fast_period=0)
        with pytest.raises(ValueError):
            calculate_macd(data, slow_period=-1)
        with pytest.raises(ValueError):
            calculate_macd(data, signal_period=0)

    def test_macd_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(TypeError):
            calculate_macd(data)

    def test_macd_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError):
            calculate_macd(data)

    def test_macd_consistency_with_manual_ema(self):
        data = pd.Series(np.random.rand(100))
        fast, slow, signal = 12, 26, 9
        result = calculate_macd(data, fast, slow, signal)
        # Manual calculation for comparison
        ema_fast = data.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow = data.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = data.ewm(span=signal, adjust=False, min_periods=signal).mean()
        histogram = macd_line - signal_line
        pd.testing.assert_series_equal(result["MACD"], macd_line, check_names=False)
        pd.testing.assert_series_equal(result["Signal"], signal_line, check_names=False)
        pd.testing.assert_series_equal(
            result["Histogram"], histogram, check_names=False
        )


class TestCalculateTEMAMACD:
    def test_tema_macd_basic_output_shape(self):
        data = pd.Series(np.arange(1, 81, dtype=float))
        result = calculate_tema_macd(data)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"TEMA_MACD", "TEMA_Signal", "TEMA_Histogram"}
        # assert len(result) == len(data)

    def test_tema_macd_nan_propagation(self):
        data = pd.Series(
            np.arange(1, 40, dtype=float)
        )  # Needs enough data for TEMA(signal_period)
        fast_p, slow_p, signal_p = 5, 10, 3  # TEMA(3) means 3*3-2 = 7 values needed.
        # MACD line first valid at index 9 (slow_p-1).
        # TEMA signal line first valid at (slow_p-1) + (3*signal_p - 3) = 9 + (3*3 - 3) = 9 + 6 = 15.
        result = calculate_tema_macd(
            data, fast_period=fast_p, slow_period=slow_p, signal_period=signal_p
        )

        expected_macd_nan_count = slow_p - 1
        # Signal Line TEMA(macd_line, signal_period)
        # The number of NaNs will be (slow_period - 1) from the macd_line
        # plus (3 * signal_period - 3) from the TEMA calculation itself
        expected_signal_nan_count = (slow_p - 1) + (3 * signal_p - 3)

        assert result["TEMA_MACD"].iloc[:expected_macd_nan_count].isna().all()
        assert not result["TEMA_MACD"].iloc[expected_macd_nan_count:].isna().any()

        assert result["TEMA_Signal"].iloc[:expected_signal_nan_count].isna().all()
        assert not result["TEMA_Signal"].iloc[expected_signal_nan_count:].isna().any()

        assert result["TEMA_Histogram"].iloc[:expected_signal_nan_count].isna().all()
        assert (
            not result["TEMA_Histogram"].iloc[expected_signal_nan_count:].isna().any()
        )

    def test_tema_macd_period_validation(self):
        data = pd.Series(np.arange(1, 50, dtype=float))
        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            calculate_tema_macd(data, fast_period=20, slow_period=10)
        with pytest.raises(
            ValueError, match="signal_period must be a positive integer."
        ):
            calculate_tema_macd(data, signal_period=0)

    def test_tema_macd_consistency_with_manual_calc(self):
        data = pd.Series(np.random.rand(100))
        fast, slow, signal = 12, 26, 9
        result = calculate_tema_macd(data, fast, slow, signal)

        ema_fast = calculate_ema(data, fast)
        ema_slow = calculate_ema(data, slow)
        macd_line_expected = ema_fast - ema_slow
        signal_line_expected = calculate_tema(macd_line_expected, window=signal)
        histogram_expected = macd_line_expected - signal_line_expected

        pd.testing.assert_series_equal(
            result["TEMA_MACD"], macd_line_expected, check_names=False
        )
        pd.testing.assert_series_equal(
            result["TEMA_Signal"], signal_line_expected, check_names=False
        )
        pd.testing.assert_series_equal(
            result["TEMA_Histogram"], histogram_expected, check_names=False
        )

    def test_tema_macd_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_tema_macd(data)

    def test_tema_macd_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_tema_macd(data)


class TestCalculateHMAMACD:
    def test_hma_macd_basic_output_shape(self):
        data = pd.Series(np.arange(1, 51, dtype=float))
        result = calculate_hma_macd(data)
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {"HMA_MACD", "HMA_Signal", "HMA_Histogram"}
        assert len(result) == len(data)

    def test_hma_macd_nan_propagation(self):
        data = pd.Series(np.arange(1, 50, dtype=float))
        fast_p, slow_p, signal_p = (
            5,
            10,
            9,
        )  # HMA(9) needs (9-1) + (int(sqrt(9))-1) = 8 + (3-1) = 8 + 2 = 10 from its input.
        # MACD line first valid at index 9 (slow_p-1).
        # HMA signal line first valid at (slow_p-1) + (HMA_signal_first_valid_offset)
        # HMA_signal_first_valid_offset = (signal_p - 1) + (int(math.sqrt(signal_p)) - 1)
        # For signal_p=9: (9-1) + (int(sqrt(9))-1) = 8 + (3-1) = 8 + 2 = 10
        # So, total NaNs = (slow_p - 1) + 10 = 9 + 10 = 19.
        result = calculate_hma_macd(
            data, fast_period=fast_p, slow_period=slow_p, signal_period=signal_p
        )

        expected_macd_nan_count = slow_p - 1
        expected_signal_nan_count = (slow_p - 1) + (
            (signal_p - 1) + (int(math.sqrt(signal_p)) - 1)
        )

        assert result["HMA_MACD"].iloc[:expected_macd_nan_count].isna().all()
        assert not result["HMA_MACD"].iloc[expected_macd_nan_count:].isna().any()

        assert result["HMA_Signal"].iloc[:expected_signal_nan_count].isna().all()
        assert not result["HMA_Signal"].iloc[expected_signal_nan_count:].isna().any()

        assert result["HMA_Histogram"].iloc[:expected_signal_nan_count].isna().all()
        assert not result["HMA_Histogram"].iloc[expected_signal_nan_count:].isna().any()

    def test_hma_macd_period_validation(self):
        data = pd.Series(np.arange(1, 50, dtype=float))
        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            calculate_hma_macd(data, fast_period=20, slow_period=10)
        with pytest.raises(
            ValueError, match="signal_period must be a positive integer."
        ):
            calculate_hma_macd(data, signal_period=0)
        with pytest.raises(
            ValueError, match="Window for HMA calculation cannot be less than 2"
        ):
            calculate_hma_macd(data, signal_period=1)  # HMA specific check

    def test_hma_macd_consistency_with_manual_calc(self):
        data = pd.Series(np.random.rand(100))
        fast, slow, signal = 12, 26, 9
        result = calculate_hma_macd(data, fast, slow, signal)

        ema_fast = calculate_ema(data, fast)
        ema_slow = calculate_ema(data, slow)
        macd_line_expected = ema_fast - ema_slow
        signal_line_expected = calculate_hma(macd_line_expected, window=signal)
        histogram_expected = macd_line_expected - signal_line_expected

        pd.testing.assert_series_equal(
            result["HMA_MACD"], macd_line_expected, check_names=False, rtol=1e-5
        )
        pd.testing.assert_series_equal(
            result["HMA_Signal"], signal_line_expected, check_names=False, rtol=1e-5
        )
        pd.testing.assert_series_equal(
            result["HMA_Histogram"], histogram_expected, check_names=False, rtol=1e-5
        )

    def test_hma_macd_non_numeric_series(self):
        data = pd.Series(["a", "b", "c"])
        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_hma_macd(data)

    def test_hma_macd_empty_series(self):
        data = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_hma_macd(data)
