from __future__ import annotations

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
calculate_wilders_macd = Indicators.calculate_wilders_macd


class TestWildersSmoothing:
    def test_wilders_smoothing_valid_input(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3, 4, 5])
        window: int = 3
        result: dict[str, pd.Series] = wilders_smoothing(data_series, window)
        result_series: pd.Series = result[f"wilders_smoothing_{window}"]
        expected: pd.Series = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()

        pd.testing.assert_series_equal(result_series, expected)

    def test_wilders_smoothing_exact_window_size(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3])
        window: int = 3
        result: dict[str, pd.Series] = wilders_smoothing(data_series, window)
        result_series: pd.Series = result[f"wilders_smoothing_{window}"]
        expected: pd.Series = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()

        pd.testing.assert_series_equal(result_series, expected)

    def test_wilders_smoothing_large_data_series(self) -> None:
        data_series: pd.Series = pd.Series(range(10000))
        window: int = 50
        result: dict[str, pd.Series] = wilders_smoothing(data_series, window)
        result_series: pd.Series = result[f"wilders_smoothing_{window}"]
        expected: pd.Series = data_series.ewm(
            com=window - 1, adjust=False, min_periods=window
        ).mean()

        pd.testing.assert_series_equal(result_series, expected)

    def test_wilders_smoothing_non_positive_window(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Window must be a positive integer."):
            wilders_smoothing(data_series, 0)

    def test_wilders_smoothing_empty_data_series(self) -> None:
        data_series: pd.Series = pd.Series([])
        window: int = 3

        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            wilders_smoothing(data_series, window)

    def test_wilders_smoothing_non_numeric_data(self) -> None:
        data_series: pd.Series = pd.Series(["a", "b", "c"])
        window: int = 3

        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            wilders_smoothing(data_series, window)


class TestCalculateSMA:
    def test_calculate_sma_valid_input(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3, 4, 5])
        window: int = 3
        result: dict[str, pd.Series] = calculate_sma(data_series, window)
        result_series: pd.Series = result[f"sma_{window}"]
        expected: pd.Series = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])

        pd.testing.assert_series_equal(result_series, expected, check_exact=False)

    def test_calculate_sma_large_window(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3])
        window: int = 5

        with pytest.raises(ValueError, match="Input data series length"):
            calculate_sma(data_series, window)

    def test_calculate_sma_empty_data_series(self) -> None:
        data_series: pd.Series = pd.Series([])
        window: int = 3

        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_sma(data_series, window)

    def test_calculate_sma_non_positive_window(self) -> None:
        data_series: pd.Series = pd.Series([1, 2, 3, 4, 5])

        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_sma(data_series, 0)

    def test_calculate_sma_non_numeric_data(self) -> None:
        data_series: pd.Series = pd.Series(["a", "b", "c"])
        window: int = 3

        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_sma(data_series, window)


class TestCalculateRSI:
    def test_calculate_rsi_default_window(self) -> None:
        data_series: pd.Series = pd.Series(
            [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
        )
        result: dict[str, pd.Series] = calculate_rsi(data_series)
        result_series: pd.Series = result["rsi_14"]

        assert len(result_series) == len(data_series)
        assert result_series.isna().sum() == 14

    def test_calculate_rsi_custom_window(self) -> None:
        data_series: pd.Series = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
        window: int = 5
        result: dict[str, pd.Series] = calculate_rsi(data_series, window)
        result_series: pd.Series = result[f"rsi_{window}"]

        assert len(result_series) == len(data_series)
        assert result_series.isna().sum() == window

    def test_calculate_rsi_constant_series(self) -> None:
        data_series: pd.Series = pd.Series([10] * 15)
        result: dict[str, pd.Series] = calculate_rsi(data_series)
        result_series: pd.Series = result["rsi_14"]

        assert (result_series.dropna() == 50).all()

    def test_calculate_rsi_empty_series(self) -> None:
        data_series: pd.Series = pd.Series([])

        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_rsi(data_series)

    def test_calculate_rsi_non_numeric_data(self) -> None:
        data_series: pd.Series = pd.Series(["a", "b", "c"])
        window: int = 3

        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_rsi(data_series, window)


class TestCalculateEMA:
    def test_calculate_ema_basic(self) -> None:
        data: pd.Series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        window: int = 3
        result: dict[str, pd.Series] = calculate_ema(data, window)
        result_series: pd.Series = result[f"ema_{window}"]
        expected: pd.Series = data.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()

        pd.testing.assert_series_equal(result_series, expected)

    def test_calculate_ema_nan_for_initial_values(self) -> None:
        data: pd.Series = pd.Series([10, 20, 30, 40, 50])
        window: int = 3
        result: dict[str, pd.Series] = calculate_ema(data, window)
        result_series: pd.Series = result[f"ema_{window}"]

        assert result_series[: window - 1].isna().all()

    def test_calculate_ema_window_equals_length(self) -> None:
        data: pd.Series = pd.Series([5, 10, 15, 20])
        window: int = 4
        result: dict[str, pd.Series] = calculate_ema(data, window)
        result_series: pd.Series = result[f"ema_{window}"]
        expected: pd.Series = data.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()

        pd.testing.assert_series_equal(result_series, expected)

    def test_calculate_ema_invalid_window(self) -> None:
        data: pd.Series = pd.Series([1, 2, 3, 4])

        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_ema(data, 0)
        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_ema(data, -2)

    def test_calculate_ema_non_numeric_series(self) -> None:
        data: pd.Series = pd.Series(["a", "b", "c"])

        with pytest.raises(
            TypeError, match="Input 'data_series' must be a numerical pandas Series"
        ):
            calculate_ema(data, 2)

    def test_calculate_ema_empty_series(self) -> None:
        data: pd.Series = pd.Series([], dtype=float)

        with pytest.raises(ValueError, match="Input 'data_series' cannot be empty."):
            calculate_ema(data, 2)


class TestCalculateDEMA:
    def test_calculate_dema_basic(self) -> None:
        data: pd.Series = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        window: int = 3
        result: dict[str, pd.Series] = calculate_dema(data, window)
        result_series: pd.Series = result[f"dema_{window}"]

        ema1: pd.Series = data.ewm(span=window, adjust=False, min_periods=window).mean()
        ema2: pd.Series = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
        expected: pd.Series = (2 * ema1) - ema2

        pd.testing.assert_series_equal(result_series, expected)

    def test_calculate_dema_nan_propagation(self) -> None:
        data: pd.Series = pd.Series([10, 20, 30, 40, 50, 60, 70, 80], dtype=float)
        window: int = 3
        result: dict[str, pd.Series] = calculate_dema(data, window)
        result_series: pd.Series = result[f"dema_{window}"]
        expected_nans: int = (2 * window) - 2

        assert result_series.iloc[:expected_nans].isna().all()
        assert not result_series.iloc[expected_nans:].isna().any()

    def test_calculate_dema_invalid_window(self) -> None:
        data: pd.Series = pd.Series([1, 2, 3, 4, 5], dtype=float)

        with pytest.raises(ValueError, match="Window must be a positive integer."):
            calculate_dema(data, 0)


class TestCalculateTEMA:
    def test_calculate_tema_basic(self) -> None:
        data: pd.Series = pd.Series(
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], dtype=float
        )
        window: int = 3
        result: dict[str, pd.Series] = calculate_tema(data, window)
        result_series: pd.Series = result[f"tema_{window}"]

        ema1: pd.Series = data.ewm(span=window, adjust=False, min_periods=window).mean()
        ema2: pd.Series = ema1.ewm(span=window, adjust=False, min_periods=window).mean()
        ema3: pd.Series = ema2.ewm(span=window, adjust=False, min_periods=window).mean()
        expected: pd.Series = (3 * ema1) - (3 * ema2) + ema3

        pd.testing.assert_series_equal(result_series, expected)

    def test_calculate_tema_nan_propagation(self) -> None:
        data: pd.Series = pd.Series(range(20), dtype=float)
        window: int = 4
        result: dict[str, pd.Series] = calculate_tema(data, window)
        result_series: pd.Series = result[f"tema_{window}"]
        expected_nans: int = (3 * window) - 3

        assert result_series.iloc[:expected_nans].isna().all()
        assert not result_series.iloc[expected_nans:].isna().any()


class TestCalculateHMA:
    def test_calculate_hma_basic(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 20, dtype=float))
        window: int = 5

        ema_half_window: pd.Series = calculate_ema(data, window // 2)[
            f"ema_{window // 2}"
        ]
        ema_full_window: pd.Series = calculate_ema(data, window)[f"ema_{window}"]
        raw_hma_component: pd.Series = (2 * ema_half_window) - ema_full_window

        sqrt_window: int = int(math.sqrt(window))
        if sqrt_window == 0:
            sqrt_window = 1
        expected: pd.Series = calculate_ema(raw_hma_component, sqrt_window)[
            f"ema_{sqrt_window}"
        ]

        result: dict[str, pd.Series] = calculate_hma(data, window)
        result_series: pd.Series = result[f"hma_{window}"]

        pd.testing.assert_series_equal(
            result_series, expected, check_exact=False, rtol=1e-5
        )

    def test_calculate_hma_window_less_than_2_raises_error(self) -> None:
        data: pd.Series = pd.Series([1, 2, 3, 4, 5], dtype=float)

        with pytest.raises(ValueError):
            calculate_hma(data, 1)


class TestCalculateMACD:
    def test_macd_basic_output_shape(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 51, dtype=float))
        result: dict[str, pd.Series] = calculate_macd(data)

        assert isinstance(result, dict)
        assert set(result.keys()) == {"macd_line", "macd_signal", "macd_histogram"}
        assert len(result["macd_line"]) == len(data)

    def test_macd_nan_for_initial_values(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 30, dtype=float))
        result: dict[str, pd.Series] = calculate_macd(
            data, fast_period=5, slow_period=10, signal_period=3
        )

        assert result["macd_line"][:9].isna().all()
        assert result["macd_signal"][:2].isna().all()
        assert result["macd_histogram"][:2].isna().all()

    def test_macd_fast_period_greater_than_slow_period(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 30, dtype=float))

        with pytest.raises(
            ValueError, match="fast_period must be less than slow_period"
        ):
            calculate_macd(data, fast_period=15, slow_period=10)

    def test_macd_consistency_with_manual_ema(self) -> None:
        data: pd.Series = pd.Series(np.random.rand(100))
        fast: int = 12
        slow: int = 26
        signal: int = 9
        result: dict[str, pd.Series] = calculate_macd(data, fast, slow, signal)

        ema_fast: pd.Series = data.ewm(span=fast, adjust=False, min_periods=fast).mean()
        ema_slow: pd.Series = data.ewm(span=slow, adjust=False, min_periods=slow).mean()
        macd_line: pd.Series = ema_fast - ema_slow
        signal_line: pd.Series = macd_line.ewm(
            span=signal, adjust=False, min_periods=signal
        ).mean()
        histogram: pd.Series = macd_line - signal_line

        pd.testing.assert_series_equal(
            result["macd_line"], macd_line, check_names=False
        )
        pd.testing.assert_series_equal(
            result["macd_signal"], signal_line, check_names=False
        )
        pd.testing.assert_series_equal(
            result["macd_histogram"], histogram, check_names=False
        )


class TestCalculateTEMAMACD:
    def test_tema_macd_basic_output_shape(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 81, dtype=float))
        result: dict[str, pd.Series] = calculate_tema_macd(data)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "tema_macd_line",
            "tema_macd_signal",
            "tema_macd_histogram",
        }

    def test_tema_macd_consistency_with_manual_calc(self) -> None:
        data: pd.Series = pd.Series(np.random.rand(100))
        fast: int = 12
        slow: int = 26
        signal: int = 9
        result: dict[str, pd.Series] = calculate_tema_macd(data, fast, slow, signal)

        ema_fast: pd.Series = calculate_ema(data, fast)[f"ema_{fast}"]
        ema_slow: pd.Series = calculate_ema(data, slow)[f"ema_{slow}"]
        macd_line_expected: pd.Series = ema_fast - ema_slow
        signal_line_expected: pd.Series = calculate_tema(
            macd_line_expected, window=signal
        )[f"tema_{signal}"]
        histogram_expected: pd.Series = macd_line_expected - signal_line_expected

        pd.testing.assert_series_equal(
            result["tema_macd_line"], macd_line_expected, check_names=False
        )
        pd.testing.assert_series_equal(
            result["tema_macd_signal"], signal_line_expected, check_names=False
        )
        pd.testing.assert_series_equal(
            result["tema_macd_histogram"], histogram_expected, check_names=False
        )


class TestCalculateHMAMACD:
    def test_hma_macd_basic_output_shape(self) -> None:
        data: pd.Series = pd.Series(np.arange(1, 51, dtype=float))
        result: dict[str, pd.Series] = calculate_hma_macd(data)

        assert isinstance(result, dict)
        assert set(result.keys()) == {
            "hma_macd_line",
            "hma_macd_signal",
            "hma_macd_histogram",
        }
        assert len(result["hma_macd_line"]) == len(data)

    def test_hma_macd_consistency_with_manual_calc(self) -> None:
        data: pd.Series = pd.Series(np.random.rand(100))
        fast: int = 12
        slow: int = 26
        signal: int = 9
        result: dict[str, pd.Series] = calculate_hma_macd(data, fast, slow, signal)

        ema_fast: pd.Series = calculate_ema(data, fast)[f"ema_{fast}"]
        ema_slow: pd.Series = calculate_ema(data, slow)[f"ema_{slow}"]
        macd_line_expected: pd.Series = ema_fast - ema_slow
        signal_line_expected: pd.Series = calculate_hma(
            macd_line_expected, window=signal
        )[f"hma_{signal}"]
        histogram_expected: pd.Series = macd_line_expected - signal_line_expected

        pd.testing.assert_series_equal(
            result["hma_macd_line"], macd_line_expected, check_names=False, rtol=1e-5
        )
        pd.testing.assert_series_equal(
            result["hma_macd_signal"],
            signal_line_expected,
            check_names=False,
            rtol=1e-5,
        )
        pd.testing.assert_series_equal(
            result["hma_macd_histogram"],
            histogram_expected,
            check_names=False,
            rtol=1e-5,
        )
