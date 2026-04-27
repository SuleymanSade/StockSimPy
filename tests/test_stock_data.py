import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path so we can import stocksimpy modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stocksimpy.core.stock_data import StockData


@pytest.fixture
def sample_valid_df():
    """Create a valid sample DataFrame for testing."""
    # Create dates without timezone information and frequency
    dates = pd.DatetimeIndex(
        ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
        name="Date",
    )
    data = {
        "Open": [100, 101, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, 104, 105, 106],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_invalid_df():
    """Create an invalid sample DataFrame for testing."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = {
        "Open": [100, 101, 102, 103, 104],
        "High": [95, 96, 97, 98, 99],  # High less than Low (invalid)
        "Low": [105, 106, 107, 108, 109],
        "Close": [102, 103, 104, 105, 106],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    return df


def test_init_with_valid_data(sample_valid_df):
    """Test initialization with valid data."""
    stock_data = StockData(sample_valid_df)
    assert isinstance(stock_data.df, pd.DataFrame)
    assert stock_data.df.index.equals(sample_valid_df.index)


def test_init_with_invalid_data():
    """Test initialization with negative volume values."""
    df = pd.DataFrame(
        {"Open": [100], "High": [105], "Low": [95], "Close": [102], "Volume": [-1000]},
        index=[pd.Timestamp("2023-01-01")],
    )

    with pytest.raises(ValueError, match="Volume data contains negative values"):
        StockData(df)


def test_validate_missing_columns():
    """Test validation with missing required columns."""
    df = pd.DataFrame(
        {"Open": [100], "Close": [102], "Volume": [1000]},
        index=[pd.Timestamp("2023-01-01")],
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        StockData(df)


def test_validate_negative_volume():
    """Test validation with negative volume values."""
    df = pd.DataFrame(
        {"Open": [100], "High": [105], "Low": [95], "Close": [102], "Volume": [-1000]},
        index=[pd.Timestamp("2023-01-01")],
    )

    with pytest.raises(ValueError, match="Volume data contains negative values"):
        StockData(df)


def test_clean_date_column():
    """Test cleaning of date column."""
    data = {
        "Date": ["2023-01-01", "2023-01-02"],
        "Open": [100, 101],
        "High": [105, 106],
        "Low": [95, 96],
        "Close": [102, 103],
        "Volume": [1000, 1100],
    }
    df = pd.DataFrame(data)
    stock_data = StockData(df)
    assert isinstance(stock_data.df.index, pd.DatetimeIndex)


def test_generate_mock_data():
    """Test mock data generation."""
    stock_data = StockData.generate_mock_data(days=10, seed=42)
    assert len(stock_data.df) == 10
    assert all(
        col in stock_data.df.columns
        for col in ["Open", "High", "Low", "Close", "Volume"]
    )

    # Verify OHLC data consistency
    df = stock_data.df
    assert (df["Low"] <= df["High"]).all(), "Low should be less than or equal to High"
    assert (df["Low"] <= df["Open"]).all(), "Low should be less than or equal to Open"
    assert (df["Low"] <= df["Close"]).all(), "Low should be less than or equal to Close"
    assert (
        df["High"] >= df["Open"]
    ).all(), "High should be greater than or equal to Open"
    assert (
        df["High"] >= df["Close"]
    ).all(), "High should be greater than or equal to Close"


def test_csv_export_import(sample_valid_df, tmp_path):
    """Test CSV export and import functionality."""
    file_path = tmp_path / "test_stock_data.csv"

    # Create and export data
    stock_data = StockData(sample_valid_df)
    stock_data.to_csv(file_path)  # The index will be saved as 'Date' column

    # Import and verify
    imported_data = StockData.from_csv(file_path)

    # Compare the values and dates
    assert (
        stock_data.df.index == imported_data.df.index
    ).all(), "Dates should be identical"
    assert (
        stock_data.df["Open"] == imported_data.df["Open"]
    ).all(), "Open prices should be identical"
    assert (
        stock_data.df["High"] == imported_data.df["High"]
    ).all(), "High prices should be identical"
    assert (
        stock_data.df["Low"] == imported_data.df["Low"]
    ).all(), "Low prices should be identical"
    assert (
        stock_data.df["Close"] == imported_data.df["Close"]
    ).all(), "Close prices should be identical"
    assert (
        stock_data.df["Volume"] == imported_data.df["Volume"]
    ).all(), "Volume should be identical"


def test_slice_data(sample_valid_df):
    """Test data slicing functionality."""
    stock_data = StockData(sample_valid_df)
    start_date = sample_valid_df.index[1]
    end_date = sample_valid_df.index[3]

    sliced_data = stock_data.slice(start_date, end_date)
    assert len(sliced_data) == 3
    assert sliced_data.index[0] == start_date
    assert sliced_data.index[-1] == end_date


def test_get_column(sample_valid_df):
    """Test getting specific column data."""
    stock_data = StockData(sample_valid_df)
    close_prices = stock_data.get("Close")
    pd.testing.assert_series_equal(close_prices, sample_valid_df["Close"])


def test_fill_missing():
    """Test filling missing values."""
    dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
    data = {
        "Open": [100, np.nan, 102, 103, 104],
        "High": [105, 106, 107, 108, 109],
        "Low": [95, 96, 97, 98, 99],
        "Close": [102, 103, np.nan, 105, 106],
        "Volume": [1000, 1100, 1200, 1300, 1400],
    }
    df = pd.DataFrame(data, index=dates)
    stock_data = StockData(df)
    stock_data.fill_missing()

    assert not stock_data.df["Open"].isna().any()
    assert not stock_data.df["Close"].isna().any()


def test_auto_loader_dataframe(sample_valid_df):
    """Test auto_loader with DataFrame input."""
    stock_data = StockData.auto_loader(sample_valid_df)
    assert isinstance(stock_data, StockData)
    # The DataFrame will have MultiIndex columns after processing
    # so we compare the underlying data, not the exact object
    assert stock_data.df.index.equals(sample_valid_df.index)
    assert set(col[0] for col in stock_data.df.columns) == set(sample_valid_df.columns)


def test_auto_loader_dict(sample_valid_df):
    """Test auto_loader with dictionary input."""
    data_dict = {
        "Date": sample_valid_df.index.strftime("%Y-%m-%d").tolist(),
        "Open": sample_valid_df["Open"].tolist(),
        "High": sample_valid_df["High"].tolist(),
        "Low": sample_valid_df["Low"].tolist(),
        "Close": sample_valid_df["Close"].tolist(),
        "Volume": sample_valid_df["Volume"].tolist(),
    }
    stock_data = StockData.auto_loader(data_dict)
    assert isinstance(stock_data, StockData)
    assert len(stock_data.df) == len(sample_valid_df)


def test_auto_loader_invalid_input():
    """Test auto_loader with invalid input."""
    with pytest.raises(TypeError, match="Unsupported source type"):
        StockData.auto_loader(123)


def test_to_dict(sample_valid_df):
    """Test conversion to dictionary."""
    stock_data = StockData(sample_valid_df)
    data_dict = stock_data.to_dict()
    assert isinstance(data_dict, list)
    assert len(data_dict) == len(sample_valid_df)


def test_check_missing(sample_valid_df):
    """Test checking for missing values."""
    stock_data = StockData(sample_valid_df)
    missing_count = stock_data.check_missing()
    assert all(count == 0 for count in missing_count)


def test_add_single_indicator(sample_valid_df):
    """Test addition of a single indicator when an indicator column exists.

    The updated `add_indicator` requires the target indicator column to exist
    in the DataFrame. When `overwrite=True` the existing column is replaced
    with the calculated values.
    """

    def sma(series, window):
        return {f"sma_{window}": series.rolling(window).mean()}

    stock_data = StockData(sample_valid_df)

    # Populate it using add_indicator with overwrite=True
    stock_data.add_indicator(sma, 3, overwrite=True)

    expected = stock_data.df[("Close", "")].rolling(3).mean()

    pd.testing.assert_series_equal(
        stock_data.df[("sma_3", "")], expected, check_names=False
    )


def test_add_multiple_indicators(sample_valid_df):
    """Test adding multiple indicators when columns exist.

    `add_indicator` will only write into pre-existing indicator columns and
    requires `overwrite=True` to replace them.
    """

    def bands(series, window):
        sma = series.rolling(window).mean()
        std = series.rolling(window).std()
        return {
            f"bb_upper_{window}": sma + 2 * std,
            f"bb_lower_{window}": sma - 2 * std,
        }

    stock_data = StockData(sample_valid_df)

    stock_data.add_indicator(bands, 2, base_col="Close", overwrite=True)

    assert ("bb_upper_2", "") in stock_data.df.columns
    assert ("bb_lower_2", "") in stock_data.df.columns


def test_add_indicator_symbol_missing_raises(sample_valid_df):
    """Specifying a non-existent symbol should raise KeyError."""

    def sma(series, window):
        return {f"sma_{window}": series.rolling(window).mean()}

    stock_data = StockData(sample_valid_df)

    with pytest.raises(KeyError, match="does NOT exist"):
        stock_data.add_indicator(sma, 3, base_col="Close", symbol="AAPL")


def test_add_indicator_non_dict_return_raises(sample_valid_df):
    """Indicator function must return a dict; otherwise TypeError is raised."""

    def bad(series, window):
        # returns a Series instead of dict
        return series.rolling(window).mean()

    stock_data = StockData(sample_valid_df)

    with pytest.raises(TypeError):
        stock_data.add_indicator(bad, 2, overwrite=True)


def test_add_indicator_invalid_indicator_type_raises(sample_valid_df):
    """Passing an invalid `indicator` type raises TypeError."""

    stock_data = StockData(sample_valid_df)

    with pytest.raises(TypeError, match="type can only be str or callable"):
        stock_data.add_indicator(123, 3, base_col="Close")


def test_add_indicator_base_sma(sample_valid_df):
    stock_data = StockData(sample_valid_df)

    stock_data.add_indicator("sma", 3)

    assert "sma_3" in stock_data.df

def test_all_indicator():
    stock_data = StockData().generate_mock_data(50)

    stock_data.add_indicator_all("")

    assert "tema_14" in stock_data.df

def test_all_indicator_invalid_key():
    stock_data = StockData().generate_mock_data(50)

    with pytest.raises(KeyError):
        stock_data.add_indicator_all("null")


def test_add_indicator_base_macd():
    stock_data = StockData.generate_mock_data()

    stock_data.add_indicator("macd")

    assert "macd_line" in stock_data.df
