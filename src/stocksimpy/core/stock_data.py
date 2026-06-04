# src/stocksimpy/core/stock_data.py
from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Callable, Dict, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd


class StockData:
    """
    Container and validator for stock market time-series data.

    Loads, validates, and exports OHLCV data with DatetimeIndex.
    Supports CSV, Excel, SQL, JSON, DataFrames, dictionaries, and yfinance.
    All loaders validate required columns (Open, High, Low, Close, Volume)
    and datetime indexing.

    Parameters
    ----------
    df : pandas.DataFrame, optional
        Stock data with DatetimeIndex and OHLCV columns. If None,
        creates an empty instance. Default is None.

    Attributes
    ----------
    df : pandas.DataFrame
        Validated stock data with DatetimeIndex and columns 'Open', 'High',
        'Low', 'Close', 'Volume'. Multi-level column indexing is used for
        multi-ticker data.


    Examples
    --------
    >>> data = StockData.from_yfinance(["AAPL"], days_before=365)  # doctest: +SKIP
    >>> data.df.head()  # doctest: +SKIP
    """

    def __init__(self, df: Optional[pd.DataFrame] = None) -> None:
        # If no DataFrame is provided, create an empty container.
        if df is None:
            self.df = pd.DataFrame()
            return

        # Otherwise, process and validate provided data.
        df = self._process_and_validate(df)
        self.df = df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and standardize input DataFrame format.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw input DataFrame.

        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame with DatetimeIndex and MultiIndex columns.

        Raises
        ------
        ValueError
            If DataFrame lacks 'Date' column or DatetimeIndex.
        """
        df = df.copy()

        # We check if the df is already a DatetimeIndex which is seen when loaded from yfinance or formatted this way
        # The below code handles the case when df isn't already DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"])
                df.set_index("Date", inplace=True)
            else:
                raise ValueError(
                    "Input DataFrame must have a 'Date' column or a DatetimeIndex."
                )

        # Convert single-level columns into a two-level MultiIndex where the
        # second level is an empty string. This keeps the API consistent for
        # code that expects tuples like ('Close', symbol).
        if not isinstance(df.columns, pd.MultiIndex):
            df.columns = pd.MultiIndex.from_tuples([(c, "") for c in df.columns])

        df.sort_index(inplace=True)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        """
        Validate DataFrame structure and data integrity.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame to validate.

        Raises
        ------
        ValueError
            If DataFrame is empty, index unsorted, has duplicates,
            or contains negative volume.
        TypeError
            If index not DatetimeIndex or columns not numeric.
        """
        if df.empty:
            raise ValueError("DataFrame cannot be empty.")

        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("DataFrame index must be a DatetimeIndex.")

        # Check for sorted and unique index
        if not df.index.is_monotonic_increasing:
            raise ValueError("DataFrame index is not sorted monotonically.")
        if df.index.has_duplicates:
            raise ValueError("DataFrame index contains duplicate dates.")

        # Checks if all the required columns are present in the data frame
        required_cols = ["Open", "Close", "Volume", "High", "Low"]
        # Inspect first-level column names for required OHLCV fields
        first_level_cols = list(df.columns.get_level_values(0).unique())
        missing = [col for col in required_cols if col not in first_level_cols]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Since the data is in MultiIndex format, validate each ticker separately
        # include empty string ticker which represents single-ticker DataFrames
        tickers = list(df.columns.get_level_values(1).unique())
        for ticker in tickers:
            for col in required_cols:
                try:
                    series = df[(col, ticker)]
                except KeyError:
                    raise ValueError(
                        f"Missing required column '{col}' for ticker '{ticker}'"
                    )
                if not pd.api.types.is_numeric_dtype(series.dtype):
                    raise TypeError(
                        f"Column '{col}' must be a numerical type. Found: {series.dtype}"
                    )

            # Check for negative volume
            if (df[("Volume", ticker)] < 0).any():
                raise ValueError("Volume data contains negative values.")

            # Basic logic checking (Low <= Open, Low <= Close, High >= Open, High >= Close)
            if (df[("Low", ticker)] > df[("High", ticker)]).any():
                raise ValueError(
                    "OHLC data inconsistency: Low price is greater than High price."
                )
            if (df[("Open", ticker)] > df[("High", ticker)]).any() or (
                df[("Open", ticker)] < df[("Low", ticker)]
            ).any():
                raise ValueError(
                    "OHLC data inconsistency: Open price is outside the High/Low range."
                )
            if (df[("Close", ticker)] > df[("High", ticker)]).any() or (
                df[("Close", ticker)] < df[("Low", ticker)]
            ).any():
                raise ValueError(
                    "OHLC data inconsistency: Close price is outside the High/Low range."
                )

    def _process_and_validate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate the input DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw input DataFrame.

        Returns
        -------
        pandas.DataFrame
            Cleaned and validated DataFrame.
        """
        df_clean = self._clean(df)
        self._validate(df_clean)
        return df_clean

    # ------------------------------------------
    # LOAD DATA

    @classmethod
    def generate_mock_data(cls, days: int = 100, seed: int = 42) -> "StockData":
        """
        Generate synthetic OHLCV data for testing.

        Parameters
        ----------
        days : int, optional
            Number of days of data to generate. Default is 100.
        seed : int, optional
            Random seed for reproducibility. Default is 42.

        Returns
        -------
        StockData
            Instance containing generated mock data.

        Examples
        --------
        >>> data = StockData.generate_mock_data(50, seed=123)
        >>> len(data.df)
        50
        """
        np.random.seed(seed)
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)

        # Generate base prices
        base_prices = np.cumsum(np.random.randn(days)) + 100

        # Generate High-Low range
        ranges = np.random.rand(days) * 2  # Random range size
        high = base_prices + ranges
        low = base_prices - ranges

        # Generate Open and Close within the High-Low range
        daily_range = high - low
        open_ = low + daily_range * np.random.rand(days)
        close = low + daily_range * np.random.rand(days)

        volume = np.random.randint(1000, 10000, size=days)

        df = pd.DataFrame(
            {
                "Date": dates,
                "Open": open_,
                "High": high,
                "Low": low,
                "Close": close,
                "Volume": volume,
            }
        )
        return cls(df)

    @classmethod
    def from_csv(cls, file_path: str) -> "StockData":
        """
        Load stock data from CSV file.

        Expects 'Date' column or uses first column as DatetimeIndex.
        Removes timezone information if present.

        Parameters
        ----------
        file_path : str
            Path to CSV file.

        Returns
        -------
        StockData
            Instance with validated CSV data.

        Raises
        ------
        FileNotFoundError
            If file does not exist.
        ValueError
            If required OHLCV columns are missing.
        """
        # Read CSV with date parsing
        df = pd.read_csv(file_path, index_col=0)
        if df.index.name == "Date" or df.index.name == "Index":
            df.index.name = "Date"
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
        elif "Date" in df.columns:
            # Convert to datetime without timezone
            df["Date"] = pd.to_datetime(df["Date"])
            if hasattr(df["Date"].dtype, "tz") and df["Date"].dtype.tz is not None:
                df["Date"] = df["Date"].dt.tz_localize(None)
            df.set_index("Date", inplace=True)

        return cls(df)

    @classmethod
    def from_excel(cls, file_path: str) -> "StockData":
        """
        Load stock data from Excel file.

        Parameters
        ----------
        file_path : str
            Path to Excel file.

        Returns
        -------
        StockData
            Instance with loaded Excel data.
        """
        df = pd.read_excel(file_path)
        return cls(df)

    @classmethod
    def from_sql(cls, query: str, connection: Any) -> "StockData":
        """
        Load stock data from SQL database.

        Parameters
        ----------
        query : str
            SQL SELECT query.
        connection
            Open database connection.

        Returns
        -------
        StockData
            Instance with query result data.
        """
        df = pd.read_sql(query, connection)
        return cls(df)

    @classmethod
    def from_yfinance(
        cls,
        tickers: str | list[str],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        days_before: Optional[int] = None,
    ) -> "StockData":
        """
        Load stock data from Yahoo Finance.

        Parameters
        ----------
        tickers : list
            List of ticker symbols, e.g., ['AAPL', 'MSFT'].
        start_date : date, optional
            Start date (inclusive). Ignored if days_before is set.
            Default is None.
        end_date : date, optional
            End date (inclusive). Ignored if days_before is set.
            Default is None.
        days_before : int, optional
            Number of days before today to retrieve. Overrides start_date
            and end_date.
            Default is None.

        Returns
        -------
        StockData
            Instance with multi-ticker OHLCV data with DatetimeIndex.

        Raises
        ------
        ImportError
            If yfinance package is not installed.

        Notes
        -----
        If `days_before` is provided, the `start_date` and `end_date` parameters are ignored.
        Requires internet connection.
        Prices are auto-adjusted for splits/dividends.
        Multi-ticker requests result in a MultiIndex DataFrame.

        Examples
        --------
        >>> data = StockData.from_yfinance(["AAPL"], days_before=365)  # doctest: +SKIP
        >>> data.df.head()  # doctest: +SKIP
        """
        import yfinance as yf

        try:
            if days_before:
                today = date.today()
                start_date = today - timedelta(days=days_before)
                end_date = today
            data = yf.download(
                tickers, start=start_date, end=end_date, auto_adjust=True
            )
            # If the download returned a DataFrame, ensure 'Date' is a column/index
            data.reset_index(inplace=True)
            # Set Date as index before adjusting columns
            if "Date" in data.columns:
                data["Date"] = pd.to_datetime(data["Date"])
                data.set_index("Date", inplace=True)

            return cls(data)
        except ImportError:
            raise ImportError(
                """
                yfinance is not installed. Install it to use Yahoo Finance loaders.
                try using: `pip install yfinance`
                """
            )

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> "StockData":
        """
        Create StockData from existing DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with OHLCV data.

        Returns
        -------
        StockData
            Instance initialized with DataFrame.
        """
        return cls(df)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StockData":
        """
        Create StockData from dictionary.

        Parameters
        ----------
        data : dict
            Column names as keys, value lists as data.

        Returns
        -------
        StockData
            Instance initialized from dictionary.
        """
        df = pd.DataFrame(data)
        return cls(df)

    @classmethod
    def from_json(cls, json_data: dict[str, Any]) -> "StockData":
        """
        Create StockData from JSON object.

        Parameters
        ----------
        json_data : dict
            Parsed JSON as dictionary.

        Returns
        -------
        StockData
            Instance initialized from JSON.
        """
        df = pd.DataFrame(json_data)
        return cls(df)

    @classmethod
    def from_sqlite(cls, query: str, db_path: str) -> "StockData":
        """
        Load stock data from SQLite database.

        Parameters
        ----------
        query : str
            SQL SELECT query.
        db_path : str
            Path to SQLite database file.

        Returns
        -------
        StockData
            Instance with query result data.

        Examples
        --------
        >>> data = StockData.from_sqlite("SELECT * FROM stocks", "data.db")  # doctest: +SKIP
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        df = pd.read_sql(query, conn)
        conn.close()
        return cls(df)

    @staticmethod
    def auto_loader(
        source: Union[pd.DataFrame, dict[str, Any], str, tuple[Any, ...]],
        **kwargs: Any,
    ) -> "StockData":
        """
        Auto-detect input type and load data.

        Supports CSV, Excel, SQLite, DataFrame, dict, JSON, and yfinance
        via tuple specification.

        Parameters
        ----------
        source : str, dict, tuple, or pandas.DataFrame
            Input source. File paths (str) trigger format detection by extension.
            Tuples trigger yfinance loading: (ticker, days_before) or
            (ticker, start_date, end_date).
        **kwargs
            Additional arguments for specific loaders (e.g., 'query' for SQLite).

        Returns
        -------
        StockData
            Instance with loaded and validated data.

        Raises
        ------
        ValueError
            If file extension is unsupported or tuple format is invalid.
        TypeError
            If source type is not recognized.

        Examples
        --------
        >>> data = StockData.auto_loader("prices.csv")  # doctest: +SKIP
        >>> data = StockData.auto_loader({"Open": [...], "Close": [...]})  # doctest: +SKIP
        """
        import os
        from datetime import date

        # DataFrame
        if isinstance(source, pd.DataFrame):
            return StockData.from_dataframe(source)

        # Dict (possibly JSON data)
        elif isinstance(source, dict):
            return StockData.from_dict(source)

        # File path string
        elif isinstance(source, str):
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".csv":
                return StockData.from_csv(source)
            elif ext in [".xls", ".xlsx"]:
                return StockData.from_excel(source)
            elif ext == ".db":
                query = kwargs.get("query", "SELECT * FROM stock_data")
                return StockData.from_sqlite(query, source)
            else:
                raise ValueError(f"Unsupported file extension: {ext}")

        # Tuple -> yfinance loader
        elif isinstance(source, tuple):
            if len(source) == 2 and isinstance(source[1], int):
                ticker, days_before = source
                return StockData.from_yfinance(tickers=ticker, days_before=days_before)
            elif (
                len(source) == 3
                and isinstance(source[1], date)
                and isinstance(source[2], date)
            ):
                ticker, start_date, end_date = source
                return StockData.from_yfinance(
                    tickers=ticker, start_date=start_date, end_date=end_date
                )
            else:
                raise ValueError(
                    "Tuple input must be (ticker, days_before) or (ticker, start_date, end_date)"
                )

        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    # -----------------------
    # BASIC INFO AND FUNCTIONALITIES

    def get(self, column: str) -> pd.Series:
        """
        Get a column from the DataFrame.

        Parameters
        ----------
        column : str
            Column name.

        Returns
        -------
        pandas.Series
            Column data.
        """
        return self.df[column]

    def slice(self, start: Any = None, end: Any = None) -> pd.DataFrame:
        """
        Slice DataFrame by date range.

        Parameters
        ----------
        start : str or datetime, optional
            Start date (inclusive).
        end : str or datetime, optional
            End date (inclusive).

        Returns
        -------
        pandas.DataFrame
            Sliced data.
        """
        return self.df.loc[start:end]

    def head(self, n: int = 5) -> pd.DataFrame:
        """
        Return first n rows.

        Parameters
        ----------
        n : int, optional
            Number of rows. Default is 5.

        Returns
        -------
        pandas.DataFrame
            First n rows.
        """
        return self.df.head(n)

    def info(self) -> None:
        """
        Display DataFrame information.

        Returns
        -------
        None
        """
        self.df.info()

    def fill_missing(self, method: Literal["ffill", "bfill"] = "ffill") -> "StockData":
        """
        Fill missing values in DataFrame.

        Parameters
        ----------
        method : str, optional
            Fill method: 'ffill' (forward fill) or 'bfill' (backward fill).
            Default is 'ffill'.

        Returns
        -------
        StockData
            Self for method chaining.
        """
        if method == "ffill":
            self.df.ffill(inplace=True)
        elif method == "bfill":
            self.df.bfill(inplace=True)
        else:
            raise ValueError(
                method
                + ' is not a fill_missing method option, use either "ffill" or "bfill"'
            )

        return self

    def check_missing(self) -> pd.Series:
        """
        Count missing values per column.

        Returns
        -------
        pandas.Series
            Missing value count per column.
        """
        return self.df.isnull().sum()

    def add_indicator(
        self,
        indicator: str | Callable[..., dict[str, pd.Series]],
        *args: Any,
        base_col: str = "Close",
        symbol: str = "",
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Add one or more technical indicators to a stock's data.

        This method applies an indicator function to a specified base column
        (e.g., "close") for a given stock symbol in a MultiIndex DataFrame.
        The indicator function must return a dictionary mapping indicator names
        to pandas Series. Each resulting indicator is inserted as a new column
        under the corresponding stock symbol in the DataFrame.

        You can use ``Indicators`` module from ``stocksimpy`` to generate these
        indicators quickly.

        Parameters
        ----------
        indicator : str or callable
            Either the name of a built-in indicator (e.g., "sma", "rsi") or a
            custom function that accepts a pandas Series and returns a dict of
            {str: pandas.Series}.
        base_col : str, default "Close"
            Base data column to apply the indicator to.
        symbol : str, default ""
            Stock symbol identifying the top-level column in the MultiIndex.
            Leave empty ("") if want to apply to all the symbols
        overwrite : bool, default False
            Whether to overwrite existing indicator columns.
        *args, **kwargs
            Additional arguments passed directly to the indicator function.


        Returns
        -------
        None
            The method mutates the StockData instance in place by adding new
            indicator columns.

        Raises
        ------
        KeyError
            If the specified stock or base column does not exist in the data.

        Examples
        --------
        >>> def calculate_sma(series, window):
        ...     return {f"sma_{window}": series.rolling(window).mean()}
        >>> data.add_indicator(calculate_sma, "close", 20, "AAPL")  # doctest: +SKIP

        >>> def calculate_bands(series, window):
        ...     sma = series.rolling(window).mean()
        ...     std = series.rolling(window).std()
        ...     return {
        ...         f"bb_upper_{window}": sma + 2 * std,
        ...         f"bb_lower_{window}": sma - 2 * std,
        ...     }
        >>> data.add_indicator(calculate_bands, "close", 20, "MSFT")  # doctest: +SKIP
        """

        if symbol != "":
            if (base_col, symbol) not in self.df:
                raise KeyError(
                    f"The key ({base_col}, {symbol}) does NOT exist in the dataframe"
                )

        if isinstance(indicator, str):
            indicator = indicator.lower()

            from stocksimpy.addons.indicators import Indicators

            indicator_func = Indicators.get_name_func()[indicator]
        elif callable(indicator):
            indicator_func = indicator
        else:
            raise TypeError(
                f"``indicator`` type can only be str or callable, but argument is of type {type(indicator)}"
            )

        if symbol == "":
            symbol_it = self.df.columns.get_level_values(1).unique()
        else:
            symbol_it = [symbol]

        for ticker in symbol_it:
            series = self.df[(base_col, ticker)]

            result = indicator_func(series, *args, **kwargs)

            if not isinstance(result, dict):
                raise TypeError("Indicator function must return dict[str, pd.Series]")

            new_cols = {}

            for name, values in result.items():
                col_key = (name, ticker)

                if overwrite or (col_key not in self.df.columns):
                    new_cols[col_key] = values

            new_df = pd.DataFrame(new_cols, index=self.df.index)
            self.df = pd.concat([self.df, new_df], axis=1)

    def add_indicator_all(
        self, symbol: str = "", base_col: str = "Close", overwrite: bool = False
    ) -> None:
        """
        Add all built-in technical indicators to the stock data for one or more symbols.

        This method iterates over all indicator functions in the Indicators module and applies them
        to the specified base column for each symbol in the DataFrame. It uses the `add_indicator`
        method for each indicator. If `symbol` is empty, indicators are added for all symbols.

        Parameters
        ----------
        symbol : str, optional
            Stock symbol identifying the top-level column in the MultiIndex. If empty (""), applies to all symbols.
        base_col : str, optional
            Base data column to apply the indicators to. Default is "Close".
        overwrite : bool, optional
            Whether to overwrite existing indicator columns. Default is False.

        Returns
        -------
        None
            The method mutates the StockData instance in place by adding new indicator columns.

        Examples
        --------
        >>> data.add_indicator_all()  # Adds all indicators to all symbols
        >>> data.add_indicator_all(symbol="AAPL", base_col="Close", overwrite=True)  # Adds all indicators for AAPL, overwriting existing
        """
        from stocksimpy.addons.indicators import Indicators

        if symbol == "":
            tickers = self.df.columns.get_level_values(1).unique()
        else:
            tickers = [symbol]

        for ticker in tickers:
            for indicator in Indicators.get_name_func():
                self.add_indicator(
                    indicator, base_col=base_col, symbol=ticker, overwrite=overwrite
                )

    # --------------------------
    # EXPORT DATA

    def to_csv(self, file_path: str, **kwargs: Any) -> str:
        """
        Export DataFrame to CSV file.

        Parameters
        ----------
        file_path : str
            Output file path.
        **kwargs
            Additional arguments passed to pandas.to_csv.

        Returns
        -------
        str
            Path to exported file.
        """
        # Ensure the index is saved as 'Date' column
        df_to_save = self.df.copy()
        df_to_save.index.name = "Date"
        # If we have MultiIndex columns, flatten them for CSV export
        if isinstance(df_to_save.columns, pd.MultiIndex):
            df_to_save.columns = [
                col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
                for col in df_to_save.columns
            ]
        df_to_save.to_csv(file_path, **kwargs)
        return file_path

    def to_excel(self, file_path: str, **kwargs: Any) -> str:
        """
        Export DataFrame to Excel file.

        Parameters
        ----------
        file_path : str
            Output file path (.xls or .xlsx).
        **kwargs
            Additional arguments passed to pandas.to_excel.

        Returns
        -------
        str
            Path to exported file.
        """
        self.df.to_excel(file_path, **kwargs)
        return file_path

    def to_sql(
        self,
        table_name: str,
        connection: Any,
        if_exists: str = "replace",
        **kwargs: Any,
    ) -> str:
        """
        Export DataFrame to SQL table.

        Parameters
        ----------
        table_name : str
            Target table name.
        connection : sqlalchemy.engine.Connection or sqlite3.Connection
            Active database connection.
        if_exists : str, optional
            Behavior if table exists: 'fail', 'replace', 'append'.
            Default is 'replace'.
        **kwargs
            Additional arguments passed to pandas.to_sql.

        Returns
        -------
        str
            Table name in database.
        """
        self.df.to_sql(
            table_name, connection, if_exists=if_exists, index=True, **kwargs
        )
        return table_name

    def to_sqlite(
        self, table_name: str, db_path: str, if_exists: str = "replace", **kwargs: Any
    ) -> str:
        """
        Export DataFrame to SQLite database.

        Parameters
        ----------
        table_name : str
            Target table name in database.
        db_path : str
            Path to SQLite database file.
        if_exists : str, optional
            Behavior if table exists: 'fail', 'replace', 'append'.
            Default is 'replace'.
        **kwargs
            Additional arguments passed to pandas.to_sql.

        Returns
        -------
        str
            Path to database file.
        """
        import sqlite3

        conn = sqlite3.connect(db_path)
        self.df.to_sql(table_name, conn, if_exists=if_exists, index=True, **kwargs)
        conn.close()
        return db_path

    def to_dataframe(self) -> pd.DataFrame:
        """
        Return copy of underlying DataFrame.

        Returns
        -------
        pandas.DataFrame
            DataFrame with DatetimeIndex and OHLCV columns.
        """
        return self.df.copy()

    def to_dict(self, orient: str = "records") -> dict[str, Any]:
        """
        Export DataFrame to dictionary.

        Parameters
        ----------
        orient : str, optional
            Dictionary orientation. Default is 'records'.

        Returns
        -------
        dict
            DataFrame as dictionary.
        """
        return self.df.to_dict(orient=orient)

    def to_json(
        self,
        file_path: Optional[str] = None,
        orient: str = "records",
        **kwargs: Any,
    ) -> str:
        """
        Export DataFrame to JSON.

        Parameters
        ----------
        file_path : str, optional
            Output file path. If None, returns JSON string.
            Default is None.
        orient : str, optional
            JSON orientation ('records', 'columns', etc.).
            Default is 'records'.
        **kwargs
            Additional arguments passed to pandas.to_json.

        Returns
        -------
        str
            JSON string

        Raises
        ------
        IOError
            If file cannot be written.
        """
        json_str = self.df.to_json(orient=orient, **kwargs)
        if file_path:
            with open(file_path, "w") as f:
                f.write(json_str)
        return json_str

    def to_custom(
        self, export_func: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """
        Export using custom function.

        Parameters
        ----------
        export_func : callable
            Function taking DataFrame as first argument.
        *args
            Positional arguments passed to export_func.
        **kwargs
            Keyword arguments passed to export_func.

        Returns
        -------
        result of export_func(self.df, *args, **kwargs)
            Export result from custom function.

        Examples
        --------
        >>> def save_parquet(df, path):
        ...     df.to_parquet(path)
        ...     return path
        >>> data.to_custom(save_parquet, "output.parquet")  # doctest: +SKIP

        Raises
        ------
        IOError
            If file cannot be written.

        Raises
        ------
        IOError
            If file cannot be written.

        Raises
        ------
        IOError
            If database cannot be written.
        """
        return export_func(self.df, *args, **kwargs)
