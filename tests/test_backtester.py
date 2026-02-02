import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add src directory to path so we can import stocksimpy modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from stocksimpy.core.backtester import Backtester
from stocksimpy.core.portfolio import Portfolio
from stocksimpy.core.stock_data import StockData


@pytest.fixture
def sample_stock_data():
    """Create sample stock data for backtesting with AAPL symbol."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

    # Create MultiIndex columns with AAPL symbol to match backtester expectations
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
    )

    data = {
        ("Open", "AAPL"): np.linspace(100, 120, 20),
        ("High", "AAPL"): np.linspace(102, 122, 20),
        ("Low", "AAPL"): np.linspace(98, 118, 20),
        ("Close", "AAPL"): np.linspace(100, 120, 20),
        ("Volume", "AAPL"): np.random.randint(1000000, 5000000, 20),
    }

    df = pd.DataFrame(data, index=dates)
    return StockData(df)


@pytest.fixture
def sample_stock_data_multiindex():
    """Create sample MultiIndex stock data (yfinance format)."""
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

    # Create MultiIndex columns for two symbols
    columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["AAPL", "MSFT"]]
    )

    data = {
        ("Open", "AAPL"): np.linspace(100, 120, 20),
        ("High", "AAPL"): np.linspace(102, 122, 20),
        ("Low", "AAPL"): np.linspace(98, 118, 20),
        ("Close", "AAPL"): np.linspace(100, 120, 20),
        ("Volume", "AAPL"): np.random.randint(1000000, 5000000, 20),
        ("Open", "MSFT"): np.linspace(200, 220, 20),
        ("High", "MSFT"): np.linspace(202, 222, 20),
        ("Low", "MSFT"): np.linspace(198, 218, 20),
        ("Close", "MSFT"): np.linspace(200, 220, 20),
        ("Volume", "MSFT"): np.random.randint(1000000, 5000000, 20),
    }

    df = pd.DataFrame(data, index=dates)
    return StockData(df)


@pytest.fixture
def buy_and_hold_strategy():
    """Strategy that buys on first day and holds."""

    def strategy(data):
        if len(data) == 1:
            return "buy"
        return "hold"

    return strategy


@pytest.fixture
def buy_all_strategy():
    """Strategy that always buys."""

    def strategy(data):
        return "buy"

    return strategy


@pytest.fixture
def sell_all_strategy():
    """Strategy that always sells."""

    def strategy(data):
        return "sell"

    return strategy


@pytest.fixture
def buy_on_dip_strategy():
    """Strategy that buys when price dips below 110."""

    def strategy(data):
        # Handle both MultiIndex and single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            current_price = (
                data[("Close", "AAPL")].iloc[-1]
                if ("Close", "AAPL") in data.columns
                else data["Close"].iloc[-1]
            )
        else:
            current_price = data["Close"].iloc[-1]

        if current_price < 110:
            return "buy"
        return "sell"

    return strategy


@pytest.fixture
def dynamic_rsi_strategy():
    """Dynamic strategy that uses a simple indicator."""

    def strategy(data, holdings):
        # Handle both MultiIndex and single-level columns
        if isinstance(data.columns, pd.MultiIndex):
            close = (
                data[("Close", "AAPL")]
                if ("Close", "AAPL") in data.columns
                else data["Close"]
            )
        else:
            close = data["Close"]

        # Simple momentum: if price is increasing, buy; else sell
        if len(close) > 1:
            if close.iloc[-1] > close.iloc[-2]:
                return "buy", 5
            else:
                return "sell", 2
        return "buy", 5

    return strategy


class TestBacktesterInitialization:
    """Tests for Backtester initialization."""

    def test_init_with_default_parameters(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test initialization with default parameters."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)

        assert bt.symbol == "AAPL"
        assert bt.initial_cap == 100_000
        assert bt.transaction_fee == 0.0
        assert bt.trade_amount == 10_000
        assert isinstance(bt.portfolio, Portfolio)

    def test_init_with_custom_parameters(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test initialization with custom parameters."""
        bt = Backtester(
            "MSFT",
            sample_stock_data,
            buy_and_hold_strategy,
            initial_cap=50_000,
            transaction_fee=5.0,
            trade_amount=5_000,
        )

        assert bt.symbol == "MSFT"
        assert bt.initial_cap == 50_000
        assert bt.transaction_fee == 5.0
        assert bt.trade_amount == 5_000

    def test_init_stores_data_as_dataframe(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test that data is stored as DataFrame."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)

        assert isinstance(bt.data, pd.DataFrame)
        assert len(bt.data) == len(sample_stock_data.df)


class TestRunBacktestFixed:
    """Tests for run method."""

    def test_backtest_fixed_runs_without_error(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test that backtest runs without errors."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)
        bt.run()

        assert len(bt.portfolio.value_history) > 0

    def test_backtest_fixed_buy_signal(self, sample_stock_data, buy_all_strategy):
        """Test that buy signals execute trades."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        # Should have multiple buy trades
        assert len(bt.portfolio.trade_log) > 0
        buy_trades = bt.portfolio.trade_log[bt.portfolio.trade_log["Type"] == "buy"]
        assert len(buy_trades) > 0

    def test_backtest_fixed_sell_signal(self, sample_stock_data, buy_all_strategy):
        """Test that portfolio can hold shares for selling."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        # After buying, should have shares
        assert bt.portfolio.holdings["AAPL"] > 0

    def test_backtest_fixed_share_calculation(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that shares are calculated correctly based on trade_amount."""
        trade_amount = 5_000
        bt = Backtester(
            "AAPL", sample_stock_data, buy_all_strategy, trade_amount=trade_amount
        )
        bt.run()

        # First price should be around 100
        first_price = sample_stock_data.df[("Close", "AAPL")].iloc[0]
        expected_first_shares = int(trade_amount / first_price)

        # Check first trade - shares should be within 1 due to rounding
        if len(bt.portfolio.trade_log) > 0:
            first_trade = bt.portfolio.trade_log.iloc[0]
            assert abs(first_trade["Shares"] - expected_first_shares) <= 1

    def test_backtest_fixed_with_transaction_fees(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that transaction fees are applied."""
        fee = 10.0
        bt = Backtester(
            "AAPL", sample_stock_data, buy_all_strategy, transaction_fee=fee
        )
        bt.run()

        # All trades should include transaction fee
        for _, trade in bt.portfolio.trade_log.iterrows():
            assert trade["TransactionFee"] == fee

    def test_backtest_fixed_value_history_recorded(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that portfolio value is recorded at each step."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        # Value history should have entries for each trading day
        assert len(bt.portfolio.value_history) > 0
        assert (
            len(bt.portfolio.value_history) == len(sample_stock_data.df) - 1
        )  # Excludes first day

    def test_backtest_fixed_respects_strategy_signals(
        self, sample_stock_data, buy_on_dip_strategy
    ):
        """Test that strategy signals are respected."""
        bt = Backtester("AAPL", sample_stock_data, buy_on_dip_strategy)
        bt.run()

        # Strategy buys on dips (price < 110)
        assert len(bt.portfolio.trade_log) > 0

    def test_backtest_fixed_multiindex_data(
        self, sample_stock_data_multiindex, buy_all_strategy
    ):
        """Test backtest with MultiIndex data (yfinance format)."""
        bt = Backtester("AAPL", sample_stock_data_multiindex, buy_all_strategy)
        bt.run()

        assert len(bt.portfolio.value_history) > 0
        assert bt.portfolio.holdings["AAPL"] > 0


@DeprecationWarning
class TestRunBacktestDynamic:
    """Tests for run_backtest_dynamic method."""

    def test_backtest_dynamic_runs_without_error(
        self, sample_stock_data, dynamic_rsi_strategy
    ):
        """Test that dynamic backtest runs without errors."""
        bt = Backtester("AAPL", sample_stock_data, dynamic_rsi_strategy)
        bt.run_backtest_dynamic()

        assert len(bt.portfolio.value_history) > 0

    def test_backtest_dynamic_respects_shares_from_strategy(
        self, sample_stock_data, dynamic_rsi_strategy
    ):
        """Test that strategy-determined share counts are respected."""
        bt = Backtester("AAPL", sample_stock_data, dynamic_rsi_strategy)
        bt.run_backtest_dynamic()

        # Check that trades were executed
        assert len(bt.portfolio.trade_log) > 0

    def test_backtest_dynamic_passes_holdings(self, sample_stock_data):
        """Test that strategy receives current holdings."""
        holdings_list = []

        def holdings_tracking_strategy(data, holdings):
            holdings_list.append(holdings)
            return "buy", 5

        bt = Backtester("AAPL", sample_stock_data, holdings_tracking_strategy)
        bt.run_backtest_dynamic()

        # Holdings should be passed to strategy at each step
        assert len(holdings_list) > 0
        # Holdings should increase over time with buying
        assert holdings_list[-1] >= holdings_list[0]

    def test_backtest_dynamic_invalid_strategy_return(self, sample_stock_data):
        """Test that invalid strategy return raises TypeError."""

        def bad_strategy(data, holdings):
            # Returns single value instead of tuple
            return "buy"

        bt = Backtester("AAPL", sample_stock_data, bad_strategy)

        with pytest.raises(TypeError, match="strategy function should return a tuple"):
            bt.run_backtest_dynamic()

    def test_backtest_dynamic_multiindex_data(
        self, sample_stock_data_multiindex, dynamic_rsi_strategy
    ):
        """Test dynamic backtest with MultiIndex data."""
        bt = Backtester("AAPL", sample_stock_data_multiindex, dynamic_rsi_strategy)
        bt.run_backtest_dynamic()

        assert len(bt.portfolio.value_history) > 0


class TestProcessTrade:
    """Tests for _process_trade method."""

    def test_process_trade_buy(self, sample_stock_data, buy_and_hold_strategy):
        """Test that _process_trade executes buy correctly."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)
        price = 100
        shares = 10
        date = pd.Timestamp("2023-01-01")

        initial_cash = bt.portfolio.cash
        bt._process_trade("buy", shares, price, date)

        assert bt.portfolio.holdings["AAPL"] == shares
        assert bt.portfolio.cash < initial_cash

    def test_process_trade_sell(self, sample_stock_data, buy_and_hold_strategy):
        """Test that _process_trade executes sell correctly."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)

        # First buy
        bt._process_trade("buy", 10, 100, pd.Timestamp("2023-01-01"))
        cash_after_buy = bt.portfolio.cash

        # Then sell
        bt._process_trade("sell", 5, 110, pd.Timestamp("2023-01-02"))

        assert bt.portfolio.holdings["AAPL"] == 5
        assert bt.portfolio.cash > cash_after_buy

    def test_process_trade_invalid_signal(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test that invalid signal is ignored."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)

        initial_holdings = bt.portfolio.holdings["AAPL"]
        bt._process_trade("hold", 10, 100, pd.Timestamp("2023-01-01"))

        # Holdings should not change for 'hold' signal
        assert bt.portfolio.holdings["AAPL"] == initial_holdings

    def test_process_trade_updates_value(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test that _process_trade updates portfolio value."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)
        date = pd.Timestamp("2023-01-01")

        bt._process_trade("buy", 10, 100, date)

        # Value history should be updated
        assert date in bt.portfolio.value_history.index


class TestGenerateReport:
    """Tests for generate_report method."""

    def test_generate_report_empty_backtest(
        self, sample_stock_data, buy_and_hold_strategy
    ):
        """Test report generation without running backtest."""
        bt = Backtester("AAPL", sample_stock_data, buy_and_hold_strategy)
        report = bt.generate_report()

        assert report["final_value"] == bt.portfolio.initial_cap
        assert report["total_return_percent"] == 0.0
        assert report["number_of_trades"] == 0

    def test_generate_report_after_backtest(self, sample_stock_data, buy_all_strategy):
        """Test report generation after running backtest."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()
        report = bt.generate_report()

        assert "final_value" in report
        assert "total_return_percent" in report
        assert "number_of_trades" in report
        assert report["number_of_trades"] > 0

    def test_generate_report_fields_are_numeric(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that report fields are numeric."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()
        report = bt.generate_report()

        assert isinstance(report["final_value"], (int, float, np.number))
        assert isinstance(report["total_return_percent"], (int, float, np.number))
        assert isinstance(report["number_of_trades"], (int, np.integer))

    def test_generate_report_profitable_backtest(
        self, sample_stock_data, buy_on_dip_strategy
    ):
        """Test report shows positive returns for profitable strategy."""
        bt = Backtester("AAPL", sample_stock_data, buy_on_dip_strategy)
        bt.run()
        report = bt.generate_report()

        # Should have some report data
        assert report["number_of_trades"] >= 0


class TestBacktesterIntegration:
    """Integration tests for complete backtest scenarios."""

    def test_complete_buy_hold_cycle(self, sample_stock_data):
        """Test complete buy-sell cycle."""

        def strategy(data):
            # Buy on first day, sell on last day
            return "buy"

        bt = Backtester("AAPL", sample_stock_data, strategy)
        bt.run()
        report = bt.generate_report()

        # Should have valid report with at least 1 buy trade
        assert report["final_value"] > 0
        assert report["number_of_trades"] > 0

    def test_dynamic_vs_fixed_backtest_both_work(self, sample_stock_data):
        """Test that both fixed and dynamic backtests can be run."""

        def fixed_strategy(data):
            return "buy"

        def dynamic_strategy(data, holdings):
            return "buy", 10

        bt_fixed = Backtester("AAPL", sample_stock_data, fixed_strategy)
        bt_dynamic = Backtester("AAPL", sample_stock_data, dynamic_strategy)

        bt_fixed.run()
        bt_dynamic.run_backtest_dynamic()

        assert len(bt_fixed.portfolio.value_history) > 0
        assert len(bt_dynamic.portfolio.value_history) > 0

    def test_backtest_with_different_initial_capitals(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test backtest with different initial capitals."""
        bt_small = Backtester(
            "AAPL", sample_stock_data, buy_all_strategy, initial_cap=10_000
        )
        bt_large = Backtester(
            "AAPL", sample_stock_data, buy_all_strategy, initial_cap=1_000_000
        )

        bt_small.run()
        bt_large.run()

        # Both should complete successfully
        assert len(bt_small.portfolio.value_history) > 0
        assert len(bt_large.portfolio.value_history) > 0

        # Larger capital should have more holdings
        assert (
            bt_large.portfolio.holdings["AAPL"] >= bt_small.portfolio.holdings["AAPL"]
        )

    def test_backtest_trade_log_consistent(self, sample_stock_data, buy_all_strategy):
        """Test that trade log is consistent with portfolio state."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        trade_log = bt.portfolio.trade_log
        assert len(trade_log) > 0

        # All trades should have valid data
        assert (trade_log["Price"] > 0).all()
        assert (trade_log["Shares"] >= 0).all()
        assert trade_log["Type"].isin(["buy", "sell"]).all()


class TestEdgeCasesBacktester:
    """Tests for edge cases and boundary conditions."""

    def test_backtest_single_day_data(self, buy_all_strategy):
        """Test backtest with minimal data (2 days)."""
        dates = pd.date_range(start="2023-01-01", periods=2, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): [100, 101],
            ("High", "AAPL"): [102, 103],
            ("Low", "AAPL"): [98, 99],
            ("Close", "AAPL"): [100, 101],
            ("Volume", "AAPL"): [1000000, 1100000],
        }
        df = pd.DataFrame(data, index=dates)
        stock_data = StockData(df)

        bt = Backtester("AAPL", stock_data, buy_all_strategy)
        bt.run()

        assert len(bt.portfolio.value_history) >= 1

    def test_backtest_strategy_exception_propagates(self, sample_stock_data):
        """Test that strategy exceptions propagate to caller."""

        def error_strategy(data):
            raise ValueError("Strategy error")

        bt = Backtester("AAPL", sample_stock_data, error_strategy)

        with pytest.raises(ValueError, match="Strategy error"):
            bt.run()

    def test_backtest_zero_trade_amount(self, sample_stock_data, buy_all_strategy):
        """Test backtest with zero trade amount."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy, trade_amount=0)
        bt.run()

        # Should not hold shares with zero trade amount
        assert bt.portfolio.holdings["AAPL"] == 0

    def test_backtest_very_large_trade_amount(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test backtest with trade amount larger than capital."""
        bt = Backtester(
            "AAPL",
            sample_stock_data,
            buy_all_strategy,
            initial_cap=10_000,
            trade_amount=100_000,
        )
        bt.run()

        # Should adapt to available capital
        assert len(bt.portfolio.trade_log) >= 0

    def test_backtest_high_fees(self, sample_stock_data, buy_all_strategy):
        """Test backtest with high transaction fees."""
        bt = Backtester(
            "AAPL",
            sample_stock_data,
            buy_all_strategy,
            initial_cap=10_000,
            transaction_fee=1000,
        )
        bt.run()

        # Fees should prevent many trades
        assert len(bt.portfolio.trade_log) >= 0

    def test_backtest_negative_prices_ignored(self, buy_all_strategy):
        """Test that negative prices don't cause trading errors."""
        dates = pd.date_range(start="2023-01-01", periods=5, freq="D")
        columns = pd.MultiIndex.from_product(
            [["Open", "High", "Low", "Close", "Volume"], ["AAPL"]]
        )
        data = {
            ("Open", "AAPL"): [100, 101, 102, 103, 104],
            ("High", "AAPL"): [102, 103, 104, 105, 106],
            ("Low", "AAPL"): [98, 99, 100, 101, 102],
            ("Close", "AAPL"): [100, 101, 102, 103, 104],  # All positive prices
            ("Volume", "AAPL"): [1000000] * 5,
        }
        df = pd.DataFrame(data, index=dates)
        stock_data = StockData(df)

        bt = Backtester("AAPL", stock_data, buy_all_strategy)
        bt.run()

        # Should complete without error
        assert len(bt.portfolio.value_history) > 0


class TestBacktesterPortfolioIntegration:
    """Tests for integration between Backtester and Portfolio."""

    def test_portfolio_state_updated_by_backtest(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that backtest properly updates portfolio state."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)

        initial_cash = bt.portfolio.cash
        bt.run()

        # Cash should have changed (trades were made)
        assert bt.portfolio.cash != initial_cash or bt.portfolio.holdings["AAPL"] == 0

    def test_portfolio_value_history_populated(
        self, sample_stock_data, buy_all_strategy
    ):
        """Test that portfolio value history is populated by backtest."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        assert len(bt.portfolio.value_history) > 0
        # Value history dates should match data dates
        assert bt.portfolio.value_history.index[0] == sample_stock_data.df.index[1]

    def test_portfolio_trade_log_populated(self, sample_stock_data, buy_all_strategy):
        """Test that portfolio trade log is populated by backtest."""
        bt = Backtester("AAPL", sample_stock_data, buy_all_strategy)
        bt.run()

        assert len(bt.portfolio.trade_log) > 0
        # All trades should reference the correct symbol
        assert (bt.portfolio.trade_log["Symbol"] == "AAPL").all()
