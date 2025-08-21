import numpy as np
import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys
import io
import argparse

from tfunify.cli import (
    _load_csv,
    _summarise,
    cmd_european,
    cmd_american,
    cmd_tsmom,
    main,
)


class TestLoadCSV:
    """Tests for CSV loading functionality."""

    def test_basic_csv_loading(self):
        """Test loading a basic CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close", "high", "low"])
            writer.writerow(["2023-01-01", "100.0", "101.0", "99.0"])
            writer.writerow(["2023-01-02", "102.0", "103.0", "101.0"])
            writer.writerow(["2023-01-03", "101.5", "102.5", "100.5"])
            csv_path = f.name

        try:
            data = _load_csv(csv_path)

            assert "close" in data
            assert "high" in data
            assert "low" in data

            np.testing.assert_array_equal(data["close"], [100.0, 102.0, 101.5])
            np.testing.assert_array_equal(data["high"], [101.0, 103.0, 102.5])
            np.testing.assert_array_equal(data["low"], [99.0, 101.0, 100.5])

        finally:
            Path(csv_path).unlink()

    def test_csv_with_only_close(self):
        """Test CSV with only close prices (high/low default to close)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close"])
            writer.writerow(["2023-01-01", "100.0"])
            writer.writerow(["2023-01-02", "102.0"])
            csv_path = f.name

        try:
            data = _load_csv(csv_path)

            # High and low should default to close
            np.testing.assert_array_equal(data["close"], [100.0, 102.0])
            np.testing.assert_array_equal(data["high"], [100.0, 102.0])
            np.testing.assert_array_equal(data["low"], [100.0, 102.0])

        finally:
            Path(csv_path).unlink()

    def test_csv_missing_close_column(self):
        """Test CSV missing required close column."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "high", "low"])
            writer.writerow(["2023-01-01", "101.0", "99.0"])
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="CSV must contain a 'close' column"):
                _load_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_nonexistent_csv_file(self):
        """Test loading nonexistent CSV file."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            _load_csv("nonexistent_file.csv")

    def test_empty_csv_file(self):
        """Test loading empty CSV file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="CSV file is empty"):
                _load_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_csv_with_invalid_data(self):
        """Test CSV with invalid numeric data."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close"])
            writer.writerow(["2023-01-01", "100.0"])
            writer.writerow(["2023-01-02", "invalid"])  # Invalid number
            csv_path = f.name

        try:
            with pytest.raises(ValueError, match="Error parsing row"):
                _load_csv(csv_path)
        finally:
            Path(csv_path).unlink()

    def test_csv_with_extra_columns(self):
        """Test CSV with extra columns beyond OHLC."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume", "extra"])
            writer.writerow(["2023-01-01", "99.5", "101.0", "99.0", "100.0", "1000000", "ignore"])
            writer.writerow(["2023-01-02", "100.5", "103.0", "100.0", "102.0", "1200000", "ignore"])
            csv_path = f.name

        try:
            data = _load_csv(csv_path)

            # Should load successfully, ignoring extra columns
            assert len(data["close"]) == 2
            np.testing.assert_array_equal(data["close"], [100.0, 102.0])

        finally:
            Path(csv_path).unlink()


class TestSummarise:
    """Tests for performance summary functionality."""

    def test_basic_summarise(self):
        """Test basic performance summary."""
        # Create sample daily P&L data
        daily_pnl = np.array([0.01, -0.005, 0.015, 0.002, -0.01, 0.008])

        summary = _summarise(daily_pnl, trading_days=252)

        # Should contain expected metrics
        assert "Ann μ=" in summary
        assert "Ann σ=" in summary
        assert "SR=" in summary
        assert f"n={len(daily_pnl)}" in summary

    def test_summarise_empty_data(self):
        """Test summary with empty data."""
        summary = _summarise(np.array([]))
        assert summary == "No data to summarize"

    def test_summarise_all_nan_data(self):
        """Test summary with all NaN data."""
        summary = _summarise(np.array([np.nan, np.nan, np.nan]))
        assert summary == "No valid data to summarize"

    def test_summarise_with_nans(self):
        """Test summary with some NaN values."""
        daily_pnl = np.array([0.01, np.nan, 0.015, np.nan, -0.01])
        summary = _summarise(daily_pnl)

        # Should only count valid observations
        assert "n=3" in summary

    def test_summarise_zero_volatility(self):
        """Test summary with zero volatility."""
        daily_pnl = np.array([0.01, 0.01, 0.01, 0.01])  # Constant returns
        summary = _summarise(daily_pnl)

        # Should handle zero volatility gracefully
        assert "SR=nan" in summary or "SR=inf" in summary

    def test_summarise_custom_trading_days(self):
        """Test summary with custom trading days."""
        daily_pnl = np.array([0.01, -0.005, 0.015])

        # Test with different trading day conventions
        summary_252 = _summarise(daily_pnl, trading_days=252)
        summary_365 = _summarise(daily_pnl, trading_days=365)

        # Results should be different due to different annualization
        assert summary_252 != summary_365


class TestCLICommands:
    """Tests for CLI command functions."""

    def setup_method(self):
        """Create test CSV file for CLI commands."""
        self.test_csv = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        writer = csv.writer(self.test_csv)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])

        # Generate sample data
        np.random.seed(42)
        n = 200
        base_price = 100.0

        for i in range(n):
            date = f"2023-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
            price_change = 0.01 * np.random.randn()
            base_price *= 1 + price_change

            open_price = base_price * (1 + 0.001 * np.random.randn())
            high_price = base_price * (1 + 0.005 * abs(np.random.randn()))
            low_price = base_price * (1 - 0.005 * abs(np.random.randn()))
            close_price = base_price
            volume = int(1000000 * (1 + 0.2 * np.random.randn()))

            writer.writerow(
                [
                    date,
                    f"{open_price:.2f}",
                    f"{high_price:.2f}",
                    f"{low_price:.2f}",
                    f"{close_price:.2f}",
                    volume,
                ]
            )

        self.test_csv.close()
        self.csv_path = self.test_csv.name

    def teardown_method(self):
        """Clean up test files."""
        Path(self.csv_path).unlink()

        # Clean up any result files
        for pattern in ["*_results.npz"]:
            for file in Path(".").glob(pattern):
                file.unlink(missing_ok=True)

    def test_cmd_european(self):
        """Test European TF command."""
        args = argparse.Namespace(
            csv=self.csv_path,
            target=0.15,
            a=252,
            span_sigma=20,
            span_long=50,
            span_short=10,
            longshort=True,
        )

        # Capture stdout
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            result = cmd_european(args)

        assert result == 0  # Success
        output = captured_output.getvalue()
        assert "European TF Results:" in output
        assert "Results saved to european_results.npz" in output

        # Check that results file was created
        assert Path("european_results.npz").exists()

    def test_cmd_american(self):
        """Test American TF command."""
        args = argparse.Namespace(
            csv=self.csv_path,
            a=252,
            span_long=50,
            span_short=10,
            atr_period=20,
            q=2.0,
            p=3.0,
            r_multiple=0.01,
        )

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            result = cmd_american(args)

        assert result == 0
        output = captured_output.getvalue()
        assert "American TF Results:" in output
        assert "Results saved to american_results.npz" in output

        assert Path("american_results.npz").exists()

    def test_cmd_tsmom(self):
        """Test TSMOM command."""
        args = argparse.Namespace(csv=self.csv_path, target=0.12, a=252, span_sigma=20, L=5, M=8)

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            result = cmd_tsmom(args)

        assert result == 0
        output = captured_output.getvalue()
        assert "TSMOM Results:" in output
        assert "Results saved to tsmom_results.npz" in output

        assert Path("tsmom_results.npz").exists()

    def test_cmd_with_invalid_csv(self):
        """Test commands with invalid CSV file."""
        args = argparse.Namespace(
            csv="nonexistent.csv",
            target=0.15,
            a=252,
            span_sigma=20,
            span_long=50,
            span_short=10,
            longshort=False,
        )

        captured_error = io.StringIO()
        with patch("sys.stderr", captured_error):
            result = cmd_european(args)

        assert result == 1  # Error
        error_output = captured_error.getvalue()
        assert "Error running European TF:" in error_output

    def test_cmd_with_invalid_parameters(self):
        """Test commands with invalid parameters."""
        # European TF with invalid span relationship
        args = argparse.Namespace(
            csv=self.csv_path,
            target=0.15,
            a=252,
            span_sigma=20,
            span_long=10,  # Less than span_short
            span_short=20,
            longshort=True,  # This should cause validation error
        )

        captured_error = io.StringIO()
        with patch("sys.stderr", captured_error):
            result = cmd_european(args)

        assert result == 1


class TestMainFunction:
    """Tests for main CLI entry point."""

    def test_main_help(self):
        """Test main function with help argument."""
        with patch("sys.argv", ["tfu", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Help should exit with code 0
            assert exc_info.value.code == 0

    def test_main_no_arguments(self):
        """Test main function with no arguments."""
        with patch("sys.argv", ["tfu"]):
            with patch("sys.stderr", io.StringIO()):
                result = main()
            # Should fail without required subcommand
            assert result == 1

    def test_main_european_subcommand(self):
        """Test main function with European TF subcommand."""
        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close"])
            writer.writerow(["2023-01-01", "100.0"])
            writer.writerow(["2023-01-02", "101.0"])
            csv_path = f.name

        try:
            with patch("sys.argv", ["tfu", "european", "--csv", csv_path]):
                with patch("sys.stdout", io.StringIO()):
                    result = main()

            assert result == 0

        finally:
            Path(csv_path).unlink()
            # Clean up result files
            for file in Path(".").glob("*_results.npz"):
                file.unlink(missing_ok=True)

    def test_main_keyboard_interrupt(self):
        """Test main function with keyboard interrupt."""

        def mock_function(args):
            raise KeyboardInterrupt()

        with patch("sys.argv", ["tfu", "european", "--csv", "dummy.csv"]):
            with patch("tfunify.cli.cmd_european", mock_function):
                with patch("sys.stderr", io.StringIO()) as stderr:
                    result = main()

        assert result == 130  # Standard code for keyboard interrupt
        assert "Operation cancelled by user" in stderr.getvalue()

    def test_main_unexpected_error(self):
        """Test main function with unexpected error."""

        def mock_function(args):
            raise RuntimeError("Unexpected error")

        with patch("sys.argv", ["tfu", "european", "--csv", "dummy.csv"]):
            with patch("tfunify.cli.cmd_european", mock_function):
                with patch("sys.stderr", io.StringIO()) as stderr:
                    result = main()

        assert result == 1
        assert "Unexpected error:" in stderr.getvalue()


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    def test_full_workflow_european(self):
        """Test complete workflow for European TF."""
        # Create realistic test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close", "high", "low"])

            np.random.seed(42)
            base_price = 100.0
            for i in range(100):
                date = f"2023-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
                base_price *= 1 + 0.01 * np.random.randn()
                high = base_price * 1.01
                low = base_price * 0.99
                writer.writerow([date, f"{base_price:.2f}", f"{high:.2f}", f"{low:.2f}"])

            csv_path = f.name

        try:
            # Test European TF with various parameters
            test_args = [
                ["tfu", "european", "--csv", csv_path, "--target", "0.10"],
                ["tfu", "european", "--csv", csv_path, "--longshort", "--span-long", "50"],
                ["tfu", "european", "--csv", csv_path, "--span-sigma", "15"],
            ]

            for args in test_args:
                with patch("sys.argv", args):
                    with patch("sys.stdout", io.StringIO()):
                        result = main()
                    assert result == 0

        finally:
            Path(csv_path).unlink()
            # Clean up all result files
            for file in Path(".").glob("*_results.npz"):
                file.unlink(missing_ok=True)

    def test_cli_error_handling(self):
        """Test CLI error handling with various error conditions."""
        error_conditions = [
            # Missing CSV file
            ["tfu", "european", "--csv", "missing.csv"],
            # Invalid parameters
            ["tfu", "american", "--csv", "dummy.csv", "--q", "-1.0"],
            ["tfu", "tsmom", "--csv", "dummy.csv", "--L", "0"],
        ]

        for args in error_conditions:
            with patch("sys.argv", args):
                with patch("sys.stderr", io.StringIO()):
                    result = main()
                assert result != 0  # Should indicate error

    @patch("tfunify.cli._download_csv_yahoo")
    def test_download_command_mock(self, mock_download):
        """Test download command with mocked yfinance."""
        mock_download.return_value = "test_data.csv"

        with patch("sys.argv", ["tfu", "download", "SPY", "--out", "test_data.csv"]):
            with patch("sys.stdout", io.StringIO()) as stdout:
                result = main()

        assert result == 0
        assert "Successfully saved" in stdout.getvalue()
        mock_download.assert_called_once()

    def test_all_systems_same_data(self):
        """Test that all three systems can run on the same dataset."""
        # Create comprehensive test data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])

            np.random.seed(123)
            base_price = 100.0
            for i in range(500):  # Enough data for TSMOM
                date = f"2023-{(i // 30) + 1:02d}-{(i % 30) + 1:02d}"
                base_price *= 1 + 0.005 * np.random.randn()

                open_p = base_price * (1 + 0.001 * np.random.randn())
                high_p = base_price * (1 + 0.01 * abs(np.random.randn()))
                low_p = base_price * (1 - 0.01 * abs(np.random.randn()))
                close_p = base_price
                volume = int(1000000 * (1 + 0.3 * np.random.randn()))

                writer.writerow(
                    [
                        date,
                        f"{open_p:.2f}",
                        f"{high_p:.2f}",
                        f"{low_p:.2f}",
                        f"{close_p:.2f}",
                        volume,
                    ]
                )

            csv_path = f.name

        try:
            # Test all three systems
            systems = [
                ["tfu", "european", "--csv", csv_path],
                ["tfu", "american", "--csv", csv_path],
                ["tfu", "tsmom", "--csv", csv_path],
            ]

            for system_args in systems:
                with patch("sys.argv", system_args):
                    with patch("sys.stdout", io.StringIO()):
                        result = main()
                    assert result == 0

        finally:
            Path(csv_path).unlink()
            # Clean up all result files
            for file in Path(".").glob("*_results.npz"):
                file.unlink(missing_ok=True)
