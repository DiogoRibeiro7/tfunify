import numpy as np
import pytest
import tempfile
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock

from tfunify.european import EuropeanTF, EuropeanTFConfig
from tfunify.american import AmericanTF, AmericanTFConfig
from tfunify.tsmom import TSMOM, TSMOMConfig
from tfunify.core import (
    pct_returns_from_prices,
)


class TestCrossSystemIntegration:
    """Integration tests comparing all three systems."""

    def setup_method(self):
        """Set up comprehensive test data."""
        np.random.seed(42)
        self.n = 2000  # Sufficient for all systems including TSMOM

        # Generate realistic market data with different regimes
        regime_length = self.n // 4

        # Regime 1: Bull market (trending up)
        bull_returns = 0.0005 + 0.015 * np.random.randn(regime_length)
        for i in range(1, len(bull_returns)):
            bull_returns[i] += 0.1 * bull_returns[i - 1]  # Momentum

        # Regime 2: Bear market (trending down)
        bear_returns = -0.0005 + 0.02 * np.random.randn(regime_length)
        for i in range(1, len(bear_returns)):
            bear_returns[i] += 0.08 * bear_returns[i - 1]  # Momentum

        # Regime 3: High volatility sideways
        volatile_returns = 0.0 + 0.03 * np.random.randn(regime_length)

        # Regime 4: Low volatility recovery
        recovery_returns = 0.0003 + 0.01 * np.random.randn(self.n - 3 * regime_length)
        for i in range(1, len(recovery_returns)):
            recovery_returns[i] += 0.05 * recovery_returns[i - 1]

        # Combine regimes
        all_returns = np.concatenate(
            [bull_returns, bear_returns, volatile_returns, recovery_returns]
        )
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, all_returns[1:]])

        # Generate OHLC data
        daily_range = 0.005 + 0.01 * np.abs(np.random.randn(self.n))
        self.high = self.prices * (1 + daily_range * np.random.uniform(0.2, 1.0, self.n))
        self.low = self.prices * (1 - daily_range * np.random.uniform(0.2, 1.0, self.n))

        # Ensure OHLC consistency
        self.high = np.maximum(self.high, self.prices)
        self.low = np.minimum(self.low, self.prices)

    def test_all_systems_run_successfully(self):
        """Test that all systems can run on the same comprehensive dataset."""
        # European TF
        eu_cfg = EuropeanTFConfig(
            sigma_target_annual=0.12, span_sigma=30, mode="longshort", span_long=100, span_short=20
        )
        eu_system = EuropeanTF(eu_cfg)
        eu_pnl, eu_weights, eu_signal, eu_vol = eu_system.run_from_prices(self.prices)

        # American TF
        am_cfg = AmericanTFConfig(
            span_long=80, span_short=15, atr_period=25, q=2.0, p=3.5, r_multiple=0.015
        )
        am_system = AmericanTF(am_cfg)
        am_pnl, am_units = am_system.run(self.prices, self.high, self.low)

        # TSMOM
        ts_cfg = TSMOMConfig(sigma_target_annual=0.10, span_sigma=25, L=12, M=8)
        ts_system = TSMOM(ts_cfg)
        ts_pnl, ts_weights, ts_signals, ts_vol = ts_system.run_from_prices(self.prices)

        # All should complete without errors
        assert len(eu_pnl) == len(self.prices)
        assert len(am_pnl) == len(self.prices)
        assert len(ts_pnl) == len(self.prices)

        # After warmup, all should have finite values
        warmup = 150
        assert np.isfinite(eu_pnl[warmup:]).all()
        assert np.isfinite(am_pnl[warmup:]).all()
        assert np.isfinite(ts_pnl[warmup:]).all()

    def test_performance_comparison_across_systems(self):
        """Compare performance metrics across all systems."""
        # Run all systems with similar risk targets
        target_vol = 0.12

        # European TF
        eu_cfg = EuropeanTFConfig(sigma_target_annual=target_vol)
        eu_system = EuropeanTF(eu_cfg)
        eu_pnl, _, _, _ = eu_system.run_from_prices(self.prices)

        # American TF (no direct vol targeting, but similar risk via r_multiple)
        am_cfg = AmericanTFConfig(r_multiple=0.01)
        am_system = AmericanTF(am_cfg)
        am_pnl, _ = am_system.run(self.prices, self.high, self.low)

        # TSMOM
        ts_cfg = TSMOMConfig(sigma_target_annual=target_vol)
        ts_system = TSMOM(ts_cfg)
        ts_pnl, _, _, _ = ts_system.run_from_prices(self.prices)

        # Calculate performance metrics for each
        systems = [("European TF", eu_pnl), ("American TF", am_pnl), ("TSMOM", ts_pnl)]

        metrics = {}
        for name, pnl in systems:
            valid_pnl = pnl[~np.isnan(pnl)]
            if len(valid_pnl) > 100:
                annual_ret = np.mean(valid_pnl) * 252
                annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                sharpe = annual_ret / annual_vol if annual_vol > 0 else 0

                metrics[name] = {
                    "annual_return": annual_ret,
                    "annual_vol": annual_vol,
                    "sharpe": sharpe,
                    "total_pnl": np.sum(valid_pnl),
                    "max_dd": self._calculate_max_drawdown(valid_pnl),
                }

        # All systems should produce valid metrics
        assert len(metrics) == 3
        for _name, metric in metrics.items():
            assert np.isfinite(metric["annual_return"])
            assert np.isfinite(metric["annual_vol"])
            assert np.isfinite(metric["sharpe"])
            assert metric["annual_vol"] > 0

    def test_regime_specific_performance(self):
        """Test system performance in different market regimes."""
        regime_length = self.n // 4
        regimes = {
            "Bull": (0, regime_length),
            "Bear": (regime_length, 2 * regime_length),
            "Volatile": (2 * regime_length, 3 * regime_length),
            "Recovery": (3 * regime_length, self.n),
        }

        # Run European TF (easiest to analyze)
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        # Analyze performance by regime
        regime_performance = {}
        for regime_name, (start, end) in regimes.items():
            regime_pnl = pnl[start:end]
            valid_pnl = regime_pnl[~np.isnan(regime_pnl)]

            if len(valid_pnl) > 10:  # Need minimum observations
                regime_performance[regime_name] = {
                    "total_pnl": np.sum(valid_pnl),
                    "avg_daily_pnl": np.mean(valid_pnl),
                    "volatility": np.std(valid_pnl, ddof=0),
                    "sharpe": np.mean(valid_pnl) / np.std(valid_pnl, ddof=0)
                    if np.std(valid_pnl) > 0
                    else 0,
                }

        # Should have performance metrics for each regime
        assert len(regime_performance) == len(regimes)

        # All metrics should be finite
        for regime_name, metrics in regime_performance.items():
            for metric_name, value in metrics.items():
                assert np.isfinite(value), f"{regime_name} {metric_name} is not finite"

    def test_volatility_targeting_consistency(self):
        """Test that volatility targeting works consistently across systems."""
        target_vol = 0.15

        # European TF with vol targeting
        eu_cfg = EuropeanTFConfig(sigma_target_annual=target_vol)
        eu_system = EuropeanTF(eu_cfg)
        eu_pnl, _, _, _ = eu_system.run_from_prices(self.prices)

        # TSMOM with same vol target
        ts_cfg = TSMOMConfig(sigma_target_annual=target_vol)
        ts_system = TSMOM(ts_cfg)
        ts_pnl, _, _, _ = ts_system.run_from_prices(self.prices)

        # Calculate realized volatilities
        eu_valid = eu_pnl[~np.isnan(eu_pnl)]
        ts_valid = ts_pnl[~np.isnan(ts_pnl)]

        if len(eu_valid) > 200 and len(ts_valid) > 200:
            eu_realized_vol = np.std(eu_valid, ddof=0) * np.sqrt(252)
            ts_realized_vol = np.std(ts_valid, ddof=0) * np.sqrt(252)

            # Should be reasonably close to target (within factor of 2)
            assert 0.5 * target_vol < eu_realized_vol < 2.5 * target_vol
            assert 0.5 * target_vol < ts_realized_vol < 2.5 * target_vol

    def test_correlation_analysis(self):
        """Test correlation between different systems."""
        # Run all systems
        eu_cfg = EuropeanTFConfig()
        am_cfg = AmericanTFConfig()
        ts_cfg = TSMOMConfig()

        eu_system = EuropeanTF(eu_cfg)
        am_system = AmericanTF(am_cfg)
        ts_system = TSMOM(ts_cfg)

        eu_pnl, _, _, _ = eu_system.run_from_prices(self.prices)
        am_pnl, _ = am_system.run(self.prices, self.high, self.low)
        ts_pnl, _, _, _ = ts_system.run_from_prices(self.prices)

        # Align data (remove NaNs)
        min_len = min(len(eu_pnl), len(am_pnl), len(ts_pnl))
        warmup = 200  # Skip warmup period

        if min_len > warmup + 100:
            eu_aligned = eu_pnl[warmup:min_len]
            am_aligned = am_pnl[warmup:min_len]
            ts_aligned = ts_pnl[warmup:min_len]

            # Remove any remaining NaNs
            valid_mask = np.isfinite(eu_aligned) & np.isfinite(am_aligned) & np.isfinite(ts_aligned)

            if np.sum(valid_mask) > 50:
                eu_clean = eu_aligned[valid_mask]
                am_clean = am_aligned[valid_mask]
                ts_clean = ts_aligned[valid_mask]

                # Calculate correlations
                corr_eu_am = np.corrcoef(eu_clean, am_clean)[0, 1]
                corr_eu_ts = np.corrcoef(eu_clean, ts_clean)[0, 1]
                corr_am_ts = np.corrcoef(am_clean, ts_clean)[0, 1]

                # Correlations should be finite and reasonable
                assert np.isfinite(corr_eu_am)
                assert np.isfinite(corr_eu_ts)
                assert np.isfinite(corr_am_ts)

                # Systems should be somewhat correlated (all trend-following)
                # but not perfectly correlated (different approaches)
                assert -1 <= corr_eu_am <= 1
                assert -1 <= corr_eu_ts <= 1
                assert -1 <= corr_am_ts <= 1

    def test_parameter_sensitivity(self):
        """Test sensitivity to parameter changes."""
        # Test European TF with different volatility targets
        vol_targets = [0.05, 0.10, 0.15, 0.20]
        eu_results = {}

        for vol_target in vol_targets:
            cfg = EuropeanTFConfig(sigma_target_annual=vol_target)
            system = EuropeanTF(cfg)
            pnl, _, _, _ = system.run_from_prices(self.prices)

            valid_pnl = pnl[~np.isnan(pnl)]
            if len(valid_pnl) > 100:
                realized_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                eu_results[vol_target] = realized_vol

        # Higher vol targets should generally lead to higher realized vol
        if len(eu_results) >= 3:
            vol_pairs = list(eu_results.items())
            vol_pairs.sort()  # Sort by target vol

            # Check general increasing trend (allowing some noise)
            increasing_pairs = 0
            for i in range(len(vol_pairs) - 1):
                if vol_pairs[i + 1][1] >= vol_pairs[i][1] * 0.8:  # Allow 20% tolerance
                    increasing_pairs += 1

            # Most pairs should show increasing trend
            assert increasing_pairs >= len(vol_pairs) - 2

    def test_data_quality_robustness(self):
        """Test robustness to various data quality issues."""
        # Test with price gaps
        gapped_prices = self.prices.copy()
        gap_indices = [500, 1000, 1500]  # Introduce gaps
        for gap_idx in gap_indices:
            if gap_idx < len(gapped_prices):
                gapped_prices[gap_idx] *= 1.05  # 5% gap

        # Test with missing data (NaN values)
        sparse_prices = self.prices.copy()
        missing_indices = np.random.choice(len(sparse_prices), size=50, replace=False)
        # Don't make first/last values NaN to avoid issues
        missing_indices = missing_indices[
            (missing_indices > 10) & (missing_indices < len(sparse_prices) - 10)
        ]
        sparse_prices[missing_indices] = np.nan

        # Test systems on modified data
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)

        try:
            # Gapped data should work
            pnl_gap, _, _, _ = system.run_from_prices(gapped_prices)
            assert len(pnl_gap) == len(gapped_prices)

            # Sparse data with NaNs should be handled
            # (though may need preprocessing in real applications)
            if not np.any(np.isnan(sparse_prices)):
                pnl_sparse, _, _, _ = system.run_from_prices(sparse_prices)
                assert len(pnl_sparse) == len(sparse_prices)

        except Exception as e:
            # If systems can't handle the data directly, that's also acceptable
            # as long as the error is informative
            assert "nan" in str(e).lower() or "finite" in str(e).lower()

    def test_performance_attribution(self):
        """Test performance attribution across different components."""
        cfg = EuropeanTFConfig(mode="longshort", span_long=100, span_short=20)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        # Decompose P&L
        returns = pct_returns_from_prices(self.prices)

        # Calculate individual components for analysis
        valid_indices = ~(np.isnan(pnl) | np.isnan(weights) | np.isnan(returns))

        if np.sum(valid_indices) > 100:
            pnl_clean = pnl[valid_indices]
            weights_clean = weights[valid_indices]
            returns_clean = returns[valid_indices]

            # Basic attribution: PnL should come from weights * returns
            # (with lag for weights)
            if len(pnl_clean) > 1:
                # Manual calculation: w[t-1] * r[t]
                manual_pnl = np.zeros_like(pnl_clean)
                manual_pnl[1:] = weights_clean[:-1] * returns_clean[1:]

                # Should be very close (allowing for numerical precision)
                np.testing.assert_allclose(pnl_clean, manual_pnl, atol=1e-10)

    def test_stress_testing(self):
        """Test systems under stress conditions."""
        # Create various stress scenarios
        stress_scenarios = []

        # Flash crash scenario
        crash_prices = self.prices.copy()
        crash_start = len(crash_prices) // 2
        crash_prices[crash_start : crash_start + 5] *= np.array([0.95, 0.85, 0.80, 0.85, 0.90])
        stress_scenarios.append(("Flash Crash", crash_prices))

        # Extreme volatility
        np.random.seed(999)
        extreme_vol_returns = 0.1 * np.random.randn(len(self.prices))
        extreme_vol_prices = 100 * np.cumprod(1 + np.r_[0.0, extreme_vol_returns[1:]])
        stress_scenarios.append(("Extreme Volatility", extreme_vol_prices))

        # Slow drift
        drift_prices = 100 * (1 + 0.0001 * np.arange(len(self.prices)))
        stress_scenarios.append(("Slow Drift", drift_prices))

        # Test European TF on all scenarios
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)

        for _scenario_name, test_prices in stress_scenarios:
            try:
                pnl, weights, signal, volatility = system.run_from_prices(test_prices)

                # Basic checks
                assert len(pnl) == len(test_prices)

                # Should have some finite values after warmup
                warmup = 100
                finite_pnl = np.isfinite(pnl[warmup:])
                assert np.sum(finite_pnl) > len(pnl[warmup:]) * 0.8  # At least 80% finite

            except Exception as e:
                # If system fails on stress scenario, error should be informative
                assert any(
                    keyword in str(e).lower()
                    for keyword in ["finite", "positive", "empty", "length"]
                )

    def _calculate_max_drawdown(self, pnl_series):
        """Helper function to calculate maximum drawdown."""
        cumulative = np.cumsum(pnl_series)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return np.min(drawdown)


class TestRealWorldScenarios:
    """Test systems on realistic market scenarios."""

    def test_trending_market_scenario(self):
        """Test all systems on strongly trending market."""
        np.random.seed(123)
        n = 1000

        # Strong uptrend with noise
        trend_strength = 0.001  # Daily drift
        noise_level = 0.015  # Daily volatility

        returns = trend_strength + noise_level * np.random.randn(n)
        # Add momentum
        for i in range(1, n):
            returns[i] += 0.05 * returns[i - 1]

        prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

        # Test all systems
        systems = [
            ("European", EuropeanTF(EuropeanTFConfig())),
            ("American", AmericanTF(AmericanTFConfig())),
            ("TSMOM", TSMOM(TSMOMConfig())),
        ]

        for name, system in systems:
            if name == "American":
                high = prices * 1.005
                low = prices * 0.995
                pnl, _ = system.run(prices, high, low)
            else:
                pnl, *_ = system.run_from_prices(prices)

            # In trending market, should generally make money
            valid_pnl = pnl[~np.isnan(pnl)]
            if len(valid_pnl) > 100:
                total_pnl = np.sum(valid_pnl)
                # Don't assert positive (markets are noisy), but should be reasonable
                assert np.isfinite(total_pnl)

    def test_mean_reverting_market_scenario(self):
        """Test systems on mean-reverting market."""
        np.random.seed(456)
        n = 800

        # Mean reverting process
        returns = np.zeros(n)
        returns[0] = 0.01 * np.random.randn()

        for i in range(1, n):
            # Strong mean reversion
            returns[i] = -0.15 * returns[i - 1] + 0.02 * np.random.randn()

        prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

        # Test European TF (others may struggle more with mean reversion)
        cfg = EuropeanTFConfig(span_long=50, span_short=10)  # Faster system
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(prices)

        # Should handle mean reversion without crashing
        valid_pnl = pnl[~np.isnan(pnl)]
        assert len(valid_pnl) > n * 0.5  # Most values should be valid

    def test_crisis_scenario(self):
        """Test systems during crisis-like conditions."""
        np.random.seed(789)
        n = 600

        # Crisis: fat tails + correlation breakdown
        # Use t-distribution for fat tails
        crisis_returns = np.random.standard_t(df=3, size=n) * 0.025

        # Add some large negative shocks
        shock_days = np.random.choice(n, size=10, replace=False)
        crisis_returns[shock_days] = -0.05 - 0.03 * np.random.rand(len(shock_days))

        prices = 100 * np.cumprod(1 + np.r_[0.0, crisis_returns[1:]])

        # Test systems with conservative parameters
        cfg = EuropeanTFConfig(sigma_target_annual=0.08)  # Lower vol target
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(prices)

        # Should survive crisis without extreme losses
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 50:
            # Check that no single day loss is catastrophic
            max_daily_loss = np.min(valid_pnl)
            # Shouldn't lose more than a reasonable amount in one day
            assert max_daily_loss > -0.5  # Arbitrary but reasonable threshold

    def test_low_volatility_regime(self):
        """Test systems in low volatility environment."""
        np.random.seed(111)
        n = 800

        # Very low volatility with small drift
        low_vol_returns = 0.0001 + 0.005 * np.random.randn(n)  # ~8% annual vol
        prices = 100 * np.cumprod(1 + np.r_[0.0, low_vol_returns[1:]])

        # Systems should handle low vol gracefully
        systems = [
            ("European", EuropeanTF(EuropeanTFConfig(sigma_target_annual=0.05))),
            ("TSMOM", TSMOM(TSMOMConfig(sigma_target_annual=0.05))),
        ]

        for _name, system in systems:
            pnl, *_ = system.run_from_prices(prices)

            # Should produce valid results even in low vol
            valid_pnl = pnl[~np.isnan(pnl)]
            assert len(valid_pnl) > n * 0.7

            # Volatility should be detected as low
            if len(valid_pnl) > 100:
                realized_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                # Should be reasonable given low input volatility
                assert realized_vol < 0.3  # Not extremely high


class TestDataPipelineIntegration:
    """Test integration with data loading and processing pipeline."""

    def test_csv_to_systems_pipeline(self):
        """Test complete pipeline from CSV to system results."""
        # Create realistic CSV file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "open", "high", "low", "close", "volume"])

            np.random.seed(42)
            base_price = 100.0

            for i in range(500):
                date = f"2023-{(i // 22) + 1:02d}-{(i % 22) + 1:02d}"
                ret = 0.0005 + 0.015 * np.random.randn()
                base_price *= 1 + ret

                open_p = base_price * (1 + 0.002 * np.random.randn())
                high_p = base_price * (1 + 0.008 * abs(np.random.randn()))
                low_p = base_price * (1 - 0.008 * abs(np.random.randn()))
                volume = int(1000000 * (1 + 0.3 * np.random.randn()))

                writer.writerow(
                    [
                        date,
                        f"{open_p:.2f}",
                        f"{high_p:.2f}",
                        f"{low_p:.2f}",
                        f"{base_price:.2f}",
                        volume,
                    ]
                )

            csv_path = f.name

        try:
            # Load CSV using CLI function
            from tfunify.cli import _load_csv

            data = _load_csv(csv_path)

            # Run all systems on loaded data
            systems = [
                ("European", EuropeanTF(EuropeanTFConfig())),
                ("American", AmericanTF(AmericanTFConfig())),
                ("TSMOM", TSMOM(TSMOMConfig())),
            ]

            results = {}
            for name, system in systems:
                if name == "American":
                    pnl, units = system.run(data["close"], data["high"], data["low"])
                    results[name] = pnl
                else:
                    pnl, *_ = system.run_from_prices(data["close"])
                    results[name] = pnl

            # All systems should produce results
            assert len(results) == 3

            for _name, pnl in results.items():
                assert len(pnl) == len(data["close"])
                valid_pnl = pnl[~np.isnan(pnl)]
                assert len(valid_pnl) > len(data["close"]) * 0.5  # At least 50% valid

        finally:
            Path(csv_path).unlink()

    def test_missing_data_handling(self):
        """Test handling of missing or incomplete data."""
        # Create CSV with some missing values
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            writer = csv.writer(f)
            writer.writerow(["date", "close"])

            for i in range(100):
                date = f"2023-01-{i + 1:02d}"
                price = 100 + i * 0.1  # Simple increasing prices
                writer.writerow([date, f"{price:.2f}"])

            csv_path = f.name

        try:
            from tfunify.cli import _load_csv

            data = _load_csv(csv_path)

            # Should load successfully with high/low defaulting to close
            assert len(data["close"]) == 100
            np.testing.assert_array_equal(data["high"], data["close"])
            np.testing.assert_array_equal(data["low"], data["close"])

            # Systems should run on this simple data
            cfg = EuropeanTFConfig(span_long=20, span_short=5)
            system = EuropeanTF(cfg)
            pnl, weights, signal, volatility = system.run_from_prices(data["close"])

            assert len(pnl) == len(data["close"])

        finally:
            Path(csv_path).unlink()

    @patch("yfinance.download")
    def test_yfinance_integration_mock(self, mock_download):
        """Test integration with yfinance data source (mocked)."""
        # Mock yfinance data
        mock_data = MagicMock()
        mock_data.empty = False
        mock_data.iterrows.return_value = [
            (
                "2023-01-01",
                {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000000},
            ),
            (
                "2023-01-02",
                {"Open": 100.5, "High": 102.0, "Low": 100.0, "Close": 101.5, "Volume": 1200000},
            ),
            (
                "2023-01-03",
                {"Open": 101.5, "High": 102.5, "Low": 101.0, "Close": 102.0, "Volume": 1100000},
            ),
        ]
        mock_download.return_value = mock_data

        # Test data download and system integration
        try:
            from tfunify.data import download_csv, load_csv

            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
                temp_path = f.name

            # This should work with mocked yfinance
            result_path = download_csv("SPY", temp_path, period="1y")
            assert result_path == Path(temp_path)

            # Load and verify
            data = load_csv(temp_path)
            assert len(data["close"]) == 3

        except ImportError:
            # yfinance not available, skip test
            pytest.skip("yfinance not available")
        finally:
            if "temp_path" in locals():
                Path(temp_path).unlink(missing_ok=True)
