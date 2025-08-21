import numpy as np
import pytest
from tfunify.core import (
    span_to_nu,
    ewma,
    ewma_variance_preserving,
    long_short_variance_preserving,
    pct_returns_from_prices,
    ewma_volatility_from_returns,
    vol_normalised_returns,
    volatility_target_weights,
    volatility_weighted_turnover,
)
from tfunify.european import EuropeanTF, EuropeanTFConfig
from tfunify.american import AmericanTF, AmericanTFConfig
from tfunify.tsmom import TSMOM, TSMOMConfig


class TestCore:
    """Test core functionality."""

    def test_span_to_nu_valid(self):
        """Test span_to_nu with valid inputs."""
        assert 0.0 < span_to_nu(2) < 1.0
        assert span_to_nu(10) == 1.0 - 2.0 / 11.0
        assert span_to_nu(1) == 1.0 - 2.0 / 2.0

    def test_span_to_nu_invalid(self):
        """Test span_to_nu with invalid inputs."""
        with pytest.raises(ValueError):
            span_to_nu(0)
        with pytest.raises(ValueError):
            span_to_nu(-1)
        with pytest.raises(ValueError):
            span_to_nu(1.5)  # Non-integer

    def test_ewma_basic(self):
        """Test basic EWMA functionality."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        nu = 0.5
        result = ewma(x, nu)

        assert len(result) == len(x)
        assert result[0] == x[0]
        assert np.allclose(result[1], 0.5 * x[1] + 0.5 * result[0])

    def test_ewma_empty_array(self):
        """Test EWMA with empty array."""
        with pytest.raises(ValueError, match="Input array cannot be empty"):
            ewma(np.array([]), 0.5)

    def test_ewma_non_finite_values(self):
        """Test EWMA with non-finite values."""
        x = np.array([1.0, np.nan, 3.0])
        with pytest.raises(ValueError, match="Input contains non-finite values"):
            ewma(x, 0.5)

    def test_ewma_invalid_nu(self):
        """Test EWMA with invalid nu."""
        x = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="nu must be in"):
            ewma(x, 0.0)
        with pytest.raises(ValueError, match="nu must be in"):
            ewma(x, 1.0)

    def test_pct_returns_from_prices_valid(self):
        """Test percentage returns calculation."""
        prices = np.array([100.0, 110.0, 99.0, 105.0])
        returns = pct_returns_from_prices(prices)

        expected = np.array([0.0, 0.1, -0.1, 6.0 / 99.0])
        assert np.allclose(returns, expected)

    def test_pct_returns_from_prices_invalid(self):
        """Test percentage returns with invalid inputs."""
        with pytest.raises(ValueError, match="prices must have length >= 2"):
            pct_returns_from_prices(np.array([100.0]))

        with pytest.raises(ValueError, match="prices must be positive"):
            pct_returns_from_prices(np.array([100.0, -50.0]))

    def test_long_short_variance_preserving_invalid_params(self):
        """Test long-short EWMA with invalid parameters."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match="Require 0 < nu_short < nu_long < 1"):
            long_short_variance_preserving(x, 0.3, 0.7)  # Wrong order

    def test_volatility_target_weights_invalid(self):
        """Test volatility target weights with invalid inputs."""
        sigma = np.array([0.01, 0.02, 0.015])

        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            volatility_target_weights(sigma, -0.1, 260)

        with pytest.raises(ValueError, match="a must be positive"):
            volatility_target_weights(sigma, 0.15, -260)


class TestEuropeanTF:
    """Test European TF system."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 500
        returns = 0.0001 + 0.02 * np.random.randn(n)
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            EuropeanTFConfig(sigma_target_annual=-0.1)

        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            EuropeanTFConfig(mode="longshort", span_short=250, span_long=20)

        with pytest.raises(ValueError, match="mode must be"):
            EuropeanTFConfig(mode="invalid")

    def test_single_mode(self):
        """Test European TF in single mode."""
        cfg = EuropeanTFConfig(sigma_target_annual=0.15, span_sigma=20, mode="single", span_long=50)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        assert len(pnl) == len(self.prices)
        assert np.isfinite(pnl[10:]).all()
        assert np.isfinite(weights[10:]).all()
        assert np.isfinite(signal[10:]).all()
        assert np.isfinite(volatility[10:]).all()

    def test_longshort_mode(self):
        """Test European TF in long-short mode."""
        cfg = EuropeanTFConfig(
            sigma_target_annual=0.15, span_sigma=20, mode="longshort", span_long=100, span_short=10
        )
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        assert len(pnl) == len(self.prices)
        assert np.isfinite(pnl[10:]).all()

    def test_empty_returns(self):
        """Test with empty returns array."""
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)

        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            system.run_from_returns(np.array([]))


class TestAmericanTF:
    """Test American TF system."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 500
        self.close = 100 + np.cumsum(0.01 * np.random.randn(n))
        self.high = self.close * (1 + 0.005 * np.abs(np.random.randn(n)))
        self.low = self.close * (1 - 0.005 * np.abs(np.random.randn(n)))

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            AmericanTFConfig(span_short=100, span_long=50)

        with pytest.raises(ValueError, match="q must be positive"):
            AmericanTFConfig(q=-1.0)

        with pytest.raises(ValueError, match="atr_period must be >= 1"):
            AmericanTFConfig(atr_period=0)

    def test_basic_run(self):
        """Test basic American TF run."""
        cfg = AmericanTFConfig(span_long=50, span_short=10, atr_period=20, q=2.0, p=3.0)
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        assert len(pnl) == len(self.close)
        assert len(units) == len(self.close)
        assert np.isfinite(pnl[30:]).all()  # Allow for ATR warmup
        assert np.isfinite(units[30:]).all()

    def test_close_only(self):
        """Test American TF with only close prices."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close)  # No high/low provided

        assert len(pnl) == len(self.close)
        assert np.isfinite(pnl[50:]).all()  # Allow for longer warmup

    def test_empty_close(self):
        """Test with empty close array."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)

        with pytest.raises(ValueError, match="Close prices cannot be empty"):
            system.run(np.array([]))

    def test_mismatched_arrays(self):
        """Test with mismatched array sizes."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)

        with pytest.raises(ValueError, match="high, low, and close must have same shape"):
            system.run(self.close, self.high[:-10], self.low)


class TestTSMOM:
    """Test TSMOM system."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        n = 1000  # Need more data for TSMOM blocks
        returns = 0.0001 + 0.02 * np.random.randn(n)
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

    def test_config_validation(self):
        """Test configuration validation."""
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            TSMOMConfig(sigma_target_annual=-0.1)

        with pytest.raises(ValueError, match="L .* must be >= 1"):
            TSMOMConfig(L=0)

        with pytest.raises(ValueError, match="M .* must be >= 1"):
            TSMOMConfig(M=0)

    def test_basic_run(self):
        """Test basic TSMOM run."""
        cfg = TSMOMConfig(sigma_target_annual=0.15, span_sigma=20, L=5, M=8)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)

        assert len(pnl) == len(self.prices)
        assert len(weights) == len(self.prices)
        assert len(signal_grid) == len(self.prices)
        assert len(volatility) == len(self.prices)

        # Check that some signals were generated
        assert np.any(signal_grid != 0)

    def test_insufficient_data(self):
        """Test with insufficient data for block structure."""
        cfg = TSMOMConfig(L=10, M=10)  # Needs at least 100 observations
        system = TSMOM(cfg)
        short_prices = self.prices[:50]  # Too short

        with pytest.raises(ValueError, match="Returns array too short"):
            system.run_from_prices(short_prices)

    def test_empty_prices(self):
        """Test with empty price array."""
        cfg = TSMOMConfig()
        system = TSMOM(cfg)

        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            system.run_from_prices(np.array([]))


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_price_point(self):
        """Test systems with minimal data."""
        # This should fail for all systems that need returns
        single_price = np.array([100.0])

        with pytest.raises(ValueError):
            pct_returns_from_prices(single_price)

    def test_constant_prices(self):
        """Test with constant price series."""
        constant_prices = np.full(100, 100.0)
        returns = pct_returns_from_prices(constant_prices)

        # All returns should be zero except the first
        assert returns[0] == 0.0
        assert np.allclose(returns[1:], 0.0)

        # European TF should handle this gracefully
        cfg = EuropeanTFConfig(span_long=20, span_short=5)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(constant_prices)

        # Should not raise errors, but PnL should be minimal
        assert np.isfinite(pnl).all()

    def test_extreme_volatility(self):
        """Test with extremely volatile price series."""
        np.random.seed(42)
        n = 200
        # Very high volatility returns
        extreme_returns = 0.1 * np.random.randn(n)
        extreme_prices = 100 * np.cumprod(1 + np.r_[0.0, extreme_returns[1:]])

        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(extreme_prices)

        # Should handle extreme cases without crashing
        assert np.isfinite(pnl[20:]).all()
        assert np.all(volatility[20:] > 0)  # Volatility should be detected


class TestIntegration:
    """Integration tests combining multiple components."""

    def setup_method(self):
        """Set up realistic test data."""
        np.random.seed(123)
        n = 1000
        # Create trending price series with noise
        trend = 0.0005 * np.arange(n)
        noise = 0.02 * np.random.randn(n)
        returns = trend + noise
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

    def test_all_systems_run(self):
        """Test that all three systems can run on the same data."""
        # European TF
        eu_cfg = EuropeanTFConfig(span_long=50, span_short=10)
        eu_system = EuropeanTF(eu_cfg)
        eu_pnl, _, _, _ = eu_system.run_from_prices(self.prices)

        # American TF
        am_cfg = AmericanTFConfig(span_long=50, span_short=10, atr_period=20)
        am_system = AmericanTF(am_cfg)
        am_pnl, _ = am_system.run(self.prices)

        # TSMOM (need sufficient data)
        ts_cfg = TSMOMConfig(L=5, M=8)
        ts_system = TSMOM(ts_cfg)
        ts_pnl, _, _, _ = ts_system.run_from_prices(self.prices)

        # All should produce valid results
        assert np.isfinite(eu_pnl[30:]).all()
        assert np.isfinite(am_pnl[30:]).all()
        assert np.isfinite(ts_pnl[30:]).all()

        # Should have some variation (not all zeros)
        assert np.std(eu_pnl[30:]) > 0
        assert np.std(am_pnl[30:]) > 0
        assert np.std(ts_pnl[30:]) > 0

    def test_performance_metrics(self):
        """Test that performance metrics are reasonable."""
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        # Calculate some basic performance metrics
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 0:
            annual_return = np.mean(valid_pnl) * 260
            annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(260)

            # Volatility should be positive
            assert annual_vol > 0

            # Sharpe ratio should be finite
            if annual_vol > 0:
                sharpe = annual_return / annual_vol
                assert np.isfinite(sharpe)

    def test_weight_constraints(self):
        """Test that position weights are reasonable."""
        cfg = EuropeanTFConfig(sigma_target_annual=0.15)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)

        # Weights should not be extreme in normal conditions
        valid_weights = weights[~np.isnan(weights)]
        if len(valid_weights) > 0:
            # Most weights should be reasonable (not extreme leverage)
            reasonable_weights = np.abs(valid_weights) < 10
            assert np.mean(reasonable_weights) > 0.8  # 80% should be reasonable
