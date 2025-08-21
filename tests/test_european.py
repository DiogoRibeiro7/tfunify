import numpy as np
import pytest
from tfunify.european import EuropeanTF, EuropeanTFConfig


class TestEuropeanTFConfig:
    """Comprehensive tests for EuropeanTFConfig validation."""

    def test_default_configuration(self):
        """Test default configuration is valid."""
        cfg = EuropeanTFConfig()
        assert cfg.sigma_target_annual == 0.15
        assert cfg.a == 260
        assert cfg.span_sigma == 33
        assert cfg.mode == "longshort"
        assert cfg.span_long == 250
        assert cfg.span_short == 20

    def test_sigma_target_validation(self):
        """Test sigma_target_annual validation."""
        # Valid values
        EuropeanTFConfig(sigma_target_annual=0.01)
        EuropeanTFConfig(sigma_target_annual=0.5)
        EuropeanTFConfig(sigma_target_annual=1.0)
        
        # Invalid values
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            EuropeanTFConfig(sigma_target_annual=0.0)
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            EuropeanTFConfig(sigma_target_annual=-0.1)

    def test_trading_days_validation(self):
        """Test trading days per year validation."""
        # Valid values
        EuropeanTFConfig(a=252)
        EuropeanTFConfig(a=365)
        EuropeanTFConfig(a=1)
        
        # Invalid values
        with pytest.raises(ValueError, match="a \\(trading days per year\\) must be positive"):
            EuropeanTFConfig(a=0)
        with pytest.raises(ValueError, match="a \\(trading days per year\\) must be positive"):
            EuropeanTFConfig(a=-252)

    def test_span_validation(self):
        """Test span parameter validation."""
        # Valid spans
        EuropeanTFConfig(span_sigma=1, span_long=10, span_short=5)
        EuropeanTFConfig(span_sigma=100, span_long=500, span_short=50)
        
        # Invalid spans
        with pytest.raises(ValueError, match="span_sigma must be >= 1"):
            EuropeanTFConfig(span_sigma=0)
        with pytest.raises(ValueError, match="span_long must be >= 1"):
            EuropeanTFConfig(span_long=0)
        with pytest.raises(ValueError, match="span_short must be >= 1"):
            EuropeanTFConfig(span_short=0)

    def test_mode_validation(self):
        """Test mode parameter validation."""
        # Valid modes
        EuropeanTFConfig(mode="single")
        EuropeanTFConfig(mode="longshort")
        
        # Invalid modes
        with pytest.raises(ValueError, match="mode must be 'single' or 'longshort'"):
            EuropeanTFConfig(mode="invalid")
        with pytest.raises(ValueError, match="mode must be 'single' or 'longshort'"):
            EuropeanTFConfig(mode="SINGLE")  # Case sensitive

    def test_longshort_span_consistency(self):
        """Test span consistency in longshort mode."""
        # Valid: short < long
        EuropeanTFConfig(mode="longshort", span_short=20, span_long=100)
        
        # Invalid: short >= long
        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            EuropeanTFConfig(mode="longshort", span_short=100, span_long=50)
        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            EuropeanTFConfig(mode="longshort", span_short=50, span_long=50)

    def test_single_mode_span_flexibility(self):
        """Test that single mode doesn't require span ordering."""
        # Should be valid even if span_short > span_long in single mode
        EuropeanTFConfig(mode="single", span_short=100, span_long=50)


class TestEuropeanTF:
    """Comprehensive tests for EuropeanTF system."""

    def setup_method(self):
        """Set up test data before each test."""
        np.random.seed(42)
        self.n = 1000
        # Generate trending price series
        drift = 0.0002
        vol = 0.015
        returns = drift + vol * np.random.randn(self.n)
        # Add some momentum
        for i in range(1, self.n):
            returns[i] += 0.05 * returns[i-1]
        
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])
        self.returns = np.diff(np.log(self.prices))
        self.returns = np.r_[0.0, self.returns]  # Add initial zero return

    def test_single_mode_basic(self):
        """Test basic single mode functionality."""
        cfg = EuropeanTFConfig(
            sigma_target_annual=0.12,
            span_sigma=20,
            mode="single",
            span_long=50
        )
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # Basic shape and finite checks
        assert len(pnl) == len(self.prices)
        assert len(weights) == len(self.prices)
        assert len(signal) == len(self.prices)
        assert len(volatility) == len(self.prices)
        
        # After warmup, values should be finite
        warmup = max(cfg.span_sigma, cfg.span_long) + 10
        assert np.isfinite(pnl[warmup:]).all()
        assert np.isfinite(weights[warmup:]).all()
        assert np.isfinite(signal[warmup:]).all()
        assert np.isfinite(volatility[warmup:]).all()

    def test_longshort_mode_basic(self):
        """Test basic longshort mode functionality."""
        cfg = EuropeanTFConfig(
            sigma_target_annual=0.15,
            span_sigma=30,
            mode="longshort",
            span_long=100,
            span_short=10
        )
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # Basic checks
        assert len(pnl) == len(self.prices)
        warmup = max(cfg.span_sigma, cfg.span_long) + 10
        assert np.isfinite(pnl[warmup:]).all()
        
        # Signal should have more variation in longshort mode
        signal_std = np.std(signal[warmup:])
        assert signal_std > 0

    def test_run_from_returns_equivalence(self):
        """Test that run_from_prices and run_from_returns give same results."""
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        
        pnl1, weights1, signal1, vol1 = system.run_from_prices(self.prices)
        pnl2, weights2, signal2, vol2 = system.run_from_returns(self.returns)
        
        # Results should be identical
        np.testing.assert_allclose(pnl1, pnl2)
        np.testing.assert_allclose(weights1, weights2)
        np.testing.assert_allclose(signal1, signal2)
        np.testing.assert_allclose(vol1, vol2)

    def test_volatility_targeting(self):
        """Test that volatility targeting works correctly."""
        target_vol = 0.10
        cfg = EuropeanTFConfig(
            sigma_target_annual=target_vol,
            span_sigma=20,
            mode="single",
            span_long=50
        )
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # Calculate realized volatility
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 100:  # Need sufficient data
            realized_vol = np.std(valid_pnl) * np.sqrt(cfg.a)
            # Should be reasonably close to target (within factor of 2)
            assert 0.5 * target_vol < realized_vol < 2.0 * target_vol

    def test_pnl_calculation_consistency(self):
        """Test P&L calculation consistency."""
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # P&L should be w[t-1] * r[t]
        returns = np.diff(np.log(self.prices))
        returns = np.r_[0.0, returns]
        
        # Manual P&L calculation
        manual_pnl = np.zeros_like(pnl)
        manual_pnl[1:] = weights[:-1] * returns[1:]
        
        np.testing.assert_allclose(pnl, manual_pnl)

    def test_signal_properties(self):
        """Test signal properties."""
        cfg = EuropeanTFConfig(mode="longshort", span_long=100, span_short=20)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # Signal should have reasonable range (not extreme)
        valid_signal = signal[~np.isnan(signal)]
        if len(valid_signal) > 0:
            assert np.abs(np.mean(valid_signal)) < 10  # Reasonable mean
            assert np.std(valid_signal) < 50  # Not extremely volatile

    def test_volatility_estimates(self):
        """Test volatility estimates."""
        cfg = EuropeanTFConfig(span_sigma=20)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        # Volatility should be positive and reasonable
        valid_vol = volatility[~np.isnan(volatility)]
        assert np.all(valid_vol > 0)
        
        # Annualized volatility should be reasonable (1% to 100%)
        annual_vol = valid_vol * np.sqrt(cfg.a)
        assert np.all(annual_vol > 0.01)  # At least 1%
        assert np.all(annual_vol < 1.0)   # Less than 100%

    def test_extreme_parameters(self):
        """Test with extreme but valid parameters."""
        # Very high vol target
        cfg_high_vol = EuropeanTFConfig(sigma_target_annual=0.5)
        system = EuropeanTF(cfg_high_vol)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        assert np.isfinite(pnl[50:]).all()
        
        # Very low vol target
        cfg_low_vol = EuropeanTFConfig(sigma_target_annual=0.01)
        system = EuropeanTF(cfg_low_vol)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        assert np.isfinite(pnl[50:]).all()
        
        # Very short spans
        cfg_short = EuropeanTFConfig(span_sigma=2, span_long=5, span_short=2)
        system = EuropeanTF(cfg_short)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        assert np.isfinite(pnl[10:]).all()

    def test_constant_prices(self):
        """Test with constant price series."""
        constant_prices = np.full(100, 100.0)
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(constant_prices)
        
        # Should handle gracefully without errors
        assert len(pnl) == len(constant_prices)
        # P&L should be mostly zero (no price changes)
        np.testing.assert_allclose(pnl, 0.0, atol=1e-10)

    def test_high_frequency_data(self):
        """Test with high frequency (many observations) data."""
        np.random.seed(123)
        n_hf = 10000
        returns_hf = 0.00001 + 0.001 * np.random.randn(n_hf)
        prices_hf = 100 * np.cumprod(1 + np.r_[0.0, returns_hf[1:]])
        
        cfg = EuropeanTFConfig(a=365*24*60)  # Minute data
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(prices_hf)
        
        # Should handle large datasets
        assert len(pnl) == n_hf
        warmup = 100
        assert np.isfinite(pnl[warmup:]).all()

    def test_trending_vs_mean_reverting_data(self):
        """Test system behavior on different market regimes."""
        np.random.seed(456)
        n = 500
        
        # Trending data
        trending_returns = 0.001 + 0.01 * np.random.randn(n)
        for i in range(1, n):
            trending_returns[i] += 0.1 * trending_returns[i-1]  # Add momentum
        trending_prices = 100 * np.cumprod(1 + np.r_[0.0, trending_returns[1:]])
        
        # Mean reverting data
        mr_returns = np.zeros(n)
        mr_returns[0] = 0.01 * np.random.randn()
        for i in range(1, n):
            mr_returns[i] = -0.1 * mr_returns[i-1] + 0.01 * np.random.randn()
        mr_prices = 100 * np.cumprod(1 + np.r_[0.0, mr_returns[1:]])
        
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        
        # Run on both datasets
        pnl_trend, _, _, _ = system.run_from_prices(trending_prices)
        pnl_mr, _, _, _ = system.run_from_prices(mr_prices)
        
        # Both should produce valid results
        assert np.isfinite(pnl_trend[50:]).all()
        assert np.isfinite(pnl_mr[50:]).all()
        
        # Trending data might produce higher Sharpe (but not guaranteed)
        trend_sharpe = np.mean(pnl_trend[50:]) / np.std(pnl_trend[50:]) if np.std(pnl_trend[50:]) > 0 else 0
        mr_sharpe = np.mean(pnl_mr[50:]) / np.std(pnl_mr[50:]) if np.std(pnl_mr[50:]) > 0 else 0
        
        # Both should be finite
        assert np.isfinite(trend_sharpe)
        assert np.isfinite(mr_sharpe)

    def test_weight_constraints(self):
        """Test that weights remain within reasonable bounds."""
        cfg = EuropeanTFConfig(sigma_target_annual=0.15)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(self.prices)
        
        valid_weights = weights[~np.isnan(weights)]
        if len(valid_weights) > 0:
            # Most weights should be reasonable (not extreme leverage)
            reasonable_weights = np.abs(valid_weights) < 20
            assert np.mean(reasonable_weights) > 0.9  # 90% should be reasonable

    def test_minimal_data(self):
        """Test with minimal amount of data."""
        # Test with just enough data
        min_prices = self.prices[:100]  # Minimum reasonable amount
        cfg = EuropeanTFConfig(span_long=20, span_short=5, span_sigma=10)
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(min_prices)
        
        assert len(pnl) == len(min_prices)
        # Some values should be finite after warmup
        warmup = 25
        assert np.any(np.isfinite(pnl[warmup:]))

    def test_different_span_combinations(self):
        """Test various span combinations."""
        span_combinations = [
            (10, 50, 5),    # Short sigma, medium long, very short short
            (100, 500, 50), # Long sigma, very long long, medium short
            (20, 100, 20),  # Equal sigma and short spans
        ]
        
        for span_sigma, span_long, span_short in span_combinations:
            cfg = EuropeanTFConfig(
                span_sigma=span_sigma,
                mode="longshort", 
                span_long=span_long,
                span_short=span_short
            )
            system = EuropeanTF(cfg)
            pnl, weights, signal, volatility = system.run_from_prices(self.prices)
            
            # Should produce valid results for all combinations
            warmup = max(span_sigma, span_long) + 20
            if warmup < len(pnl):
                assert np.isfinite(pnl[warmup:]).all()

    def test_edge_case_very_small_prices(self):
        """Test with very small price values."""
        small_prices = self.prices * 1e-6  # Micro prices
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(small_prices)
        
        # Should handle small prices without numerical issues
        warmup = 50
        assert np.isfinite(pnl[warmup:]).all()
        assert np.isfinite(weights[warmup:]).all()

    def test_edge_case_very_large_prices(self):
        """Test with very large price values."""
        large_prices = self.prices * 1e6  # Million dollar prices
        cfg = EuropeanTFConfig()
        system = EuropeanTF(cfg)
        pnl, weights, signal, volatility = system.run_from_prices(large_prices)
        
        # Should handle large prices without numerical issues
        warmup = 50
        assert np.isfinite(pnl[warmup:]).all()
        assert np.isfinite(weights[warmup:]).all()

    def test_configuration_immutability(self):
        """Test that configuration doesn't change during execution."""
        cfg = EuropeanTFConfig(sigma_target_annual=0.12, span_long=100)
        original_target = cfg.sigma_target_annual
        original_span = cfg.span_long
        
        system = EuropeanTF(cfg)
        system.run_from_prices(self.prices)
        
        # Configuration should remain unchanged
        assert cfg.sigma_target_annual == original_target
        assert cfg.span_long == original_span
