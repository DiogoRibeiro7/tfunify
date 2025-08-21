import numpy as np
import pytest
import math
from tfunify.tsmom import TSMOM, TSMOMConfig


class TestTSMOMConfig:
    """Comprehensive tests for TSMOMConfig validation."""

    def test_default_configuration(self):
        """Test default configuration is valid."""
        cfg = TSMOMConfig()
        assert cfg.sigma_target_annual == 0.15
        assert cfg.a == 260
        assert cfg.span_sigma == 33
        assert cfg.L == 10
        assert cfg.M == 10

    def test_sigma_target_validation(self):
        """Test sigma_target_annual validation."""
        # Valid values
        TSMOMConfig(sigma_target_annual=0.01)
        TSMOMConfig(sigma_target_annual=1.0)
        
        # Invalid values
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            TSMOMConfig(sigma_target_annual=0.0)
        with pytest.raises(ValueError, match="sigma_target_annual must be positive"):
            TSMOMConfig(sigma_target_annual=-0.1)

    def test_trading_days_validation(self):
        """Test trading days per year validation."""
        # Valid values
        TSMOMConfig(a=252)
        TSMOMConfig(a=365)
        TSMOMConfig(a=1)
        
        # Invalid values
        with pytest.raises(ValueError, match="a \\(trading days per year\\) must be positive"):
            TSMOMConfig(a=0)
        with pytest.raises(ValueError, match="a \\(trading days per year\\) must be positive"):
            TSMOMConfig(a=-260)

    def test_span_sigma_validation(self):
        """Test span_sigma validation."""
        # Valid values
        TSMOMConfig(span_sigma=1)
        TSMOMConfig(span_sigma=100)
        
        # Invalid values
        with pytest.raises(ValueError, match="span_sigma must be >= 1"):
            TSMOMConfig(span_sigma=0)
        with pytest.raises(ValueError, match="span_sigma must be >= 1"):
            TSMOMConfig(span_sigma=-5)

    def test_L_validation(self):
        """Test L (block length) validation."""
        # Valid values
        TSMOMConfig(L=1)
        TSMOMConfig(L=100)
        
        # Invalid values
        with pytest.raises(ValueError, match="L \\(block length\\) must be >= 1"):
            TSMOMConfig(L=0)
        with pytest.raises(ValueError, match="L \\(block length\\) must be >= 1"):
            TSMOMConfig(L=-10)

    def test_M_validation(self):
        """Test M (number of blocks) validation."""
        # Valid values
        TSMOMConfig(M=1)
        TSMOMConfig(M=50)
        
        # Invalid values
        with pytest.raises(ValueError, match="M \\(number of blocks\\) must be >= 1"):
            TSMOMConfig(M=0)
        with pytest.raises(ValueError, match="M \\(number of blocks\\) must be >= 1"):
            TSMOMConfig(M=-5)

    def test_extreme_valid_combinations(self):
        """Test extreme but valid parameter combinations."""
        # Very small blocks
        TSMOMConfig(L=1, M=1)
        
        # Very large blocks
        TSMOMConfig(L=100, M=50)
        
        # Unbalanced combinations
        TSMOMConfig(L=1, M=100)
        TSMOMConfig(L=100, M=1)


class TestTSMOM:
    """Comprehensive tests for TSMOM system."""

    def setup_method(self):
        """Set up test data before each test."""
        np.random.seed(42)
        self.n = 2000  # Need more data for TSMOM blocks
        
        # Generate data with momentum characteristics
        base_returns = 0.0002 + 0.015 * np.random.randn(self.n)
        
        # Add momentum persistence
        momentum_returns = np.zeros(self.n)
        momentum_returns[0] = base_returns[0]
        for i in range(1, self.n):
            # Momentum with persistence
            momentum_returns[i] = base_returns[i] + 0.1 * momentum_returns[i-1]
        
        self.prices = 100 * np.cumprod(1 + np.r_[0.0, momentum_returns[1:]])
        self.returns = np.diff(np.log(self.prices))
        self.returns = np.r_[0.0, self.returns]

    def test_basic_functionality(self):
        """Test basic TSMOM functionality."""
        cfg = TSMOMConfig(
            sigma_target_annual=0.12,
            span_sigma=20,
            L=10,
            M=8
        )
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Basic shape checks
        assert len(pnl) == len(self.prices)
        assert len(weights) == len(self.prices)
        assert len(signal_grid) == len(self.prices)
        assert len(volatility) == len(self.prices)
        
        # After sufficient warmup, values should be finite
        min_required = cfg.L * cfg.M + cfg.span_sigma + 10
        if min_required < len(self.prices):
            assert np.isfinite(pnl[min_required:]).all()
            assert np.isfinite(weights[min_required:]).all()
            assert np.isfinite(volatility[min_required:]).all()

    def test_run_from_returns_equivalence(self):
        """Test that run_from_prices and run_from_returns give same results."""
        cfg = TSMOMConfig(L=5, M=6)
        system = TSMOM(cfg)
        
        pnl1, weights1, signal1, vol1 = system.run_from_prices(self.prices)
        pnl2, weights2, signal2, vol2 = system.run_from_returns(self.returns)
        
        # Results should be identical
        np.testing.assert_allclose(pnl1, pnl2)
        np.testing.assert_allclose(weights1, weights2)
        np.testing.assert_allclose(signal1, signal2)
        np.testing.assert_allclose(vol1, vol2)

    def test_block_structure_signals(self):
        """Test that signals are generated at correct block intervals."""
        L, M = 5, 4
        cfg = TSMOMConfig(L=L, M=M, span_sigma=10)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Signals should be generated at grid points (multiples of L)
        grid_points = np.arange(0, len(self.prices), L)
        non_zero_signals = signal_grid[signal_grid != 0]
        
        # Should have some signals generated
        assert len(non_zero_signals) > 0
        
        # Check that signals appear at expected intervals
        signal_indices = np.where(signal_grid != 0)[0]
        if len(signal_indices) > 0:
            # Most signal indices should be multiples of L
            grid_aligned = signal_indices % L == 0
            assert np.mean(grid_aligned) > 0.8  # Most should be grid-aligned

    def test_signal_calculation_logic(self):
        """Test signal calculation logic."""
        # Create simple trending data for easier verification
        L, M = 3, 2
        trend_returns = np.array([0.0, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, 0.02, 0.02])
        trend_prices = 100 * np.cumprod(1 + trend_returns)
        
        cfg = TSMOMConfig(L=L, M=M, span_sigma=2, sigma_target_annual=0.1)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_returns(trend_returns)
        
        # Should produce some signals
        assert len(signal_grid) == len(trend_returns)

    def test_volatility_targeting(self):
        """Test volatility targeting mechanism."""
        target_vol = 0.08
        cfg = TSMOMConfig(
            sigma_target_annual=target_vol,
            L=5,
            M=6,
            span_sigma=15
        )
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Check that realized volatility is in reasonable range
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 100:
            realized_vol = np.std(valid_pnl) * np.sqrt(cfg.a)
            # Should be within reasonable bounds of target
            assert 0.3 * target_vol < realized_vol < 3.0 * target_vol

    def test_weight_forward_filling(self):
        """Test that weights are forward-filled between grid points."""
        L = 10
        cfg = TSMOMConfig(L=L, M=5, span_sigma=10)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Weights should be forward-filled between grid points
        # Check that weights don't change between grid points (except at boundaries)
        for i in range(L, len(weights) - L, L):
            # Within a block, weights should be constant (forward-filled)
            block_weights = weights[i:i+L]
            if np.any(np.isfinite(block_weights)):
                # If any weights in block are finite, check for consistency
                finite_weights = block_weights[np.isfinite(block_weights)]
                if len(finite_weights) > 1:
                    np.testing.assert_allclose(finite_weights, finite_weights[0], atol=1e-10)

    def test_different_block_sizes(self):
        """Test various block size combinations."""
        block_combinations = [
            (1, 10),   # Very short blocks, many blocks
            (20, 3),   # Long blocks, few blocks
            (5, 5),    # Balanced
            (15, 8),   # Medium blocks
        ]
        
        for L, M in block_combinations:
            cfg = TSMOMConfig(L=L, M=M, span_sigma=10)
            system = TSMOM(cfg)
            
            # Need sufficient data for the configuration
            min_required = L * M + 50
            if len(self.prices) >= min_required:
                pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
                
                # Should produce valid results
                warmup = L * M + 20
                assert np.isfinite(pnl[warmup:]).all()

    def test_insufficient_data_error(self):
        """Test error handling with insufficient data."""
        # Configuration requiring more data than available
        cfg = TSMOMConfig(L=50, M=20)  # Needs 1000+ observations
        system = TSMOM(cfg)
        
        # Use insufficient data
        short_prices = self.prices[:500]
        
        with pytest.raises(ValueError, match="Returns array too short"):
            system.run_from_prices(short_prices)

    def test_minimal_sufficient_data(self):
        """Test with minimal sufficient data."""
        L, M = 5, 4
        cfg = TSMOMConfig(L=L, M=M, span_sigma=5)
        system = TSMOM(cfg)
        
        # Use just enough data
        min_required = L * M + 10
        minimal_prices = self.prices[:min_required + 50]
        
        pnl, weights, signal_grid, volatility = system.run_from_prices(minimal_prices)
        assert len(pnl) == len(minimal_prices)

    def test_momentum_detection(self):
        """Test momentum detection capability."""
        # Create clear momentum data
        np.random.seed(789)
        n = 500
        
        # Strong positive momentum
        mom_returns = np.zeros(n)
        mom_returns[0] = 0.01
        for i in range(1, n):
            mom_returns[i] = 0.8 * mom_returns[i-1] + 0.005 + 0.01 * np.random.randn()
        
        mom_prices = 100 * np.cumprod(1 + np.r_[0.0, mom_returns[1:]])
        
        cfg = TSMOMConfig(L=10, M=5, span_sigma=15)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(mom_prices)
        
        # Should detect positive momentum (weights should be predominantly positive)
        valid_weights = weights[~np.isnan(weights)]
        if len(valid_weights) > 50:
            avg_weight = np.mean(valid_weights)
            # With strong positive momentum, average weight should be positive
            assert avg_weight > 0

    def test_mean_reverting_data(self):
        """Test behavior on mean-reverting data."""
        # Create mean-reverting data
        np.random.seed(456)
        n = 800
        mr_returns = np.zeros(n)
        mr_returns[0] = 0.01 * np.random.randn()
        for i in range(1, n):
            mr_returns[i] = -0.2 * mr_returns[i-1] + 0.01 * np.random.randn()
        
        mr_prices = 100 * np.cumprod(1 + np.r_[0.0, mr_returns[1:]])
        
        cfg = TSMOMConfig(L=8, M=6)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(mr_prices)
        
        # Should handle mean-reverting data without errors
        warmup = cfg.L * cfg.M + 30
        assert np.isfinite(pnl[warmup:]).all()

    def test_pnl_calculation_consistency(self):
        """Test P&L calculation consistency."""
        cfg = TSMOMConfig(L=5, M=4)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # P&L should be w[t-1] * r[t]
        returns = np.diff(np.log(self.prices))
        returns = np.r_[0.0, returns]
        
        # Manual P&L calculation
        manual_pnl = np.zeros_like(pnl)
        manual_pnl[1:] = weights[:-1] * returns[1:]
        
        np.testing.assert_allclose(pnl, manual_pnl)

    def test_signal_normalization(self):
        """Test signal normalization factor."""
        L, M = 6, 4
        cfg = TSMOMConfig(L=L, M=M)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Signal should be normalized by sqrt(M * L)
        norm_factor = math.sqrt(M * L)
        non_zero_signals = signal_grid[signal_grid != 0]
        
        # Signals should be reasonable magnitude (scaled by normalization)
        if len(non_zero_signals) > 0:
            max_signal = np.max(np.abs(non_zero_signals))
            # Should be bounded by normalization factor
            assert max_signal <= norm_factor * 2  # Allow some reasonable multiple

    def test_volatility_estimates(self):
        """Test volatility estimation."""
        cfg = TSMOMConfig(span_sigma=20)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Volatility should be positive and reasonable
        valid_vol = volatility[~np.isnan(volatility)]
        assert np.all(valid_vol > 0)
        
        # Daily volatility should be reasonable (0.1% to 10%)
        assert np.all(valid_vol > 0.001)
        assert np.all(valid_vol < 0.1)

    def test_extreme_parameter_combinations(self):
        """Test with extreme but valid parameters."""
        # Very short term
        cfg_short = TSMOMConfig(L=1, M=2, span_sigma=2, sigma_target_annual=0.05)
        system = TSMOM(cfg_short)
        
        # Need minimal data for short-term config
        short_data = self.prices[:50]
        pnl, weights, signal_grid, volatility = system.run_from_prices(short_data)
        assert len(pnl) == len(short_data)
        
        # Very long term (if we have enough data)
        if len(self.prices) >= 1000:
            cfg_long = TSMOMConfig(L=20, M=10, span_sigma=50, sigma_target_annual=0.25)
            system = TSMOM(cfg_long)
            pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
            warmup = 250
            assert np.isfinite(pnl[warmup:]).all()

    def test_empty_returns_validation(self):
        """Test validation of empty returns."""
        cfg = TSMOMConfig()
        system = TSMOM(cfg)
        
        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            system.run_from_returns(np.array([]))
        
        with pytest.raises(ValueError, match="Returns array cannot be empty"):
            system.run_from_prices(np.array([]))

    def test_configuration_immutability(self):
        """Test that configuration doesn't change during execution."""
        cfg = TSMOMConfig(L=8, M=6, sigma_target_annual=0.12)
        original_L = cfg.L
        original_M = cfg.M
        original_sigma = cfg.sigma_target_annual
        
        system = TSMOM(cfg)
        system.run_from_prices(self.prices)
        
        # Configuration should remain unchanged
        assert cfg.L == original_L
        assert cfg.M == original_M
        assert cfg.sigma_target_annual == original_sigma

    def test_constant_price_handling(self):
        """Test handling of constant prices."""
        constant_prices = np.full(200, 100.0)
        cfg = TSMOMConfig(L=5, M=4)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(constant_prices)
        
        # Should handle gracefully
        assert len(pnl) == len(constant_prices)
        # P&L should be zero (no price changes)
        np.testing.assert_allclose(pnl, 0.0, atol=1e-10)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small price movements
        small_prices = 100.0 + 1e-8 * np.cumsum(np.random.randn(500))
        cfg = TSMOMConfig(L=5, M=6)
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(small_prices)
        
        # Should handle small movements without numerical issues
        assert np.isfinite(pnl).all()
        assert np.isfinite(weights).all()
        assert np.isfinite(volatility).all()

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs."""
        cfg = TSMOMConfig(L=6, M=5)
        system1 = TSMOM(cfg)
        system2 = TSMOM(cfg)
        
        pnl1, weights1, signal1, vol1 = system1.run_from_prices(self.prices)
        pnl2, weights2, signal2, vol2 = system2.run_from_prices(self.prices)
        
        # Results should be identical
        np.testing.assert_allclose(pnl1, pnl2)
        np.testing.assert_allclose(weights1, weights2)
        np.testing.assert_allclose(signal1, signal2)
        np.testing.assert_allclose(vol1, vol2)

    def test_performance_metrics_validity(self):
        """Test that performance metrics are reasonable."""
        cfg = TSMOMConfig()
        system = TSMOM(cfg)
        pnl, weights, signal_grid, volatility = system.run_from_prices(self.prices)
        
        # Calculate basic performance metrics
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 100:
            total_pnl = np.sum(valid_pnl)
            pnl_vol = np.std(valid_pnl)
            
            # Metrics should be finite
            assert np.isfinite(total_pnl)
            assert np.isfinite(pnl_vol)
            assert pnl_vol >= 0
            
            # Sharpe ratio should be finite
            if pnl_vol > 0:
                sharpe = np.mean(valid_pnl) / pnl_vol
                assert np.isfinite(sharpe)
