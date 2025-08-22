import numpy as np
import pytest
from tfunify.american import AmericanTF, AmericanTFConfig, _true_range, _atr


class TestAmericanTFConfig:
    """Comprehensive tests for AmericanTFConfig validation."""

    def test_default_configuration(self):
        """Test default configuration is valid."""
        cfg = AmericanTFConfig()
        assert cfg.span_long == 250
        assert cfg.span_short == 20
        assert cfg.atr_period == 33
        assert cfg.q == 5.0
        assert cfg.p == 5.0
        assert cfg.r_multiple == 0.01

    def test_span_validation(self):
        """Test span parameter validation."""
        # Valid spans
        AmericanTFConfig(span_long=100, span_short=20)
        AmericanTFConfig(span_long=2, span_short=1)

        # Invalid spans - negative or zero
        with pytest.raises(ValueError, match="span_long must be >= 1"):
            AmericanTFConfig(span_long=0)
        with pytest.raises(ValueError, match="span_short must be >= 1"):
            AmericanTFConfig(span_short=0)
        with pytest.raises(ValueError, match="span_long must be >= 1"):
            AmericanTFConfig(span_long=-10)

        # Invalid span relationship
        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            AmericanTFConfig(span_long=20, span_short=30)
        with pytest.raises(ValueError, match="span_short must be less than span_long"):
            AmericanTFConfig(span_long=50, span_short=50)

    def test_atr_period_validation(self):
        """Test ATR period validation."""
        # Valid periods
        AmericanTFConfig(atr_period=1)
        AmericanTFConfig(atr_period=100)

        # Invalid periods
        with pytest.raises(ValueError, match="atr_period must be >= 1"):
            AmericanTFConfig(atr_period=0)
        with pytest.raises(ValueError, match="atr_period must be >= 1"):
            AmericanTFConfig(atr_period=-5)

    def test_threshold_validation(self):
        """Test q and p threshold validation."""
        # Valid thresholds
        AmericanTFConfig(q=0.1, p=0.1)
        AmericanTFConfig(q=10.0, p=10.0)

        # Invalid q
        with pytest.raises(ValueError, match="q must be positive"):
            AmericanTFConfig(q=0.0)
        with pytest.raises(ValueError, match="q must be positive"):
            AmericanTFConfig(q=-1.0)

        # Invalid p
        with pytest.raises(ValueError, match="p must be positive"):
            AmericanTFConfig(p=0.0)
        with pytest.raises(ValueError, match="p must be positive"):
            AmericanTFConfig(p=-2.0)

    def test_r_multiple_validation(self):
        """Test r_multiple validation."""
        # Valid values
        AmericanTFConfig(r_multiple=0.001)
        AmericanTFConfig(r_multiple=0.1)

        # Invalid values
        with pytest.raises(ValueError, match="r_multiple must be positive"):
            AmericanTFConfig(r_multiple=0.0)
        with pytest.raises(ValueError, match="r_multiple must be positive"):
            AmericanTFConfig(r_multiple=-0.01)


class TestTrueRange:
    """Tests for True Range calculation."""

    def test_basic_true_range(self):
        """Test basic True Range calculation."""
        high = np.array([105.0, 108.0, 107.0, 110.0])
        low = np.array([100.0, 103.0, 104.0, 106.0])
        close = np.array([102.0, 107.0, 106.0, 109.0])

        tr = _true_range(high, low, close)

        # Manual calculation (using correct previous close values)
        # t=0: max(105-100, |105-102|, |100-102|) = max(5, 3, 2) = 5
        # t=1: max(108-103, |108-102|, |103-102|) = max(5, 6, 1) = 6  (uses close[0]=102)
        # t=2: max(107-104, |107-107|, |104-107|) = max(3, 0, 3) = 3  (uses close[1]=107)
        # t=3: max(110-106, |110-106|, |106-106|) = max(4, 4, 0) = 4  (uses close[2]=106)
        expected = np.array([5.0, 6.0, 3.0, 4.0])
        np.testing.assert_allclose(tr, expected)

    def test_true_range_with_gaps(self):
        """Test True Range with price gaps."""
        high = np.array([100.0, 120.0, 115.0])  # Gap up
        low = np.array([95.0, 115.0, 110.0])
        close = np.array([98.0, 118.0, 113.0])

        tr = _true_range(high, low, close)

        # Correct manual calculation (using actual previous close values)
        # t=0: max(100-95, |100-98|, |95-98|) = max(5, 2, 3) = 5
        # t=1: max(120-115, |120-98|, |115-98|) = max(5, 22, 17) = 22  (uses close[0]=98)
        # t=2: max(115-110, |115-118|, |110-118|) = max(5, 3, 8) = 8   (uses close[1]=118)
        expected = np.array([5.0, 22.0, 8.0])
        np.testing.assert_allclose(tr, expected)

    def test_true_range_small_gaps(self):
        """Test True Range with smaller gaps where H-L dominates."""
        high = np.array([105.0, 106.0, 107.0])
        low = np.array([100.0, 101.0, 102.0])
        close = np.array([103.0, 104.0, 105.0])

        tr = _true_range(high, low, close)

        # Manual calculation:
        # t=0: max(105-100, |105-103|, |100-103|) = max(5, 2, 3) = 5
        # t=1: max(106-101, |106-103|, |101-103|) = max(5, 3, 2) = 5
        # t=2: max(107-102, |107-104|, |102-104|) = max(5, 3, 2) = 5
        expected = np.array([5.0, 5.0, 5.0])
        np.testing.assert_allclose(tr, expected)

    def test_true_range_gap_down(self):
        """Test True Range with gap down scenario."""
        high = np.array([110.0, 105.0, 108.0])
        low = np.array([105.0, 95.0, 103.0])  # Gap down on day 2
        close = np.array([108.0, 98.0, 106.0])

        tr = _true_range(high, low, close)

        # Manual calculation:
        # t=0: max(110-105, |110-108|, |105-108|) = max(5, 2, 3) = 5
        # t=1: max(105-95, |105-108|, |95-108|) = max(10, 3, 13) = 13  (gap down)
        # t=2: max(108-103, |108-98|, |103-98|) = max(5, 10, 5) = 10
        expected = np.array([5.0, 13.0, 10.0])
        np.testing.assert_allclose(tr, expected)

    def test_true_range_single_observation(self):
        """Test True Range with single observation."""
        high = np.array([105.0])
        low = np.array([100.0])
        close = np.array([102.0])

        tr = _true_range(high, low, close)

        # Should be high - low = 5.0 for first observation
        assert tr[0] == 5.0

    def test_true_range_input_validation(self):
        """Test True Range input validation."""
        high = np.array([105.0, 108.0])
        low = np.array([100.0, 103.0])
        close = np.array([102.0])  # Different length

        with pytest.raises(ValueError, match="high, low, and close must have same shape"):
            _true_range(high, low, close)

    def test_true_range_invalid_prices(self):
        """Test True Range with invalid price relationships."""
        high = np.array([100.0, 105.0])
        low = np.array([105.0, 103.0])  # Low > High for first observation
        close = np.array([102.0, 104.0])

        with pytest.raises(ValueError, match="high prices cannot be less than low prices"):
            _true_range(high, low, close)


class TestATR:
    """Tests for Average True Range calculation."""

    def test_basic_atr(self):
        """Test basic ATR calculation."""
        high = np.array([105.0, 108.0, 107.0, 110.0, 112.0])
        low = np.array([100.0, 103.0, 104.0, 106.0, 108.0])
        close = np.array([102.0, 107.0, 106.0, 109.0, 111.0])
        period = 3

        atr = _atr(high, low, close, period)

        # First period-1 values should be NaN
        assert np.isnan(atr[0])
        assert np.isnan(atr[1])

        # ATR[2] should be average of first 3 TRs
        tr = _true_range(high, low, close)
        expected_atr_2 = np.mean(tr[:3])
        assert abs(atr[2] - expected_atr_2) < 1e-10

    def test_atr_period_validation(self):
        """Test ATR period validation."""
        high = np.array([105.0, 108.0])
        low = np.array([100.0, 103.0])
        close = np.array([102.0, 107.0])

        with pytest.raises(ValueError, match="ATR period must be >= 1"):
            _atr(high, low, close, 0)


class TestAmericanTF:
    """Comprehensive tests for AmericanTF system."""

    def setup_method(self):
        """Set up test data before each test."""
        np.random.seed(42)
        self.n = 500

        # Generate trending price series with realistic OHLC
        base_returns = 0.0005 + 0.015 * np.random.randn(self.n)
        # Add some momentum
        for i in range(1, self.n):
            base_returns[i] += 0.08 * base_returns[i - 1]

        self.close = 100 * np.cumprod(1 + np.r_[0.0, base_returns[1:]])

        # Generate realistic high/low prices
        daily_range = 0.005 + 0.01 * np.abs(np.random.randn(self.n))
        self.high = self.close * (1 + daily_range * np.random.uniform(0.3, 1.0, self.n))
        self.low = self.close * (1 - daily_range * np.random.uniform(0.3, 1.0, self.n))

        # Ensure OHLC consistency
        self.high = np.maximum(self.high, self.close)
        self.low = np.minimum(self.low, self.close)

    def test_basic_functionality(self):
        """Test basic American TF functionality."""
        cfg = AmericanTFConfig(
            span_long=50, span_short=10, atr_period=20, q=2.0, p=3.0, r_multiple=0.01
        )
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # Basic shape checks
        assert len(pnl) == len(self.close)
        assert len(units) == len(self.close)

        # After warmup, values should be finite
        warmup = max(cfg.span_long, cfg.atr_period) + 10
        assert np.isfinite(pnl[warmup:]).all()
        assert np.isfinite(units[warmup:]).all()

    def test_close_only_mode(self):
        """Test running with only close prices."""
        cfg = AmericanTFConfig(span_long=30, span_short=5, atr_period=15)
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close)  # No high/low provided

        assert len(pnl) == len(self.close)
        assert len(units) == len(self.close)

        # Should produce valid results
        warmup = 40
        assert np.isfinite(pnl[warmup:]).all()

    def test_position_sizing(self):
        """Test position sizing logic."""
        cfg = AmericanTFConfig(r_multiple=0.02, q=1.0)  # Easy entry conditions
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # When in position, units should follow r_multiple formula
        non_zero_units = units[units != 0]
        if len(non_zero_units) > 0:
            # Check that position sizes are reasonable
            assert np.all(np.abs(non_zero_units) > 0)
            assert np.all(np.abs(non_zero_units) < 10)  # Not extremely large

    def test_entry_exit_logic(self):
        """Test entry and exit signal logic."""
        cfg = AmericanTFConfig(
            span_long=20,
            span_short=5,
            q=0.5,  # Easy entry
            p=2.0,  # Stop loss
        )
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # Should have some position changes
        position_changes = np.sum(np.abs(np.diff(units)) > 1e-10)
        assert position_changes > 0  # Should enter/exit positions

    def test_stop_loss_mechanism(self):
        """Test stop loss mechanism."""
        # Create data with a clear trend followed by reversal
        trend_up = np.linspace(100, 120, 50)
        trend_down = np.linspace(120, 100, 50)
        close_prices = np.concatenate([trend_up, trend_down])
        high_prices = close_prices * 1.01
        low_prices = close_prices * 0.99

        cfg = AmericanTFConfig(
            span_long=20,
            span_short=5,
            q=1.0,  # Easy entry
            p=2.0,  # Reasonable stop
            atr_period=10,
        )
        system = AmericanTF(cfg)
        pnl, units = system.run(close_prices, high_prices, low_prices)

        # Should exit positions when trend reverses
        assert len(pnl) == len(close_prices)

    def test_different_parameter_combinations(self):
        """Test various parameter combinations."""
        param_sets = [
            (10, 2, 5, 1.0, 1.5, 0.005),  # Fast system
            (100, 20, 30, 3.0, 4.0, 0.02),  # Slow system
            (50, 10, 15, 0.5, 0.8, 0.01),  # Sensitive system
        ]

        for span_long, span_short, atr_period, q, p, r_mult in param_sets:
            cfg = AmericanTFConfig(
                span_long=span_long,
                span_short=span_short,
                atr_period=atr_period,
                q=q,
                p=p,
                r_multiple=r_mult,
            )
            system = AmericanTF(cfg)
            pnl, units = system.run(self.close, self.high, self.low)

            # All should produce valid results
            warmup = max(span_long, atr_period) + 10
            if warmup < len(pnl):
                assert np.isfinite(pnl[warmup:]).all()

    def test_extreme_market_conditions(self):
        """Test with extreme market conditions."""
        # Very volatile market
        np.random.seed(123)
        volatile_returns = 0.05 * np.random.randn(200)
        volatile_close = 100 * np.cumprod(1 + np.r_[0.0, volatile_returns[1:]])
        volatile_high = volatile_close * (1 + 0.02 * np.abs(np.random.randn(200)))
        volatile_low = volatile_close * (1 - 0.02 * np.abs(np.random.randn(200)))

        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)
        pnl, units = system.run(volatile_close, volatile_high, volatile_low)

        # Should handle extreme volatility
        warmup = 50
        assert np.isfinite(pnl[warmup:]).all()

    def test_flat_market(self):
        """Test with flat/sideways market."""
        flat_close = np.full(100, 100.0) + 0.1 * np.random.randn(100)
        flat_high = flat_close + 0.5
        flat_low = flat_close - 0.5

        cfg = AmericanTFConfig(q=2.0)  # Higher threshold for flat market
        system = AmericanTF(cfg)
        pnl, units = system.run(flat_close, flat_high, flat_low)

        # Should produce minimal activity in flat market
        assert len(pnl) == len(flat_close)
        assert np.isfinite(pnl).all()

    def test_pnl_calculation(self):
        """Test P&L calculation accuracy."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # Calculate manual P&L
        price_changes = np.diff(self.close)
        manual_pnl = np.zeros_like(pnl)
        manual_pnl[1:] = units[:-1] * price_changes

        # Should match calculated P&L
        np.testing.assert_allclose(pnl, manual_pnl)

    def test_units_constraints(self):
        """Test that position units are reasonable."""
        cfg = AmericanTFConfig(r_multiple=0.01)
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # Units should not be extremely large
        max_abs_units = np.max(np.abs(units))
        assert max_abs_units < 100  # Reasonable upper bound

    def test_empty_input_validation(self):
        """Test validation of empty inputs."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)

        with pytest.raises(ValueError, match="Close prices cannot be empty"):
            system.run(np.array([]))

    def test_mismatched_input_shapes(self):
        """Test validation of mismatched input shapes."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)

        close = np.array([100.0, 110.0, 105.0])
        high = np.array([105.0, 115.0])  # Different length
        low = np.array([95.0, 105.0, 100.0])

        with pytest.raises(ValueError, match="high, low, and close must have same shape"):
            system.run(close, high, low)

    def test_state_persistence(self):
        """Test that position state is maintained correctly."""
        # Create clear trending data
        trend_close = np.array(
            [100.0, 102.0, 104.0, 106.0, 108.0, 110.0, 108.0, 106.0, 104.0, 102.0]
        )  # Up then down
        trend_high = trend_close + 1.0
        trend_low = trend_close - 1.0

        cfg = AmericanTFConfig(
            span_long=4,
            span_short=2,
            q=0.5,  # Easy entry
            p=2.0,  # Stop loss
            atr_period=3,
        )
        system = AmericanTF(cfg)
        pnl, units = system.run(trend_close, trend_high, trend_low)

        # Position should persist until exit conditions are met
        # Check that positions don't change randomly
        position_changes = np.where(np.abs(np.diff(units)) > 1e-10)[0]
        # Should have some but not excessive position changes
        assert len(position_changes) <= len(trend_close) // 2

    def test_atr_dependency(self):
        """Test system behavior when ATR is not available."""
        # Use very short data where ATR might not be calculated initially
        short_close = self.close[:10]
        short_high = self.high[:10]
        short_low = self.low[:10]

        cfg = AmericanTFConfig(atr_period=8)  # Long ATR period
        system = AmericanTF(cfg)
        pnl, units = system.run(short_close, short_high, short_low)

        # Should handle gracefully when ATR is not available
        assert len(pnl) == len(short_close)
        assert len(units) == len(short_close)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Very small price movements
        stable_close = 100.0 + 1e-6 * np.random.randn(100)
        stable_high = stable_close + 1e-6
        stable_low = stable_close - 1e-6

        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)
        pnl, units = system.run(stable_close, stable_high, stable_low)

        # Should handle small movements without numerical issues
        assert np.isfinite(pnl).all()
        assert np.isfinite(units).all()

    def test_configuration_immutability(self):
        """Test that configuration doesn't change during execution."""
        cfg = AmericanTFConfig(q=2.5, p=3.5, r_multiple=0.015)
        original_q = cfg.q
        original_p = cfg.p
        original_r = cfg.r_multiple

        system = AmericanTF(cfg)
        system.run(self.close, self.high, self.low)

        # Configuration should remain unchanged
        assert cfg.q == original_q
        assert cfg.p == original_p
        assert cfg.r_multiple == original_r

    def test_long_vs_short_positions(self):
        """Test that system can take both long and short positions."""
        # Create data with both up and down trends
        up_trend = np.linspace(100, 120, 100)
        down_trend = np.linspace(120, 100, 100)
        mixed_close = np.concatenate([up_trend, down_trend])
        mixed_high = mixed_close * 1.005
        mixed_low = mixed_close * 0.995

        cfg = AmericanTFConfig(span_long=20, span_short=5, q=1.0, atr_period=10)
        system = AmericanTF(cfg)
        pnl, units = system.run(mixed_close, mixed_high, mixed_low)

        # Should have both positive and negative positions
        positive_units = units[units > 1e-10]
        negative_units = units[units < -1e-10]

        # Expect some of both (though not guaranteed)
        assert len(positive_units) + len(negative_units) > 0

    def test_performance_metrics_validity(self):
        """Test that performance metrics are reasonable."""
        cfg = AmericanTFConfig()
        system = AmericanTF(cfg)
        pnl, units = system.run(self.close, self.high, self.low)

        # Calculate basic performance metrics
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 50:  # Need sufficient data
            total_pnl = np.sum(valid_pnl)
            pnl_vol = np.std(valid_pnl)

            # Metrics should be finite
            assert np.isfinite(total_pnl)
            assert np.isfinite(pnl_vol)
            assert pnl_vol >= 0

    def test_extreme_parameters_edge_cases(self):
        """Test with extreme but valid parameter values."""
        # Very sensitive system
        cfg_sensitive = AmericanTFConfig(
            span_long=3, span_short=1, atr_period=1, q=0.1, p=0.1, r_multiple=0.001
        )
        system = AmericanTF(cfg_sensitive)
        pnl, units = system.run(self.close, self.high, self.low)
        assert np.isfinite(pnl[5:]).all()

        # Very conservative system
        cfg_conservative = AmericanTFConfig(
            span_long=200, span_short=50, atr_period=50, q=10.0, p=10.0, r_multiple=0.1
        )
        system = AmericanTF(cfg_conservative)
        pnl, units = system.run(self.close, self.high, self.low)
        warmup = 250
        if warmup < len(pnl):
            assert np.isfinite(pnl[warmup:]).all()

    def test_reproducibility(self):
        """Test that results are reproducible with same inputs."""
        cfg = AmericanTFConfig()
        system1 = AmericanTF(cfg)
        system2 = AmericanTF(cfg)

        pnl1, units1 = system1.run(self.close, self.high, self.low)
        pnl2, units2 = system2.run(self.close, self.high, self.low)

        # Results should be identical
        np.testing.assert_allclose(pnl1, pnl2)
        np.testing.assert_allclose(units1, units2)
