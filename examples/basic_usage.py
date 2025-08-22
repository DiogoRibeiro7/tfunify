#!/usr/bin/env python3
"""
Basic Usage Example for tfunify

This example demonstrates the basic usage of all three trend-following systems
with synthetic data. Perfect for getting started and understanding the API.

Requirements:
    pip install tfunify matplotlib
"""

import numpy as np

# ===== CORE TFUNIFY IMPORTS =====
try:
    from tfunify import (
        EuropeanTF,
        EuropeanTFConfig,
        AmericanTF,
        AmericanTFConfig,
        TSMOM,
        TSMOMConfig,
    )

    TFUNIFY_AVAILABLE = True
except ImportError as e:
    TFUNIFY_AVAILABLE = False
    print(f"Error: tfunify package required for this example.")
    print(f"Install with: pip install tfunify")
    print(f"Error details: {e}")

# ===== OPTIONAL DEPENDENCIES =====

# Plotting support
try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting not available. Install with: pip install matplotlib")


def check_requirements():
    """Check if all required dependencies are available."""
    missing = []

    if not TFUNIFY_AVAILABLE:
        missing.append("tfunify (core package)")

    optional_missing = []
    if not PLOTTING_AVAILABLE:
        optional_missing.append("matplotlib (for plots)")

    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall missing dependencies:")
        print("  pip install tfunify")
        return False

    if optional_missing:
        print("Missing optional dependencies:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print("  Some features may not be available.")

    return True


def generate_sample_data(
    n_days: int = 1000, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate sample OHLC price data for testing.

    Creates a trending price series with realistic volatility and some noise.
    """
    np.random.seed(seed)

    # Generate base returns with slight positive drift
    base_returns = 0.0002 + 0.015 * np.random.randn(n_days)  # ~5% annual drift, 24% vol

    # Add some momentum/trending behavior
    momentum = np.zeros(n_days)
    for i in range(1, n_days):
        momentum[i] = 0.05 * momentum[i - 1] + 0.02 * base_returns[i - 1]

    returns = base_returns + momentum

    # Generate prices
    prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

    # Generate realistic OHLC from prices (simplified)
    daily_vol = 0.01 * np.abs(np.random.randn(n_days))
    high = prices * (1 + daily_vol)
    low = prices * (1 - daily_vol)

    return prices, high, low


def run_european_tf_example():
    """Demonstrate European TF system."""
    print("=" * 60)
    print("EUROPEAN TF SYSTEM EXAMPLE")
    print("=" * 60)

    # Generate sample data
    prices, _, _ = generate_sample_data(1500)

    # Configure European TF system
    config = EuropeanTFConfig(
        sigma_target_annual=0.15,  # Target 15% annual volatility
        a=260,  # 260 trading days per year
        span_sigma=33,  # ~1.5 months for volatility estimation
        mode="longshort",  # Use long-short filter
        span_long=250,  # ~1 year slow filter
        span_short=20,  # ~1 month fast filter
    )

    print(f"Configuration:")
    print(f"  Target vol: {config.sigma_target_annual:.1%}")
    print(f"  Mode: {config.mode}")
    print(f"  Long span: {config.span_long} days")
    print(f"  Short span: {config.span_short} days")
    print()

    # Run the system
    system = EuropeanTF(config)
    pnl, weights, signal, volatility = system.run_from_prices(prices)

    # Calculate performance metrics
    valid_pnl = pnl[~np.isnan(pnl)]
    annual_return = np.mean(valid_pnl) * config.a
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(config.a)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    max_weight = np.max(np.abs(weights[~np.isnan(weights)]))

    print(f"Results ({len(valid_pnl)} trading days):")
    print(f"  Annual return: {annual_return:.2%}")
    print(f"  Annual volatility: {annual_vol:.2%}")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Max absolute weight: {max_weight:.2f}")
    print(f"  Total P&L: {np.sum(valid_pnl):.2f}")
    print()

    return pnl, weights, signal, prices


def run_american_tf_example():
    """Demonstrate American TF system."""
    print("=" * 60)
    print("AMERICAN TF SYSTEM EXAMPLE")
    print("=" * 60)

    # Generate sample data
    close, high, low = generate_sample_data(1000)

    # Configure American TF system
    config = AmericanTFConfig(
        span_long=100,  # ~4 months slow MA
        span_short=20,  # ~1 month fast MA
        atr_period=20,  # ~1 month ATR period
        q=2.5,  # Entry threshold: 2.5 * ATR
        p=3.0,  # Stop loss: 3.0 * ATR
        r_multiple=0.02,  # Risk 2% of price per ATR
    )

    print(f"Configuration:")
    print(f"  Long MA: {config.span_long} days")
    print(f"  Short MA: {config.span_short} days")
    print(f"  ATR period: {config.atr_period} days")
    print(f"  Entry threshold: {config.q} × ATR")
    print(f"  Stop loss: {config.p} × ATR")
    print(f"  Risk multiple: {config.r_multiple}")
    print()

    # Run the system
    system = AmericanTF(config)
    pnl, units = system.run(close, high, low)

    # Calculate performance metrics
    valid_pnl = pnl[~np.isnan(pnl)]
    total_pnl = np.sum(valid_pnl)
    annual_return = np.mean(valid_pnl) * 260
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(260)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Trading statistics
    position_changes = np.sum(np.abs(np.diff(units)) > 0)
    max_units = np.max(np.abs(units))
    avg_holding_period = len(units) / position_changes if position_changes > 0 else 0

    print(f"Results ({len(valid_pnl)} trading days):")
    print(f"  Annual return: {annual_return:.2%}")
    print(f"  Annual volatility: {annual_vol:.2%}")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Total P&L: {total_pnl:.2f}")
    print(f"  Position changes: {position_changes}")
    print(f"  Max units: {max_units:.3f}")
    print(f"  Avg holding period: {avg_holding_period:.1f} days")
    print()

    return pnl, units, close


def run_tsmom_example():
    """Demonstrate TSMOM system."""
    print("=" * 60)
    print("TSMOM SYSTEM EXAMPLE")
    print("=" * 60)

    # Generate sample data (need more for TSMOM blocks)
    prices, _, _ = generate_sample_data(1200)

    # Configure TSMOM system
    config = TSMOMConfig(
        sigma_target_annual=0.12,  # Target 12% annual volatility
        a=260,  # 260 trading days per year
        span_sigma=33,  # Volatility estimation span
        L=10,  # 10-day blocks
        M=12,  # Average over 12 blocks
    )

    print(f"Configuration:")
    print(f"  Target vol: {config.sigma_target_annual:.1%}")
    print(f"  Block length (L): {config.L} days")
    print(f"  Number of blocks (M): {config.M}")
    print(f"  Lookback period: {config.L * config.M} days")
    print()

    # Run the system
    system = TSMOM(config)
    pnl, weights, signal_grid, volatility = system.run_from_prices(prices)

    # Calculate performance metrics
    valid_pnl = pnl[~np.isnan(pnl)]
    annual_return = np.mean(valid_pnl) * config.a
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(config.a)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Signal statistics
    non_zero_signals = signal_grid[signal_grid != 0]
    signal_changes = np.sum(np.abs(np.diff(signal_grid)) > 0)

    print(f"Results ({len(valid_pnl)} trading days):")
    print(f"  Annual return: {annual_return:.2%}")
    print(f"  Annual volatility: {annual_vol:.2%}")
    print(f"  Sharpe ratio: {sharpe_ratio:.2f}")
    print(f"  Total P&L: {np.sum(valid_pnl):.2f}")
    print(f"  Signal changes: {signal_changes}")
    print(f"  Non-zero signals: {len(non_zero_signals)}")
    if len(non_zero_signals) > 0:
        print(f"  Avg signal strength: {np.mean(np.abs(non_zero_signals)):.2f}")
    print()

    return pnl, weights, signal_grid, prices


def create_comparison_plot(eu_pnl, am_pnl, ts_pnl, prices):
    """Create a comparison plot of all three systems."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    try:
        # Calculate cumulative P&L
        eu_cum = np.nancumsum(eu_pnl)
        am_cum = np.nancumsum(am_pnl)
        ts_cum = np.nancumsum(ts_pnl)

        # Normalize price for comparison
        price_norm = (prices / prices[0] - 1) * 100

        plt.figure(figsize=(12, 8))

        # Plot cumulative P&L
        plt.subplot(2, 1, 1)
        plt.plot(eu_cum, label="European TF", linewidth=2)
        plt.plot(am_cum, label="American TF", linewidth=2)
        plt.plot(ts_cum, label="TSMOM", linewidth=2)
        plt.title("Cumulative P&L Comparison")
        plt.ylabel("Cumulative P&L")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot underlying price
        plt.subplot(2, 1, 2)
        plt.plot(price_norm, label="Price (normalized)", color="black", alpha=0.7)
        plt.title("Underlying Price Movement")
        plt.ylabel("Price Change (%)")
        plt.xlabel("Trading Days")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("examples/basic_usage_comparison.png", dpi=150, bbox_inches="tight")
        plt.show()

        print("Comparison plot saved as 'examples/basic_usage_comparison.png'")

    except Exception as e:
        print(f"Error creating plot: {e}")


def main():
    """Run all examples and create comparison."""
    print("TFUNIFY BASIC USAGE EXAMPLES")
    print("This example demonstrates all three trend-following systems")
    print("with synthetic data that includes trending behavior.\n")

    # Check dependencies first
    if not check_requirements():
        print("\nExample cannot run due to missing dependencies.")
        return 1

    # Run all systems
    eu_pnl, eu_weights, eu_signal, prices = run_european_tf_example()
    am_pnl, am_units, am_close = run_american_tf_example()
    ts_pnl, ts_weights, ts_signal, ts_prices = run_tsmom_example()

    # Summary comparison
    print("=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)

    systems = [("European TF", eu_pnl), ("American TF", am_pnl), ("TSMOM", ts_pnl)]

    for name, pnl in systems:
        valid_pnl = pnl[~np.isnan(pnl)]
        if len(valid_pnl) > 0:
            annual_ret = np.mean(valid_pnl) * 260
            annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(260)
            sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
            print(
                f"{name:12s}: Return={annual_ret:6.2%}, Vol={annual_vol:6.2%}, Sharpe={sharpe:5.2f}"
            )

    # Create comparison plot
    if PLOTTING_AVAILABLE:
        print("\nCreating comparison plot...")
        # Use the longest common length for comparison
        min_len = min(len(eu_pnl), len(am_pnl), len(ts_pnl))
        create_comparison_plot(
            eu_pnl[:min_len], am_pnl[:min_len], ts_pnl[:min_len], prices[:min_len]
        )
    else:
        print("\nSkipping plots (matplotlib not available)")

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print("Key takeaways:")
    print("1. Each system has different risk/return characteristics")
    print("2. European TF uses volatility targeting for consistent risk")
    print("3. American TF uses breakouts with ATR-based stops")
    print("4. TSMOM uses block-averaged momentum signals")
    print("5. All systems can be easily configured and backtested")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
