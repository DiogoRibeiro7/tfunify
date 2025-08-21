#!/usr/bin/env python3
"""
Real Data Analysis Example for tfunify

This example demonstrates how to use tfunify with real market data from Yahoo Finance.
It shows data downloading, preprocessing, and analysis with all three systems.

Requirements:
    pip install tfunify[yahoo] matplotlib pandas
"""

import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# ===== CORE TFUNIFY IMPORTS =====
try:
    from tfunify import EuropeanTF, EuropeanTFConfig, AmericanTF, AmericanTFConfig, TSMOM, TSMOMConfig
    TFUNIFY_AVAILABLE = True
except ImportError as e:
    TFUNIFY_AVAILABLE = False
    print(f"Error: tfunify package required for this example.")
    print(f"Install with: pip install tfunify")
    print(f"Error details: {e}")

# ===== OPTIONAL DEPENDENCIES =====

# Yahoo Finance data integration
try:
    from tfunify.data import download_csv, load_csv
    YAHOO_AVAILABLE = True
except ImportError:
    YAHOO_AVAILABLE = False
    print("Warning: Yahoo Finance integration not available.")
    print("Install with: pip install tfunify[yahoo]")

# Data handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. Install with: pip install pandas")

# Plotting support
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Plotting not available. Install with: pip install matplotlib")


def check_requirements():
    """Check if all required dependencies are available."""
    missing = []
    
    if not TFUNIFY_AVAILABLE:
        missing.append("tfunify (core package)")
    if not YAHOO_AVAILABLE:
        missing.append("yfinance (for data download)")
    if not PANDAS_AVAILABLE:
        missing.append("pandas (for data handling)")
    
    optional_missing = []
    if not PLOTTING_AVAILABLE:
        optional_missing.append("matplotlib (for plots)")
    
    if missing:
        print("Missing required dependencies:")
        for dep in missing:
            print(f"  - {dep}")
        print("\nInstall missing dependencies:")
        print("  pip install tfunify[yahoo] pandas")
        return False
    
    if optional_missing:
        print("Missing optional dependencies:")
        for dep in optional_missing:
            print(f"  - {dep}")
        print("  Some features may not be available.")
    
    return True


def download_market_data(symbol: str = "SPY", period: str = "5y") -> dict:
    """Download real market data from Yahoo Finance."""
    if not YAHOO_AVAILABLE:
        raise ImportError(
            "Yahoo Finance integration required. Install with: pip install tfunify[yahoo]"
        )

    print(f"Downloading {symbol} data for period: {period}")

    # Create data directory if it doesn't exist
    data_dir = Path("examples/data")
    data_dir.mkdir(exist_ok=True)

    # Download data
    file_path = data_dir / f"{symbol}_{period}.csv"
    try:
        download_csv(symbol, file_path, period=period)
        print(f"Data saved to: {file_path}")

        # Load and return the data
        data = load_csv(file_path)

        # Also load dates if we can
        if PANDAS_AVAILABLE:
            df = pd.read_csv(file_path)
            data["dates"] = pd.to_datetime(df["date"])
        else:
            # Fallback without pandas
            import csv
            from datetime import datetime
            dates = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dates.append(datetime.strptime(row['date'], '%Y-%m-%d'))
            data["dates"] = dates

        print(
            f"Loaded {len(data['close'])} observations from {data['dates'][0]} to {data['dates'][-1]}"
        )
        return data

    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def analyze_data_quality(data: dict) -> None:
    """Analyze the quality of the market data."""
    print("\n" + "=" * 50)
    print("DATA QUALITY ANALYSIS")
    print("=" * 50)

    close = data["close"]
    high = data["high"]
    low = data["low"]

    # Basic statistics
    print(f"Observations: {len(close):,}")
    if PANDAS_AVAILABLE and hasattr(data['dates'], 'iloc'):
        print(f"Date range: {data['dates'].iloc[0].date()} to {data['dates'].iloc[-1].date()}")
    else:
        print(f"Date range: {data['dates'][0]} to {data['dates'][-1]}")
    print(f"Price range: ${np.min(close):.2f} - ${np.max(close):.2f}")

    # Check for missing data
    missing_close = np.sum(~np.isfinite(close))
    missing_high = np.sum(~np.isfinite(high))
    missing_low = np.sum(~np.isfinite(low))

    print(f"\nMissing data:")
    print(f"  Close: {missing_close} ({missing_close / len(close) * 100:.2f}%)")
    print(f"  High:  {missing_high} ({missing_high / len(high) * 100:.2f}%)")
    print(f"  Low:   {missing_low} ({missing_low / len(low) * 100:.2f}%)")

    # Check OHLC consistency
    invalid_hl = np.sum(high < low)
    invalid_hc = np.sum(high < close)
    invalid_lc = np.sum(low > close)

    print(f"\nOHLC consistency:")
    print(f"  High < Low: {invalid_hl} violations")
    print(f"  High < Close: {invalid_hc} violations")
    print(f"  Low > Close: {invalid_lc} violations")

    # Calculate daily returns and volatility
    returns = np.diff(np.log(close))
    daily_vol = np.std(returns) * 100
    annual_vol = daily_vol * np.sqrt(252)

    print(f"\nVolatility analysis:")
    print(f"  Daily volatility: {daily_vol:.2f}%")
    print(f"  Annualized volatility: {annual_vol:.1f}%")
    print(f"  Max daily return: {np.max(returns) * 100:+.2f}%")
    print(f"  Min daily return: {np.min(returns) * 100:+.2f}%")


def run_comprehensive_backtest(data: dict) -> dict:
    """Run all three systems on real market data."""
    print("\n" + "=" * 50)
    print("COMPREHENSIVE BACKTEST")
    print("=" * 50)

    prices = data["close"]
    high = data["high"]
    low = data["low"]

    results = {}

    # European TF - Conservative configuration
    print("\n1. European TF System")
    print("-" * 30)
    eu_config = EuropeanTFConfig(
        sigma_target_annual=0.10,  # 10% target vol (conservative)
        a=252,  # 252 trading days (market standard)
        span_sigma=44,  # ~2 months volatility estimation
        mode="longshort",  # Long-short filter
        span_long=252,  # 1 year slow filter
        span_short=44,  # ~2 months fast filter
    )

    print(f"  Target volatility: {eu_config.sigma_target_annual:.1%}")
    print(f"  Lookback periods: {eu_config.span_short} / {eu_config.span_long} days")

    eu_system = EuropeanTF(eu_config)
    eu_pnl, eu_weights, eu_signal, eu_vol = eu_system.run_from_prices(prices)
    results["european"] = {
        "pnl": eu_pnl,
        "weights": eu_weights,
        "signal": eu_signal,
        "config": eu_config,
    }

    # American TF - Breakout configuration
    print("\n2. American TF System")
    print("-" * 30)
    am_config = AmericanTFConfig(
        span_long=200,  # ~8 months trend filter
        span_short=50,  # ~2 months fast filter
        atr_period=20,  # 1 month ATR
        q=2.0,  # Moderate entry threshold
        p=4.0,  # Wider stop loss
        r_multiple=0.01,  # 1% risk per trade
    )

    print(f"  Entry threshold: {am_config.q} × ATR")
    print(f"  Stop loss: {am_config.p} × ATR")
    print(f"  Risk per trade: {am_config.r_multiple:.1%}")

    am_system = AmericanTF(am_config)
    am_pnl, am_units = am_system.run(prices, high, low)
    results["american"] = {"pnl": am_pnl, "units": am_units, "config": am_config}

    # TSMOM - Medium-term momentum
    print("\n3. TSMOM System")
    print("-" * 30)
    ts_config = TSMOMConfig(
        sigma_target_annual=0.12,  # 12% target vol
        a=252,
        span_sigma=44,  # ~2 months volatility
        L=20,  # 1 month blocks
        M=6,  # 6 months lookback
    )

    print(f"  Target volatility: {ts_config.sigma_target_annual:.1%}")
    print(f"  Block structure: {ts_config.L} day blocks, {ts_config.M} blocks lookback")
    print(f"  Total lookback: {ts_config.L * ts_config.M} days")

    ts_system = TSMOM(ts_config)
    ts_pnl, ts_weights, ts_signals, ts_vol = ts_system.run_from_prices(prices)
    results["tsmom"] = {
        "pnl": ts_pnl,
        "weights": ts_weights,
        "signals": ts_signals,
        "config": ts_config,
    }

    return results


def calculate_performance_metrics(pnl: np.ndarray, name: str) -> dict:
    """Calculate comprehensive performance metrics."""
    valid_pnl = pnl[~np.isnan(pnl)]

    if len(valid_pnl) == 0:
        return {"name": name, "valid_days": 0}

    # Basic metrics
    total_return = np.sum(valid_pnl)
    annual_return = np.mean(valid_pnl) * 252
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Risk metrics
    cumulative_pnl = np.cumsum(valid_pnl)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = cumulative_pnl - running_max
    max_drawdown = np.min(drawdown)

    # Win/loss statistics
    positive_days = np.sum(valid_pnl > 0)
    negative_days = np.sum(valid_pnl < 0)
    win_rate = positive_days / len(valid_pnl) if len(valid_pnl) > 0 else 0

    # Average win/loss
    wins = valid_pnl[valid_pnl > 0]
    losses = valid_pnl[valid_pnl < 0]
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf

    # Calmar ratio
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else np.inf

    return {
        "name": name,
        "valid_days": len(valid_pnl),
        "total_return": total_return,
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
        "positive_days": positive_days,
        "negative_days": negative_days,
    }


def print_performance_summary(results: dict) -> None:
    """Print comprehensive performance summary."""
    print("\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    metrics = []
    for system_name, system_results in results.items():
        pnl = system_results["pnl"]
        metrics.append(calculate_performance_metrics(pnl, system_name.title()))

    # Print header
    print(
        f"{'System':<12} {'Annual Ret':<10} {'Annual Vol':<10} {'Sharpe':<8} {'Max DD':<10} {'Calmar':<8} {'Win Rate':<9}"
    )
    print("-" * 80)

    # Print metrics for each system
    for m in metrics:
        if m["valid_days"] > 0:
            print(
                f"{m['name']:<12} "
                f"{m['annual_return']:>9.2%} "
                f"{m['annual_volatility']:>9.2%} "
                f"{m['sharpe_ratio']:>7.2f} "
                f"{m['max_drawdown']:>9.2f} "
                f"{m['calmar_ratio']:>7.1f} "
                f"{m['win_rate']:>8.1%}"
            )

    # Detailed breakdown
    print("\n" + "=" * 50)
    print("DETAILED BREAKDOWN")
    print("=" * 50)

    for m in metrics:
        if m["valid_days"] > 0:
            print(f"\n{m['name']} System:")
            print(f"  Valid trading days: {m['valid_days']:,}")
            print(f"  Total P&L: {m['total_return']:,.2f}")
            print(f"  Annual return: {m['annual_return']:.2%}")
            print(f"  Annual volatility: {m['annual_volatility']:.2%}")
            print(f"  Sharpe ratio: {m['sharpe_ratio']:.2f}")
            print(f"  Maximum drawdown: {m['max_drawdown']:.2f}")
            print(f"  Calmar ratio: {m['calmar_ratio']:.1f}")
            print(
                f"  Win rate: {m['win_rate']:.1%} ({m['positive_days']} wins, {m['negative_days']} losses)"
            )
            print(f"  Average win: {m['avg_win']:.3f}")
            print(f"  Average loss: {m['avg_loss']:.3f}")
            print(f"  Win/Loss ratio: {m['win_loss_ratio']:.2f}")


def create_performance_plots(data: dict, results: dict) -> None:
    """Create comprehensive performance plots."""
    if not PLOTTING_AVAILABLE:
        print("Plotting not available. Install matplotlib to see charts.")
        return

    dates = data["dates"]
    prices = data["close"]

    # Create figure with subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Real Market Data Analysis - Trend Following Systems", fontsize=16)

    # 1. Price chart
    ax1 = axes[0, 0]
    ax1.plot(dates, prices, "k-", linewidth=1)
    ax1.set_title("Price Chart")
    ax1.set_ylabel("Price ($)")
    ax1.grid(True, alpha=0.3)
    if hasattr(mdates, 'DateFormatter'):
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 2. Cumulative P&L
    ax2 = axes[0, 1]
    for system_name, system_results in results.items():
        pnl = system_results["pnl"]
        valid_mask = ~np.isnan(pnl)
        cum_pnl = np.cumsum(np.where(valid_mask, pnl, 0))
        ax2.plot(dates, cum_pnl, label=system_name.title(), linewidth=2)

    ax2.set_title("Cumulative P&L")
    ax2.set_ylabel("Cumulative P&L")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    if hasattr(mdates, 'DateFormatter'):
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # 3. European TF weights
    ax3 = axes[1, 0]
    eu_weights = results["european"]["weights"]
    valid_mask = ~np.isnan(eu_weights)
    if hasattr(dates, '__getitem__'):
        ax3.plot(np.array(dates)[valid_mask], eu_weights[valid_mask], "b-", linewidth=1)
    else:
        ax3.plot(range(len(eu_weights[valid_mask])), eu_weights[valid_mask], "b-", linewidth=1)
    ax3.set_title("European TF - Position Weights")
    ax3.set_ylabel("Weight")
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # 4. American TF units
    ax4 = axes[1, 1]
    am_units = results["american"]["units"]
    valid_mask = ~np.isnan(am_units)
    if hasattr(dates, '__getitem__'):
        ax4.plot(np.array(dates)[valid_mask], am_units[valid_mask], "r-", linewidth=1)
    else:
        ax4.plot(range(len(am_units[valid_mask])), am_units[valid_mask], "r-", linewidth=1)
    ax4.set_title("American TF - Position Units")
    ax4.set_ylabel("Units")
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # 5. TSMOM signals
    ax5 = axes[2, 0]
    ts_signals = results["tsmom"]["signals"]
    valid_mask = ~np.isnan(ts_signals)
    if hasattr(dates, '__getitem__'):
        ax5.plot(np.array(dates)[valid_mask], ts_signals[valid_mask], "g-", linewidth=1)
    else:
        ax5.plot(range(len(ts_signals[valid_mask])), ts_signals[valid_mask], "g-", linewidth=1)
    ax5.set_title("TSMOM - Signal Grid")
    ax5.set_ylabel("Signal Strength")
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # 6. Rolling Sharpe ratios
    ax6 = axes[2, 1]
    window = 252  # 1 year rolling window

    for system_name, system_results in results.items():
        pnl = system_results["pnl"]
        valid_mask = ~np.isnan(pnl)

        if np.sum(valid_mask) > window:
            rolling_sharpe = []
            valid_dates = []

            for i in range(window, len(pnl)):
                if np.sum(valid_mask[i - window : i]) >= window * 0.8:  # At least 80% valid data
                    window_pnl = pnl[i - window : i]
                    window_valid = window_pnl[~np.isnan(window_pnl)]

                    if len(window_valid) > 0:
                        mean_ret = np.mean(window_valid) * 252
                        vol_ret = np.std(window_valid, ddof=0) * np.sqrt(252)
                        sharpe = mean_ret / vol_ret if vol_ret > 0 else 0
                        rolling_sharpe.append(sharpe)
                        if hasattr(dates, 'iloc'):
                            valid_dates.append(dates.iloc[i])
                        else:
                            valid_dates.append(dates[i])

            if rolling_sharpe:
                ax6.plot(valid_dates, rolling_sharpe, label=system_name.title(), linewidth=2)

    ax6.set_title("Rolling 1-Year Sharpe Ratio")
    ax6.set_ylabel("Sharpe Ratio")
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=0, color="k", linestyle="--", alpha=0.5)
    if hasattr(mdates, 'DateFormatter'):
        ax6.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Format x-axes
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()

    # Save plot
    output_path = Path("examples/real_data_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nPerformance plots saved to: {output_path}")


def analyze_market_regimes(data: dict, results: dict) -> None:
    """Analyze system performance across different market regimes."""
    print("\n" + "=" * 50)
    print("MARKET REGIME ANALYSIS")
    print("=" * 50)

    prices = data["close"]
    dates = data["dates"]

    # Define market regimes based on rolling returns
    window = 60  # 3-month windows
    rolling_returns = []
    regime_dates = []

    for i in range(window, len(prices)):
        ret = (prices[i] / prices[i - window] - 1) * 100
        rolling_returns.append(ret)
        if hasattr(dates, 'iloc'):
            regime_dates.append(dates.iloc[i])
        else:
            regime_dates.append(dates[i])

    rolling_returns = np.array(rolling_returns)

    # Define regimes (simple classification)
    bull_threshold = 5  # > 5% in 3 months
    bear_threshold = -5  # < -5% in 3 months

    bull_mask = rolling_returns > bull_threshold
    bear_mask = rolling_returns < bear_threshold
    sideways_mask = ~(bull_mask | bear_mask)

    regimes = {"Bull Market": bull_mask, "Bear Market": bear_mask, "Sideways Market": sideways_mask}

    print(f"Market regime classification (based on 3-month rolling returns):")
    print(
        f"  Bull markets (>{bull_threshold}%): {np.sum(bull_mask)} periods ({np.sum(bull_mask) / len(bull_mask) * 100:.1f}%)"
    )
    print(
        f"  Bear markets (<{bear_threshold}%): {np.sum(bear_mask)} periods ({np.sum(bear_mask) / len(bear_mask) * 100:.1f}%)"
    )
    print(
        f"  Sideways markets: {np.sum(sideways_mask)} periods ({np.sum(sideways_mask) / len(sideways_mask) * 100:.1f}%)"
    )

    # Analyze system performance by regime
    print(f"\nSystem performance by market regime:")
    print(f"{'System':<12} {'Regime':<15} {'Ann. Return':<12} {'Ann. Vol':<10} {'Sharpe':<8}")
    print("-" * 65)

    for system_name, system_results in results.items():
        pnl = system_results["pnl"]

        # Align PnL with regime periods (skip warmup period)
        if len(pnl) > window:
            aligned_pnl = pnl[window:]

            for regime_name, regime_mask in regimes.items():
                if np.sum(regime_mask) > 0:
                    regime_pnl = aligned_pnl[regime_mask]
                    valid_pnl = regime_pnl[~np.isnan(regime_pnl)]

                    if len(valid_pnl) > 0:
                        ann_ret = np.mean(valid_pnl) * 252
                        ann_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

                        print(
                            f"{system_name.title():<12} {regime_name:<15} "
                            f"{ann_ret:>11.2%} {ann_vol:>9.2%} {sharpe:>7.2f}"
                        )


def main():
    """Main function to run real data analysis."""
    print("TFUNIFY REAL MARKET DATA ANALYSIS")
    print("This example analyzes trend-following systems using real market data")
    print("from Yahoo Finance.\n")

    # Check dependencies first
    if not check_requirements():
        print("\nExample cannot run due to missing dependencies.")
        return 1

    # Configuration
    SYMBOL = "SPY"  # S&P 500 ETF
    PERIOD = "5y"  # 5 years of data

    try:
        # Step 1: Download and analyze data
        print("Step 1: Downloading market data...")
        data = download_market_data(SYMBOL, PERIOD)
        analyze_data_quality(data)

        # Step 2: Run backtests
        print("\nStep 2: Running comprehensive backtests...")
        results = run_comprehensive_backtest(data)

        # Step 3: Performance analysis
        print_performance_summary(results)

        # Step 4: Market regime analysis
        analyze_market_regimes(data, results)

        # Step 5: Create plots
        print("\nStep 5: Creating performance plots...")
        create_performance_plots(data, results)

        # Summary insights
        print("\n" + "=" * 50)
        print("KEY INSIGHTS")
        print("=" * 50)
        print("1. European TF: Smooth volatility-targeted approach")
        print("2. American TF: Event-driven with clear entry/exit rules")
        print("3. TSMOM: Block-based momentum with medium-term signals")
        print("4. Each system responds differently to market regimes")
        print("5. Risk-adjusted returns vary significantly by approach")

        print("\nAnalysis complete! Check 'examples/data/' for downloaded data")
        if PLOTTING_AVAILABLE:
            print("and 'examples/real_data_analysis.png' for performance charts.")
        else:
            print("Install matplotlib to generate performance charts.")

        return 0

    except (ImportError, ValueError) as e:
        print(f"Error in analysis: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you have installed:")
        print("   pip install tfunify[yahoo] matplotlib pandas")
        print("2. Check your internet connection for data download")
        print("3. Try a different symbol or shorter period")
        print("   if data issues persist")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
