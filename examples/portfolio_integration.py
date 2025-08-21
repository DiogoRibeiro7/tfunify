#!/usr/bin/env python3
"""
Portfolio Integration Example for tfunify

This example demonstrates how to integrate trend-following systems into
broader portfolio management strategies, including:

- Traditional portfolio optimization
- Risk parity approaches
- Ensemble trend-following strategies
- Portfolio overlay strategies
- Risk management and position sizing
"""
import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# Attempt to import tfunify classes, handle missing attributes gracefully
try:
    from tfunify import (
        EuropeanTF, EuropeanTFConfig,
        AmericanTF, AmericanTFConfig, 
        TSMOM, TSMOMConfig,
    )
    TFUNIFY_AVAILABLE = True
except (ImportError, AttributeError) as e:
    TFUNIFY_AVAILABLE = False
    EuropeanTF = None
    EuropeanTFConfig = None
    AmericanTF = None
    AmericanTFConfig = None
    TSMOM = None
    TSMOMConfig = None
    print(f"Error: tfunify package required for portfolio integration.")
    print(f"Install with: pip install tfunify")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy.optimize import minimize

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class AssetData:
    """Container for asset data."""

    name: str
    prices: np.ndarray
    returns: np.ndarray

    def __post_init__(self):
        if self.returns is None and self.prices is not None:
            self.returns = np.diff(np.log(self.prices))


@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""

    name: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    var_95: float
    skewness: float
    beta_to_market: float


def generate_multi_asset_data(n_days: int = 1500, seed: int = 42) -> Dict[str, AssetData]:
    """Generate synthetic multi-asset universe for testing."""
    np.random.seed(seed)

    # Market factor (common to all assets)
    market_returns = 0.0003 + 0.015 * np.random.randn(n_days)
    market_prices = 100 * np.cumprod(1 + np.r_[0.0, market_returns[1:]])

    assets = {}

    # 1. Equity Index (high beta to market)
    equity_returns = 0.0002 + 1.2 * market_returns + 0.008 * np.random.randn(n_days)
    equity_prices = 100 * np.cumprod(1 + np.r_[0.0, equity_returns[1:]])
    assets["Equity"] = AssetData("Equity", equity_prices, equity_returns)

    # 2. Bonds (low beta, negative correlation during stress)
    bond_returns = 0.0001 - 0.3 * market_returns + 0.005 * np.random.randn(n_days)
    bond_prices = 100 * np.cumprod(1 + np.r_[0.0, bond_returns[1:]])
    assets["Bonds"] = AssetData("Bonds", bond_prices, bond_returns)

    # 3. Commodities (moderate beta, inflation hedge)
    commodity_returns = 0.0001 + 0.6 * market_returns + 0.018 * np.random.randn(n_days)
    commodity_prices = 100 * np.cumprod(1 + np.r_[0.0, commodity_returns[1:]])
    assets["Commodities"] = AssetData("Commodities", commodity_prices, commodity_returns)

    # 4. Currency (trending behavior)
    # Add trend component to currency
    currency_trend = np.zeros(n_days)
    for i in range(1, n_days):
        currency_trend[i] = 0.98 * currency_trend[i - 1] + 0.02 * np.random.randn()

    currency_returns = 0.00005 + 0.01 * currency_trend + 0.012 * np.random.randn(n_days)
    currency_prices = 100 * np.cumprod(1 + np.r_[0.0, currency_returns[1:]])
    assets["Currency"] = AssetData("Currency", currency_prices, currency_returns)

    # Store market data for beta calculation
    assets["Market"] = AssetData("Market", market_prices, market_returns)

    return assets


def setup_trend_following_systems() -> Dict[str, Tuple]:
    """Set up ensemble of trend-following systems."""
    systems = {}

    # Conservative European TF
    eu_config = EuropeanTFConfig(
        sigma_target_annual=0.08,  # Low vol target
        a=252,
        span_sigma=44,
        mode="longshort",
        span_long=200,
        span_short=30,
    )
    systems["Conservative TF"] = (EuropeanTF, eu_config)

    # Aggressive European TF
    eu_agg_config = EuropeanTFConfig(
        sigma_target_annual=0.15,  # Higher vol target
        a=252,
        span_sigma=33,
        mode="longshort",
        span_long=150,
        span_short=20,
    )
    systems["Aggressive TF"] = (EuropeanTF, eu_agg_config)

    # Breakout system
    am_config = AmericanTFConfig(
        span_long=100, span_short=20, atr_period=25, q=2.0, p=3.5, r_multiple=0.012
    )
    systems["Breakout"] = (AmericanTF, am_config)

    # Medium-term momentum
    ts_config = TSMOMConfig(sigma_target_annual=0.10, a=252, span_sigma=44, L=12, M=6)
    systems["TSMOM"] = (TSMOM, ts_config)

    return systems


def run_trend_following_ensemble(
    assets: Dict[str, AssetData], systems: Dict[str, Tuple]
) -> Dict[str, Dict[str, np.ndarray]]:
    """Run trend-following systems on multiple assets."""
    print("Running trend-following ensemble...")

    results = {}

    for asset_name, asset_data in assets.items():
        if asset_name == "Market":  # Skip market index
            continue

        print(f"  Processing {asset_name}...")
        asset_results = {}

        for system_name, (system_class, config) in systems.items():
            try:
                if system_name == "Breakout":
                    # American TF needs OHLC - approximate
                    high = asset_data.prices * 1.01
                    low = asset_data.prices * 0.99
                    system = system_class(config)
                    pnl, units = system.run(asset_data.prices, high, low)
                    asset_results[system_name] = {"pnl": pnl, "positions": units}
                else:
                    system = system_class(config)
                    pnl, weights, *_ = system.run_from_prices(asset_data.prices)
                    asset_results[system_name] = {"pnl": pnl, "positions": weights}

            except Exception as e:
                print(f"    Error running {system_name} on {asset_name}: {e}")
                asset_results[system_name] = {
                    "pnl": np.zeros_like(asset_data.prices),
                    "positions": np.zeros_like(asset_data.prices),
                }

        results[asset_name] = asset_results

    return results


def calculate_portfolio_metrics(
    returns: np.ndarray, market_returns: np.ndarray, name: str
) -> PortfolioMetrics:
    """Calculate comprehensive portfolio metrics."""
    valid_returns = returns[~np.isnan(returns)]
    valid_market = market_returns[~np.isnan(market_returns)]

    if len(valid_returns) == 0:
        return PortfolioMetrics(
            name=name,
            total_return=0,
            annual_return=0,
            annual_volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            var_95=0,
            skewness=0,
            beta_to_market=0,
        )

    # Basic metrics
    total_return = np.sum(valid_returns)
    annual_return = np.mean(valid_returns) * 252
    annual_vol = np.std(valid_returns, ddof=0) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Drawdown analysis
    cumulative = np.cumsum(valid_returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    # Risk metrics
    var_95 = np.percentile(valid_returns, 5)

    # Distribution metrics
    try:
        from scipy import stats

        skewness = stats.skew(valid_returns)
    except ImportError:
        skewness = 0

    # Beta calculation
    if len(valid_market) > 0:
        # Align returns for beta calculation
        min_len = min(len(valid_returns), len(valid_market))
        ret_aligned = valid_returns[:min_len]
        mkt_aligned = valid_market[:min_len]

        if len(ret_aligned) > 10 and np.var(mkt_aligned) > 0:
            beta_to_market = np.cov(ret_aligned, mkt_aligned)[0, 1] / np.var(mkt_aligned)
        else:
            beta_to_market = 0
    else:
        beta_to_market = 0

    return PortfolioMetrics(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        var_95=var_95,
        skewness=skewness,
        beta_to_market=beta_to_market,
    )


def create_traditional_portfolio(assets: Dict[str, AssetData]) -> np.ndarray:
    """Create traditional 60/40 portfolio."""
    equity_returns = assets["Equity"].returns
    bond_returns = assets["Bonds"].returns

    # 60% equity, 40% bonds
    portfolio_returns = 0.6 * equity_returns + 0.4 * bond_returns
    return portfolio_returns


def create_risk_parity_portfolio(assets: Dict[str, AssetData], lookback: int = 252) -> np.ndarray:
    """Create risk parity portfolio with rolling rebalancing."""
    asset_names = ["Equity", "Bonds", "Commodities", "Currency"]
    returns_matrix = np.array([assets[name].returns for name in asset_names])

    n_assets, n_periods = returns_matrix.shape
    portfolio_returns = np.zeros(n_periods)

    for t in range(lookback, n_periods):
        # Calculate rolling covariance matrix
        window_returns = returns_matrix[:, t - lookback : t]
        cov_matrix = np.cov(window_returns)

        # Risk parity weights (inverse volatility)
        volatilities = np.sqrt(np.diag(cov_matrix))
        weights = (1 / volatilities) / np.sum(1 / volatilities)

        # Apply weights to next period return
        next_returns = returns_matrix[:, t]
        portfolio_returns[t] = np.dot(weights, next_returns)

    return portfolio_returns


def create_trend_following_overlay(
    assets: Dict[str, AssetData],
    tf_results: Dict[str, Dict[str, np.ndarray]],
    base_portfolio_returns: np.ndarray,
    overlay_allocation: float = 0.2,
) -> np.ndarray:
    """Create portfolio with trend-following overlay."""

    # Aggregate trend-following signals across assets and systems
    total_tf_pnl = np.zeros_like(base_portfolio_returns)

    # Equal weight across assets and systems for simplicity
    n_assets = len([k for k in tf_results.keys() if k != "Market"])
    n_systems = len(list(tf_results.values())[0])

    for asset_name, asset_systems in tf_results.items():
        for system_name, system_results in asset_systems.items():
            pnl = system_results["pnl"]
            # Normalize by asset count and system count
            total_tf_pnl += pnl / (n_assets * n_systems)

    # Combine base portfolio with TF overlay
    combined_returns = (
        1 - overlay_allocation
    ) * base_portfolio_returns + overlay_allocation * total_tf_pnl

    return combined_returns


def optimize_trend_following_allocation(
    assets: Dict[str, AssetData],
    tf_results: Dict[str, Dict[str, np.ndarray]],
    base_portfolio_returns: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Optimize allocation to trend-following strategies."""
    if not SCIPY_AVAILABLE:
        print("SciPy not available. Using fixed 20% allocation.")
        return 0.2, create_trend_following_overlay(assets, tf_results, base_portfolio_returns, 0.2)

    # Aggregate TF returns
    total_tf_pnl = np.zeros_like(base_portfolio_returns)
    n_assets = len([k for k in tf_results.keys() if k != "Market"])
    n_systems = len(list(tf_results.values())[0])

    for asset_name, asset_systems in tf_results.items():
        for system_name, system_results in asset_systems.items():
            pnl = system_results["pnl"]
            total_tf_pnl += pnl / (n_assets * n_systems)

    def objective(allocation):
        """Maximize Sharpe ratio of combined portfolio."""
        allocation = allocation[0]  # Extract scalar from array
        allocation = max(0.0, min(1.0, allocation))  # Constrain to [0,1]

        combined_returns = (1 - allocation) * base_portfolio_returns + allocation * total_tf_pnl

        valid_returns = combined_returns[~np.isnan(combined_returns)]
        if len(valid_returns) < 50:
            return 999.0  # Penalty for insufficient data

        mean_ret = np.mean(valid_returns)
        std_ret = np.std(valid_returns, ddof=0)

        if std_ret == 0:
            return 999.0

        sharpe = mean_ret / std_ret
        return -sharpe  # Minimize negative Sharpe

    # Optimize allocation
    result = minimize(objective, [0.2], bounds=[(0.0, 0.5)], method="L-BFGS-B")

    optimal_allocation = result.x[0]
    optimal_returns = create_trend_following_overlay(
        assets, tf_results, base_portfolio_returns, optimal_allocation
    )

    return optimal_allocation, optimal_returns


def create_ensemble_trend_following(
    tf_results: Dict[str, Dict[str, np.ndarray]], method: str = "equal_weight"
) -> np.ndarray:
    """Create ensemble of trend-following strategies."""

    # Collect all PnL series
    all_pnl = []
    for asset_name, asset_systems in tf_results.items():
        for system_name, system_results in asset_systems.items():
            all_pnl.append(system_results["pnl"])

    if not all_pnl:
        return np.array([])

    # Ensure all series have same length
    min_length = min(len(pnl) for pnl in all_pnl)
    aligned_pnl = [pnl[:min_length] for pnl in all_pnl]
    pnl_matrix = np.array(aligned_pnl)

    if method == "equal_weight":
        # Simple equal weighting
        ensemble_pnl = np.mean(pnl_matrix, axis=0)

    elif method == "sharpe_weighted":
        # Weight by historical Sharpe ratios
        weights = []
        for pnl in aligned_pnl:
            valid_pnl = pnl[~np.isnan(pnl)]
            if len(valid_pnl) > 50:
                mean_ret = np.mean(valid_pnl)
                std_ret = np.std(valid_pnl, ddof=0)
                sharpe = mean_ret / std_ret if std_ret > 0 else 0
                weights.append(max(0, sharpe))  # No negative weights
            else:
                weights.append(0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(aligned_pnl)] * len(aligned_pnl)

        ensemble_pnl = np.average(pnl_matrix, axis=0, weights=weights)

    elif method == "correlation_adjusted":
        # Adjust for correlations (simplified approach)
        correlation_matrix = np.corrcoef(pnl_matrix)

        # Inverse correlation weighting (simplified)
        avg_correlations = np.mean(np.abs(correlation_matrix), axis=1)
        weights = 1.0 / (1.0 + avg_correlations)
        weights = weights / np.sum(weights)

        ensemble_pnl = np.average(pnl_matrix, axis=0, weights=weights)

    else:
        raise ValueError(f"Unknown ensemble method: {method}")

    return ensemble_pnl


def create_comprehensive_plots(
    assets: Dict[str, AssetData],
    portfolios: Dict[str, np.ndarray],
    metrics: Dict[str, PortfolioMetrics],
) -> None:
    """Create comprehensive portfolio analysis plots."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle("Portfolio Integration Analysis", fontsize=16)

    # 1. Cumulative Performance
    ax1 = axes[0, 0]
    for name, returns in portfolios.items():
        cumulative = np.cumsum(returns)
        ax1.plot(cumulative, label=name, linewidth=2)

    ax1.set_title("Cumulative Performance Comparison")
    ax1.set_ylabel("Cumulative Return")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Risk-Return Scatter
    ax2 = axes[0, 1]
    for name, metric in metrics.items():
        ax2.scatter(metric.annual_volatility, metric.annual_return, label=name, s=100, alpha=0.7)

    ax2.set_title("Risk-Return Profile")
    ax2.set_xlabel("Annual Volatility")
    ax2.set_ylabel("Annual Return")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Sharpe Ratio Comparison
    ax3 = axes[1, 0]
    names = list(metrics.keys())
    sharpes = [metrics[name].sharpe_ratio for name in names]
    colors = plt.cm.viridis(np.linspace(0, 1, len(names)))

    bars = ax3.bar(names, sharpes, color=colors)
    ax3.set_title("Sharpe Ratio Comparison")
    ax3.set_ylabel("Sharpe Ratio")
    ax3.tick_params(axis="x", rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # Add value labels on bars
    for bar, sharpe in zip(bars, sharpes):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{sharpe:.2f}",
            ha="center",
            va="bottom",
        )

    # 4. Maximum Drawdown Comparison
    ax4 = axes[1, 1]
    drawdowns = [metrics[name].max_drawdown for name in names]
    bars = ax4.bar(names, drawdowns, color=colors)
    ax4.set_title("Maximum Drawdown Comparison")
    ax4.set_ylabel("Maximum Drawdown")
    ax4.tick_params(axis="x", rotation=45)
    ax4.grid(True, alpha=0.3)

    # 5. Rolling Sharpe Ratios
    ax5 = axes[2, 0]
    window = 252  # 1-year rolling window

    for name, returns in portfolios.items():
        if len(returns) > window:
            rolling_sharpe = []
            for i in range(window, len(returns)):
                window_returns = returns[i - window : i]
                valid_returns = window_returns[~np.isnan(window_returns)]
                if len(valid_returns) > window * 0.8:  # At least 80% valid data
                    mean_ret = np.mean(valid_returns) * 252
                    std_ret = np.std(valid_returns, ddof=0) * np.sqrt(252)
                    sharpe = mean_ret / std_ret if std_ret > 0 else 0
                    rolling_sharpe.append(sharpe)
                else:
                    rolling_sharpe.append(np.nan)

            ax5.plot(range(window, len(returns)), rolling_sharpe, label=name, linewidth=2)

    ax5.set_title("Rolling 1-Year Sharpe Ratio")
    ax5.set_ylabel("Sharpe Ratio")
    ax5.set_xlabel("Time")
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    # 6. Portfolio Metrics Summary
    ax6 = axes[2, 1]
    ax6.axis("off")

    # Create summary table
    summary_text = "Portfolio Summary Metrics\n\n"
    summary_text += f"{'Portfolio':<20} {'Sharpe':<8} {'Vol':<8} {'MaxDD':<8} {'Beta':<6}\n"
    summary_text += "-" * 55 + "\n"

    for name, metric in metrics.items():
        summary_text += (
            f"{name:<20} {metric.sharpe_ratio:<8.2f} "
            f"{metric.annual_volatility:<8.2%} "
            f"{metric.max_drawdown:<8.2f} "
            f"{metric.beta_to_market:<6.2f}\n"
        )

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout()
    plt.savefig("examples/portfolio_integration.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("Portfolio integration plots saved to: examples/portfolio_integration.png")


def main():
    """Main function for portfolio integration example."""
    print("TFUNIFY PORTFOLIO INTEGRATION EXAMPLE")
    print("Comprehensive portfolio strategies with trend-following integration")
    print("=" * 75)

    # Check core dependencies first
    if not TFUNIFY_AVAILABLE:
        print("\nError: Cannot run without tfunify package.")
        print("Install with: pip install tfunify")
        return 1

    # Step 1: Generate multi-asset universe
    print("\nStep 1: Generating multi-asset universe...")
    assets = generate_multi_asset_data(n_days=1800, seed=42)

    print("Generated assets:")
    for name, asset in assets.items():
        if name != "Market":
            annual_ret = np.mean(asset.returns) * 252
            annual_vol = np.std(asset.returns) * np.sqrt(252)
            print(f"  {name:<12}: Return={annual_ret:6.2%}, Vol={annual_vol:6.2%}")

    # Step 2: Set up trend-following systems
    print("\nStep 2: Setting up trend-following ensemble...")
    tf_systems = setup_trend_following_systems()

    print("Trend-following systems:")
    for name, (system_class, config) in tf_systems.items():
        print(f"  {name}: {system_class.__name__}")

    # Step 3: Run trend-following systems
    print("\nStep 3: Running trend-following systems...")
    tf_results = run_trend_following_ensemble(assets, tf_systems)

    # Step 4: Create portfolio strategies
    print("\nStep 4: Creating portfolio strategies...")
    portfolios = {}

    # Traditional 60/40 portfolio
    traditional_returns = create_traditional_portfolio(assets)
    portfolios["60/40 Traditional"] = traditional_returns

    # Risk parity portfolio
    risk_parity_returns = create_risk_parity_portfolio(assets)
    portfolios["Risk Parity"] = risk_parity_returns

    # TF overlay on traditional portfolio
    tf_overlay_returns = create_trend_following_overlay(
        assets, tf_results, traditional_returns, overlay_allocation=0.2
    )
    portfolios["60/40 + TF Overlay"] = tf_overlay_returns

    # Optimized TF allocation
    optimal_allocation, optimal_returns = optimize_trend_following_allocation(
        assets, tf_results, traditional_returns
    )
    portfolios[f"60/40 + TF Optimal ({optimal_allocation:.1%})"] = optimal_returns

    # Pure ensemble TF strategies
    ensemble_equal = create_ensemble_trend_following(tf_results, "equal_weight")
    if len(ensemble_equal) > 0:
        portfolios["TF Ensemble (Equal)"] = ensemble_equal

    ensemble_sharpe = create_ensemble_trend_following(tf_results, "sharpe_weighted")
    if len(ensemble_sharpe) > 0:
        portfolios["TF Ensemble (Sharpe)"] = ensemble_sharpe

    # Step 5: Calculate performance metrics
    print("\nStep 5: Calculating performance metrics...")
    market_returns = assets["Market"].returns
    metrics = {}

    for name, returns in portfolios.items():
        metrics[name] = calculate_portfolio_metrics(returns, market_returns, name)

    # Step 6: Performance analysis
    print("\n" + "=" * 75)
    print("PORTFOLIO PERFORMANCE SUMMARY")
    print("=" * 75)

    print(f"{'Portfolio':<25} {'Ann Ret':<8} {'Ann Vol':<8} {'Sharpe':<8} {'MaxDD':<8} {'Beta':<6}")
    print("-" * 75)

    for name, metric in metrics.items():
        print(
            f"{name:<25} "
            f"{metric.annual_return:<8.2%} "
            f"{metric.annual_volatility:<8.2%} "
            f"{metric.sharpe_ratio:<8.2f} "
            f"{metric.max_drawdown:<8.2f} "
            f"{metric.beta_to_market:<6.2f}"
        )

    # Step 7: Risk analysis
    print(f"\n" + "=" * 50)
    print("RISK ANALYSIS")
    print("=" * 50)

    print(f"{'Portfolio':<25} {'VaR 95%':<10} {'Skewness':<10} {'Calmar':<8}")
    print("-" * 60)

    for name, metric in metrics.items():
        calmar_display = f"{metric.calmar_ratio:.2f}" if metric.calmar_ratio != np.inf else "∞"
        print(f"{name:<25} {metric.var_95:<10.4f} {metric.skewness:<10.2f} {calmar_display:<8}")

    # Step 8: Create comprehensive plots
    print("\nStep 8: Creating portfolio analysis plots...")
    create_comprehensive_plots(assets, portfolios, metrics)

    # Step 9: Key insights and recommendations
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)

    # Find best portfolio by Sharpe ratio
    best_portfolio = max(metrics.items(), key=lambda x: x[1].sharpe_ratio)
    best_name, best_metrics = best_portfolio

    print(f"1. Best risk-adjusted performance: {best_name}")
    print(f"   - Sharpe ratio: {best_metrics.sharpe_ratio:.3f}")
    print(f"   - Annual return: {best_metrics.annual_return:.2%}")
    print(f"   - Annual volatility: {best_metrics.annual_volatility:.2%}")

    # Compare with traditional portfolio
    traditional_sharpe = metrics["60/40 Traditional"].sharpe_ratio
    improvement = best_metrics.sharpe_ratio - traditional_sharpe
    print(f"2. Improvement over traditional 60/40: {improvement:+.3f} Sharpe")

    # Analyze diversification benefits
    tf_overlay_beta = metrics.get("60/40 + TF Overlay", metrics["60/40 Traditional"]).beta_to_market
    traditional_beta = metrics["60/40 Traditional"].beta_to_market
    print(f"3. Beta reduction from TF overlay: {traditional_beta:.3f} → {tf_overlay_beta:.3f}")

    print(f"4. Optimal TF allocation: {optimal_allocation:.1%}")

    print("\nRecommendations:")
    print("- Trend-following strategies provide valuable diversification")
    print("- Ensemble approaches reduce single-system risk")
    print("- Overlay strategies can improve risk-adjusted returns")
    print("- Regular rebalancing and allocation optimization recommended")
    print("- Consider market regime detection for dynamic allocation")

    print(f"\nPortfolio integration analysis complete!")
    print(f"Check 'examples/portfolio_integration.png' for detailed charts.")


if __name__ == "__main__":
    main()
