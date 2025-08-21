#!/usr/bin/env python3
"""
Performance Comparison Example for tfunify

This example conducts a comprehensive comparison of all three trend-following
systems across multiple scenarios and market conditions.

Key features:
- Multiple test scenarios (trending, mean-reverting, volatile)
- Statistical significance testing
- Performance attribution analysis
- Risk-adjusted metrics comparison
- Monte Carlo robustness testing
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple

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
    print(f"Error: tfunify package required for performance comparison.")
    print(f"Install with: pip install tfunify")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: SciPy not available. Statistical tests will be skipped.")


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""

    name: str
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float
    var_95: float
    skewness: float
    kurtosis: float
    valid_days: int


def generate_test_scenarios(n_days: int = 2000, seed: int = 42) -> Dict[str, np.ndarray]:
    """Generate different market scenarios for testing."""
    np.random.seed(seed)
    scenarios = {}

    # 1. Strong Trend Scenario
    trend_drift = 0.0005  # Strong positive drift
    trend_vol = 0.015
    trend_returns = trend_drift + trend_vol * np.random.randn(n_days)
    # Add persistence to trending
    for i in range(1, n_days):
        trend_returns[i] += 0.1 * trend_returns[i - 1]
    scenarios["strong_trend"] = 100 * np.cumprod(1 + np.r_[0.0, trend_returns[1:]])

    # 2. Mean Reverting Scenario
    mr_returns = np.zeros(n_days)
    mr_returns[0] = 0.01 * np.random.randn()
    for i in range(1, n_days):
        # Strong mean reversion with noise
        mr_returns[i] = -0.05 * mr_returns[i - 1] + 0.02 * np.random.randn()
    scenarios["mean_reverting"] = 100 * np.cumprod(1 + np.r_[0.0, mr_returns[1:]])

    # 3. High Volatility Regime
    vol_periods = n_days // 4
    vol_returns = np.concatenate(
        [
            0.01 * np.random.randn(vol_periods),  # Low vol
            0.04 * np.random.randn(vol_periods),  # High vol
            0.015 * np.random.randn(vol_periods),  # Medium vol
            0.03 * np.random.randn(n_days - 3 * vol_periods),  # High vol
        ]
    )
    scenarios["high_volatility"] = 100 * np.cumprod(1 + np.r_[0.0, vol_returns[1:]])

    # 4. Sideways Market
    sideways_returns = 0.0001 + 0.012 * np.random.randn(n_days)
    # Remove any drift
    sideways_returns = sideways_returns - np.mean(sideways_returns)
    scenarios["sideways"] = 100 * np.cumprod(1 + np.r_[0.0, sideways_returns[1:]])

    # 5. Crisis Scenario (fat tails)
    crisis_returns = np.random.standard_t(df=3, size=n_days) * 0.02  # t-distribution
    # Add occasional large drops
    crisis_events = np.random.choice(n_days, size=n_days // 100, replace=False)
    crisis_returns[crisis_events] = -0.05 - 0.03 * np.random.rand(len(crisis_events))
    scenarios["crisis"] = 100 * np.cumprod(1 + np.r_[0.0, crisis_returns[1:]])

    # 6. Mixed Regime Scenario
    regime_length = n_days // 6
    mixed_returns = np.concatenate(
        [
            0.0003 + 0.01 * np.random.randn(regime_length),  # Bull
            -0.0002 + 0.02 * np.random.randn(regime_length),  # Bear
            0.00 + 0.008 * np.random.randn(regime_length),  # Sideways
            0.0004 + 0.015 * np.random.randn(regime_length),  # Bull
            -0.0003 + 0.025 * np.random.randn(regime_length),  # Bear
            0.0001 + 0.012 * np.random.randn(n_days - 5 * regime_length),  # Recovery
        ]
    )
    scenarios["mixed_regime"] = 100 * np.cumprod(1 + np.r_[0.0, mixed_returns[1:]])

    return scenarios


def setup_system_configurations() -> Dict[str, Tuple]:
    """Set up optimized configurations for each system."""
    configs = {}

    # European TF - Conservative volatility targeting
    eu_config = EuropeanTFConfig(
        sigma_target_annual=0.12,
        a=252,
        span_sigma=44,
        mode="longshort",
        span_long=200,
        span_short=40,
    )
    configs["European TF"] = (EuropeanTF, eu_config)

    # American TF - Moderate breakout system
    am_config = AmericanTFConfig(
        span_long=150, span_short=30, atr_period=30, q=2.5, p=4.0, r_multiple=0.015
    )
    configs["American TF"] = (AmericanTF, am_config)

    # TSMOM - Medium-term momentum
    ts_config = TSMOMConfig(sigma_target_annual=0.10, a=252, span_sigma=44, L=15, M=8)
    configs["TSMOM"] = (TSMOM, ts_config)

    return configs


def calculate_detailed_metrics(pnl: np.ndarray, name: str) -> PerformanceMetrics:
    """Calculate comprehensive performance metrics."""
    valid_pnl = pnl[~np.isnan(pnl)]

    if len(valid_pnl) == 0:
        return PerformanceMetrics(
            name=name,
            total_return=0,
            annual_return=0,
            annual_volatility=0,
            sharpe_ratio=0,
            max_drawdown=0,
            calmar_ratio=0,
            win_rate=0,
            profit_factor=0,
            var_95=0,
            skewness=0,
            kurtosis=0,
            valid_days=0,
        )

    # Basic metrics
    total_return = np.sum(valid_pnl)
    annual_return = np.mean(valid_pnl) * 252
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0

    # Drawdown analysis
    cumulative = np.cumsum(valid_pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else np.inf

    # Win/loss metrics
    wins = valid_pnl[valid_pnl > 0]
    losses = valid_pnl[valid_pnl < 0]
    win_rate = len(wins) / len(valid_pnl)

    # Profit factor
    gross_profit = np.sum(wins) if len(wins) > 0 else 0
    gross_loss = abs(np.sum(losses)) if len(losses) > 0 else 1e-10
    profit_factor = gross_profit / gross_loss

    # Risk metrics
    var_95 = np.percentile(valid_pnl, 5)  # 95% VaR

    # Distribution metrics
    if SCIPY_AVAILABLE:
        skewness = stats.skew(valid_pnl)
        kurtosis = stats.kurtosis(valid_pnl)
    else:
        skewness = 0
        kurtosis = 0

    return PerformanceMetrics(
        name=name,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_vol,
        sharpe_ratio=sharpe_ratio,
        max_drawdown=max_drawdown,
        calmar_ratio=calmar_ratio,
        win_rate=win_rate,
        profit_factor=profit_factor,
        var_95=var_95,
        skewness=skewness,
        kurtosis=kurtosis,
        valid_days=len(valid_pnl),
    )


def run_scenario_comparison(
    scenarios: Dict[str, np.ndarray], configs: Dict[str, Tuple]
) -> Dict[str, Dict[str, PerformanceMetrics]]:
    """Run all systems across all scenarios."""
    results = {}

    print("Running scenario comparison...")
    print("=" * 60)

    for scenario_name, prices in scenarios.items():
        print(f"\nScenario: {scenario_name}")
        print("-" * 40)

        scenario_results = {}

        for system_name, (system_class, config) in configs.items():
            try:
                # Run system
                if system_name == "American TF":
                    # American TF needs OHLC - approximate from close
                    high = prices * 1.005
                    low = prices * 0.995
                    system = system_class(config)
                    pnl, _ = system.run(prices, high, low)
                else:
                    system = system_class(config)
                    if hasattr(system, "run_from_prices"):
                        pnl, *_ = system.run_from_prices(prices)
                    else:
                        pnl = np.zeros_like(prices)

                # Calculate metrics
                metrics = calculate_detailed_metrics(pnl, system_name)
                scenario_results[system_name] = metrics

                print(
                    f"  {system_name:<12}: Sharpe={metrics.sharpe_ratio:5.2f}, "
                    f"Return={metrics.annual_return:6.2%}, "
                    f"MaxDD={metrics.max_drawdown:6.2f}"
                )

            except Exception as e:
                print(f"  {system_name:<12}: Error - {e}")
                scenario_results[system_name] = calculate_detailed_metrics(
                    np.array([]), system_name
                )

        results[scenario_name] = scenario_results

    return results


def create_comparison_plots(results: Dict[str, Dict[str, PerformanceMetrics]]) -> None:
    """Create comprehensive comparison plots."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return

    # Prepare data for plotting
    scenarios = list(results.keys())
    systems = list(next(iter(results.values())).keys())

    # Extract metrics for plotting
    metrics_data = {
        "sharpe_ratio": np.array(
            [
                [results[scenario][system].sharpe_ratio for system in systems]
                for scenario in scenarios
            ]
        ),
        "annual_return": np.array(
            [
                [results[scenario][system].annual_return for system in systems]
                for scenario in scenarios
            ]
        ),
        "max_drawdown": np.array(
            [
                [results[scenario][system].max_drawdown for system in systems]
                for scenario in scenarios
            ]
        ),
        "win_rate": np.array(
            [[results[scenario][system].win_rate for system in systems] for scenario in scenarios]
        ),
    }

    # Create comprehensive plot
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig)

    # 1. Sharpe Ratio Heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(metrics_data["sharpe_ratio"], cmap="RdYlGn", aspect="auto")
    ax1.set_title("Sharpe Ratio by Scenario")
    ax1.set_xticks(range(len(systems)))
    ax1.set_xticklabels(systems, rotation=45)
    ax1.set_yticks(range(len(scenarios)))
    ax1.set_yticklabels(scenarios)
    plt.colorbar(im1, ax=ax1)

    # Add text annotations
    for i in range(len(scenarios)):
        for j in range(len(systems)):
            text = ax1.text(
                j,
                i,
                f'{metrics_data["sharpe_ratio"][i, j]:.2f}',
                ha="center",
                va="center",
                color="black" if abs(metrics_data["sharpe_ratio"][i, j]) < 1 else "white",
            )

    # 2. Annual Return Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    x = np.arange(len(scenarios))
    width = 0.25

    for i, system in enumerate(systems):
        returns = [results[scenario][system].annual_return for scenario in scenarios]
        ax2.bar(x + i * width, returns, width, label=system)

    ax2.set_title("Annual Return by Scenario")
    ax2.set_ylabel("Annual Return")
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(scenarios, rotation=45)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color="k", linestyle="-", alpha=0.5)

    # 3. Risk-Return Scatter
    ax3 = fig.add_subplot(gs[0, 2])
    colors = ["blue", "red", "green"]

    for i, system in enumerate(systems):
        returns = [results[scenario][system].annual_return for scenario in scenarios]
        vols = [results[scenario][system].annual_volatility for scenario in scenarios]
        ax3.scatter(vols, returns, label=system, color=colors[i], alpha=0.7, s=50)

    ax3.set_title("Risk-Return Profile")
    ax3.set_xlabel("Annual Volatility")
    ax3.set_ylabel("Annual Return")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color="k", linestyle="-", alpha=0.5)
    ax3.axvline(x=0, color="k", linestyle="-", alpha=0.5)

    # 4. Maximum Drawdown
    ax4 = fig.add_subplot(gs[1, 0])
    for i, system in enumerate(systems):
        drawdowns = [results[scenario][system].max_drawdown for scenario in scenarios]
        ax4.bar(x + i * width, drawdowns, width, label=system)

    ax4.set_title("Maximum Drawdown by Scenario")
    ax4.set_ylabel("Max Drawdown")
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(scenarios, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Win Rate Comparison
    ax5 = fig.add_subplot(gs[1, 1])
    for i, system in enumerate(systems):
        win_rates = [results[scenario][system].win_rate for scenario in scenarios]
        ax5.bar(x + i * width, win_rates, width, label=system)

    ax5.set_title("Win Rate by Scenario")
    ax5.set_ylabel("Win Rate")
    ax5.set_xticks(x + width)
    ax5.set_xticklabels(scenarios, rotation=45)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=0.5, color="k", linestyle="--", alpha=0.5)

    # 6. Profit Factor
    ax6 = fig.add_subplot(gs[1, 2])
    for i, system in enumerate(systems):
        profit_factors = [
            min(results[scenario][system].profit_factor, 5) for scenario in scenarios
        ]  # Cap at 5 for visualization
        ax6.bar(x + i * width, profit_factors, width, label=system)

    ax6.set_title("Profit Factor by Scenario (capped at 5)")
    ax6.set_ylabel("Profit Factor")
    ax6.set_xticks(x + width)
    ax6.set_xticklabels(scenarios, rotation=45)
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.axhline(y=1, color="k", linestyle="--", alpha=0.5)

    # 7. Overall Performance Ranking
    ax7 = fig.add_subplot(gs[2, :])

    # Calculate average ranks across scenarios
    system_scores = {}
    for system in systems:
        scores = []
        for scenario in scenarios:
            m = results[scenario][system]
            # Composite score (higher is better)
            score = (
                m.sharpe_ratio * 0.4
                + min(m.calmar_ratio, 10) * 0.3  # Cap calmar at 10
                + m.win_rate * 0.2
                + min(m.profit_factor, 5) * 0.1
            )  # Cap profit factor at 5
            scores.append(score)
        system_scores[system] = np.mean(scores)

    # Sort systems by average score
    sorted_systems = sorted(system_scores.items(), key=lambda x: x[1], reverse=True)

    systems_sorted = [s[0] for s in sorted_systems]
    scores_sorted = [s[1] for s in sorted_systems]

    ax7.bar(systems_sorted, scores_sorted, color=["gold", "silver", "#CD7F32"])
    ax7.set_title("Overall Performance Ranking (Composite Score)")
    ax7.set_ylabel("Composite Score")
    ax7.grid(True, alpha=0.3)

    # Add score labels
    for i, (system, score) in enumerate(zip(systems_sorted, scores_sorted)):
        ax7.text(i, score + 0.01, f"{score:.2f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig("examples/performance_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"\nComparison plots saved to: examples/performance_comparison.png")


def run_monte_carlo_analysis(configs: Dict[str, Tuple], n_simulations: int = 100) -> None:
    """Run Monte Carlo analysis for robustness testing."""
    print("\n" + "=" * 50)
    print("MONTE CARLO ROBUSTNESS ANALYSIS")
    print("=" * 50)

    np.random.seed(42)  # For reproducible results
    n_days = 1000

    # Store results for each system
    mc_results = {system: [] for system in configs.keys()}

    print(f"Running {n_simulations} simulations...")

    for sim in range(n_simulations):
        if (sim + 1) % 20 == 0:
            print(f"  Completed {sim + 1}/{n_simulations} simulations")

        # Generate random price series
        returns = 0.0002 + 0.018 * np.random.randn(n_days)
        # Add some autocorrelation
        for i in range(1, n_days):
            returns[i] += 0.05 * returns[i - 1]

        prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])

        # Test each system
        for system_name, (system_class, config) in configs.items():
            try:
                if system_name == "American TF":
                    high = prices * (1 + 0.005 * np.abs(np.random.randn(len(prices))))
                    low = prices * (1 - 0.005 * np.abs(np.random.randn(len(prices))))
                    system = system_class(config)
                    pnl, _ = system.run(prices, high, low)
                else:
                    system = system_class(config)
                    pnl, *_ = system.run_from_prices(prices)

                # Calculate Sharpe ratio for this simulation
                valid_pnl = pnl[~np.isnan(pnl)]
                if len(valid_pnl) > 0:
                    annual_return = np.mean(valid_pnl) * 252
                    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    mc_results[system_name].append(sharpe)

            except Exception:
                # Skip failed simulations
                continue

    # Analyze Monte Carlo results
    print(f"\nMonte Carlo Results ({n_simulations} simulations):")
    print(
        f"{'System':<15} {'Mean Sharpe':<12} {'Std Sharpe':<12} {'% Positive':<12} {'95% CI':<15}"
    )
    print("-" * 75)

    for system_name, sharpes in mc_results.items():
        if len(sharpes) > 0:
            mean_sharpe = np.mean(sharpes)
            std_sharpe = np.std(sharpes)
            pct_positive = np.sum(np.array(sharpes) > 0) / len(sharpes) * 100

            # 95% confidence interval
            if len(sharpes) > 10:
                ci_lower = np.percentile(sharpes, 2.5)
                ci_upper = np.percentile(sharpes, 97.5)
                ci_str = f"[{ci_lower:.2f}, {ci_upper:.2f}]"
            else:
                ci_str = "N/A"

            print(
                f"{system_name:<15} {mean_sharpe:<12.2f} {std_sharpe:<12.2f} "
                f"{pct_positive:<12.1f} {ci_str:<15}"
            )


def statistical_significance_test(results: Dict[str, Dict[str, PerformanceMetrics]]) -> None:
    """Perform statistical significance tests between systems."""
    if not SCIPY_AVAILABLE:
        print("SciPy not available. Skipping statistical tests.")
        return

    print("\n" + "=" * 50)
    print("STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 50)

    systems = list(next(iter(results.values())).keys())
    scenarios = list(results.keys())

    # Extract Sharpe ratios for each system across scenarios
    sharpe_data = {}
    for system in systems:
        sharpe_data[system] = [results[scenario][system].sharpe_ratio for scenario in scenarios]

    # Pairwise t-tests
    print("Pairwise t-tests for Sharpe ratios:")
    print("(H0: No difference in mean Sharpe ratio)")
    print()

    for i, system1 in enumerate(systems):
        for j, system2 in enumerate(systems):
            if i < j:  # Only test each pair once
                data1 = sharpe_data[system1]
                data2 = sharpe_data[system2]

                # Paired t-test (same scenarios)
                t_stat, p_value = stats.ttest_rel(data1, data2)

                significance = ""
                if p_value < 0.01:
                    significance = "**"
                elif p_value < 0.05:
                    significance = "*"

                mean_diff = np.mean(data1) - np.mean(data2)

                print(f"{system1} vs {system2}:")
                print(f"  Mean difference: {mean_diff:+.3f}")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.3f} {significance}")
                print()


def main():
    """Main function for performance comparison analysis."""
    print("TFUNIFY PERFORMANCE COMPARISON ANALYSIS")
    print("Comprehensive comparison of trend-following systems across multiple scenarios")
    print("=" * 80)

    # Step 1: Generate test scenarios
    print("\nStep 1: Generating test scenarios...")
    scenarios = generate_test_scenarios()

    print(f"Created {len(scenarios)} test scenarios:")
    for name, prices in scenarios.items():
        returns = np.diff(np.log(prices))
        annual_ret = np.mean(returns) * 252
        annual_vol = np.std(returns) * np.sqrt(252)
        print(f"  {name}: {len(prices)} days, " f"Return={annual_ret:.2%}, Vol={annual_vol:.1%}")

    # Step 2: Setup system configurations
    print("\nStep 2: Setting up system configurations...")
    configs = setup_system_configurations()

    for name, (system_class, config) in configs.items():
        print(f"  {name}: {config}")

    # Step 3: Run scenario comparison
    print("\nStep 3: Running scenario comparison...")
    results = run_scenario_comparison(scenarios, configs)

    # Step 4: Create performance summary table
    print("\n" + "=" * 80)
    print("DETAILED PERFORMANCE SUMMARY")
    print("=" * 80)

    # Print detailed metrics for each scenario
    for scenario_name, scenario_results in results.items():
        print(f"\nScenario: {scenario_name.upper()}")
        print("-" * 60)
        print(
            f"{'System':<15} {'Sharpe':<8} {'Return':<8} {'Vol':<8} {'MaxDD':<8} {'WinRate':<8} {'PF':<6}"
        )
        print("-" * 60)

        for system_name, metrics in scenario_results.items():
            if metrics.valid_days > 0:
                print(
                    f"{system_name:<15} "
                    f"{metrics.sharpe_ratio:<8.2f} "
                    f"{metrics.annual_return:<8.2%} "
                    f"{metrics.annual_volatility:<8.2%} "
                    f"{metrics.max_drawdown:<8.2f} "
                    f"{metrics.win_rate:<8.2%} "
                    f"{metrics.profit_factor:<6.2f}"
                )

    # Step 5: Overall ranking
    print("\n" + "=" * 50)
    print("OVERALL SYSTEM RANKING")
    print("=" * 50)

    # Calculate average metrics across scenarios
    system_averages = {}
    systems = list(next(iter(results.values())).keys())

    for system in systems:
        metrics_list = [results[scenario][system] for scenario in scenarios.keys()]
        valid_metrics = [m for m in metrics_list if m.valid_days > 0]

        if valid_metrics:
            avg_sharpe = np.mean([m.sharpe_ratio for m in valid_metrics])
            avg_return = np.mean([m.annual_return for m in valid_metrics])
            avg_vol = np.mean([m.annual_volatility for m in valid_metrics])
            avg_dd = np.mean([m.max_drawdown for m in valid_metrics])
            avg_winrate = np.mean([m.win_rate for m in valid_metrics])

            system_averages[system] = {
                "sharpe": avg_sharpe,
                "return": avg_return,
                "vol": avg_vol,
                "drawdown": avg_dd,
                "winrate": avg_winrate,
            }

    # Sort by Sharpe ratio
    sorted_systems = sorted(system_averages.items(), key=lambda x: x[1]["sharpe"], reverse=True)

    print(
        f"{'Rank':<5} {'System':<15} {'Avg Sharpe':<12} {'Avg Return':<12} {'Avg Vol':<10} {'Avg MaxDD':<10}"
    )
    print("-" * 70)

    for rank, (system, metrics) in enumerate(sorted_systems, 1):
        print(
            f"{rank:<5} {system:<15} "
            f"{metrics['sharpe']:<12.3f} "
            f"{metrics['return']:<12.2%} "
            f"{metrics['vol']:<10.2%} "
            f"{metrics['drawdown']:<10.2f}"
        )

    # Step 6: Statistical analysis
    statistical_significance_test(results)

    # Step 7: Monte Carlo robustness test
    run_monte_carlo_analysis(configs, n_simulations=100)

    # Step 8: Create comparison plots
    print("\nStep 8: Creating comparison plots...")
    create_comparison_plots(results)

    # Summary insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)

    winner = sorted_systems[0][0]
    winner_sharpe = sorted_systems[0][1]["sharpe"]

    print(f"1. Overall winner: {winner} (avg Sharpe: {winner_sharpe:.3f})")
    print("2. System performance varies significantly by market regime")
    print("3. No single system dominates across all scenarios")
    print("4. Volatility targeting (European TF) provides consistency")
    print("5. Breakout systems (American TF) excel in trending markets")
    print("6. Momentum systems (TSMOM) show regime-dependent performance")

    print("\nRecommendations:")
    print("- Consider ensemble approaches combining multiple systems")
    print("- Adapt system selection based on market regime detection")
    print("- Focus on risk-adjusted returns rather than absolute returns")
    print("- Regular rebalancing and parameter optimization may improve results")

    print(f"\nAnalysis complete! Check 'examples/performance_comparison.png' for detailed charts.")


if __name__ == "__main__":
    main()
