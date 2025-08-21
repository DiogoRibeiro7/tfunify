#!/usr/bin/env python3
"""
Parameter Optimization Example for tfunify

This example demonstrates how to optimize parameters for trend-following systems
using various optimization techniques including grid search, random search,
and Bayesian optimization.

Key features:
- Grid search for systematic parameter exploration
- Random search for efficiency
- Walk-forward analysis for robustness
- Out-of-sample validation
- Parameter sensitivity analysis
"""

import sys
import numpy as np
import time
from itertools import product
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

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
except ImportError:
    TFUNIFY_AVAILABLE = False
    print(f"Error: tfunify package required for optimization examples.")
    print(f"Install with: pip install tfunify")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from scipy.optimize import differential_evolution

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


@dataclass
class OptimizationResult:
    """Container for optimization results."""

    best_params: Dict[str, Any]
    best_score: float
    all_results: List[Tuple[Dict[str, Any], float]]
    optimization_time: float
    system_name: str


def generate_optimization_data(n_days: int = 2000, seed: int = 42) -> np.ndarray:
    """Generate realistic price data for optimization."""
    np.random.seed(seed)

    # Create trending data with varying volatility regimes
    base_drift = 0.0003
    base_vol = 0.015

    returns = np.zeros(n_days)
    vol_regime = np.ones(n_days) * base_vol

    # Add regime changes
    regime_changes = [500, 1000, 1500]
    vol_multipliers = [1.0, 1.5, 0.8, 1.2]

    for i, change_point in enumerate(regime_changes):
        if change_point < n_days:
            vol_regime[change_point:] *= vol_multipliers[i + 1]

    # Generate returns with momentum
    momentum = 0.0
    for i in range(n_days):
        # Add momentum persistence
        momentum = 0.95 * momentum + 0.05 * np.random.randn()
        returns[i] = base_drift + vol_regime[i] * np.random.randn() + 0.002 * momentum

    # Convert to prices
    prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])
    return prices


def objective_function(pnl: np.ndarray) -> float:
    """
    Objective function for optimization.

    Uses risk-adjusted return with penalty for excessive drawdown.
    """
    valid_pnl = pnl[~np.isnan(pnl)]

    if len(valid_pnl) < 50:  # Need minimum observations
        return -999.0

    # Calculate metrics
    annual_return = np.mean(valid_pnl) * 252
    annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)

    if annual_vol == 0:
        return -999.0

    sharpe_ratio = annual_return / annual_vol

    # Calculate maximum drawdown
    cumulative = np.cumsum(valid_pnl)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_drawdown = np.min(drawdown)

    # Penalty for excessive drawdown
    dd_penalty = 0
    if abs(max_drawdown) > annual_return * 2:  # DD > 2x annual return
        dd_penalty = abs(max_drawdown) / annual_return

    # Objective: Sharpe ratio minus drawdown penalty
    objective = sharpe_ratio - dd_penalty

    return objective


def optimize_european_tf_grid(
    prices: np.ndarray, train_start: int = 0, train_end: int = None
) -> OptimizationResult:
    """Optimize European TF parameters using grid search."""
    print("Optimizing European TF (Grid Search)...")

    if train_end is None:
        train_end = len(prices)

    train_prices = prices[train_start:train_end]

    # Define parameter grid
    param_grid = {
        "sigma_target_annual": [0.08, 0.10, 0.12, 0.15, 0.18],
        "span_sigma": [22, 33, 44, 66],
        "span_long": [150, 200, 250, 300],
        "span_short": [20, 30, 40, 50],
        "mode": ["single", "longshort"],
    }

    start_time = time.time()
    results = []
    best_score = -999.0
    best_params = None

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = np.prod([len(values) for values in param_values])
    print(f"  Testing {total_combinations} parameter combinations...")

    completed = 0
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))

        # Skip invalid combinations
        if params["mode"] == "longshort" and params["span_short"] >= params["span_long"]:
            continue

        try:
            config = EuropeanTFConfig(**params, a=252)
            system = EuropeanTF(config)
            pnl, _, _, _ = system.run_from_prices(train_prices)

            score = objective_function(pnl)
            results.append((params.copy(), score))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        except Exception:
            # Skip invalid parameter combinations
            continue

        completed += 1
        if completed % 50 == 0:
            print(f"    Completed {completed}/{total_combinations} combinations")

    optimization_time = time.time() - start_time

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results,
        optimization_time=optimization_time,
        system_name="European TF",
    )


def optimize_american_tf_random(
    prices: np.ndarray, train_start: int = 0, train_end: int = None, n_trials: int = 200
) -> OptimizationResult:
    """Optimize American TF parameters using random search."""
    print("Optimizing American TF (Random Search)...")

    if train_end is None:
        train_end = len(prices)

    train_prices = prices[train_start:train_end]

    # Define parameter ranges
    param_ranges = {
        "span_long": (100, 300),
        "span_short": (10, 60),
        "atr_period": (15, 50),
        "q": (1.5, 4.0),
        "p": (2.0, 6.0),
        "r_multiple": (0.005, 0.03),
    }

    start_time = time.time()
    results = []
    best_score = -999.0
    best_params = None

    print(f"  Testing {n_trials} random parameter combinations...")

    for trial in range(n_trials):
        # Generate random parameters
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name in ["span_long", "span_short", "atr_period"]:
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                params[param_name] = np.random.uniform(min_val, max_val)

        # Ensure span_short < span_long
        if params["span_short"] >= params["span_long"]:
            params["span_short"] = max(10, params["span_long"] - 20)

        try:
            config = AmericanTFConfig(**params)
            system = AmericanTF(config)

            # Generate synthetic OHLC
            high = train_prices * (1 + 0.01 * np.abs(np.random.randn(len(train_prices))))
            low = train_prices * (1 - 0.01 * np.abs(np.random.randn(len(train_prices))))

            pnl, _ = system.run(train_prices, high, low)

            score = objective_function(pnl)
            results.append((params.copy(), score))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        except Exception:
            # Skip invalid parameter combinations
            continue

        if (trial + 1) % 50 == 0:
            print(f"    Completed {trial + 1}/{n_trials} trials")

    optimization_time = time.time() - start_time

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results,
        optimization_time=optimization_time,
        system_name="American TF",
    )


def optimize_tsmom_bayesian(
    prices: np.ndarray, train_start: int = 0, train_end: int = None, n_calls: int = 100
) -> OptimizationResult:
    """Optimize TSMOM parameters using differential evolution (pseudo-Bayesian)."""
    print("Optimizing TSMOM (Differential Evolution)...")

    if not SCIPY_AVAILABLE:
        print("  SciPy not available. Using random search instead.")
        return optimize_tsmom_random(prices, train_start, train_end, n_calls)

    if train_end is None:
        train_end = len(prices)

    train_prices = prices[train_start:train_end]

    # Define parameter bounds
    bounds = [
        (0.06, 0.20),  # sigma_target_annual
        (20, 60),  # span_sigma
        (5, 25),  # L
        (4, 15),  # M
    ]

    param_names = ["sigma_target_annual", "span_sigma", "L", "M"]

    def objective_wrapper(x):
        """Wrapper function for scipy optimization."""
        params = dict(zip(param_names, x))
        params["span_sigma"] = int(params["span_sigma"])
        params["L"] = int(params["L"])
        params["M"] = int(params["M"])
        params["a"] = 252

        try:
            config = TSMOMConfig(**params)
            system = TSMOM(config)
            pnl, _, _, _ = system.run_from_prices(train_prices)

            score = objective_function(pnl)
            return -score  # Minimize negative score

        except Exception:
            return 999.0  # Large penalty for invalid parameters

    start_time = time.time()

    print(f"  Running differential evolution with {n_calls} function evaluations...")

    # Run optimization
    result = differential_evolution(
        objective_wrapper, bounds, maxiter=n_calls // 10, popsize=10, seed=42, disp=False
    )

    # Extract results
    best_params_values = result.x
    best_params = dict(zip(param_names, best_params_values))
    best_params["span_sigma"] = int(best_params["span_sigma"])
    best_params["L"] = int(best_params["L"])
    best_params["M"] = int(best_params["M"])
    best_params["a"] = 252

    best_score = -result.fun

    optimization_time = time.time() - start_time

    # Create dummy all_results (differential evolution doesn't track all trials)
    all_results = [(best_params.copy(), best_score)]

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=all_results,
        optimization_time=optimization_time,
        system_name="TSMOM",
    )


def optimize_tsmom_random(
    prices: np.ndarray, train_start: int = 0, train_end: int = None, n_trials: int = 100
) -> OptimizationResult:
    """Fallback random search for TSMOM when SciPy is not available."""
    if train_end is None:
        train_end = len(prices)

    train_prices = prices[train_start:train_end]

    param_ranges = {
        "sigma_target_annual": (0.06, 0.20),
        "span_sigma": (20, 60),
        "L": (5, 25),
        "M": (4, 15),
    }

    start_time = time.time()
    results = []
    best_score = -999.0
    best_params = None

    for trial in range(n_trials):
        params = {}
        for param_name, (min_val, max_val) in param_ranges.items():
            if param_name in ["span_sigma", "L", "M"]:
                params[param_name] = np.random.randint(min_val, max_val + 1)
            else:
                params[param_name] = np.random.uniform(min_val, max_val)

        params["a"] = 252

        try:
            config = TSMOMConfig(**params)
            system = TSMOM(config)
            pnl, _, _, _ = system.run_from_prices(train_prices)

            score = objective_function(pnl)
            results.append((params.copy(), score))

            if score > best_score:
                best_score = score
                best_params = params.copy()

        except Exception:
            continue

    optimization_time = time.time() - start_time

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        all_results=results,
        optimization_time=optimization_time,
        system_name="TSMOM",
    )


def walk_forward_analysis(
    prices: np.ndarray,
    system_configs: Dict[str, Any],
    train_window: int = 1000,
    test_window: int = 250,
    reoptimize_freq: int = 250,
) -> Dict[str, Dict]:
    """
    Perform walk-forward analysis with periodic reoptimization.
    """
    print("\n" + "=" * 50)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 50)

    results = {
        "European TF": {"pnl": [], "params_history": []},
        "American TF": {"pnl": [], "params_history": []},
        "TSMOM": {"pnl": [], "params_history": []},
    }

    n_periods = (len(prices) - train_window) // test_window
    print(f"Running {n_periods} walk-forward periods...")
    print(f"Train window: {train_window} days, Test window: {test_window} days")
    print(f"Reoptimization frequency: {reoptimize_freq} days")

    current_params = system_configs.copy()

    for period in range(n_periods):
        train_start = period * test_window
        train_end = train_start + train_window
        test_start = train_end
        test_end = test_start + test_window

        if test_end > len(prices):
            break

        print(
            f"\nPeriod {period + 1}/{n_periods}: "
            f"Train[{train_start}:{train_end}], Test[{test_start}:{test_end}]"
        )

        # Reoptimize parameters if needed
        if period % (reoptimize_freq // test_window) == 0:
            print("  Reoptimizing parameters...")

            # European TF optimization (simplified)
            eu_opt = optimize_european_tf_grid(prices, train_start, train_end)
            if eu_opt.best_params:
                current_params["European TF"] = eu_opt.best_params

            # American TF optimization (simplified)
            am_opt = optimize_american_tf_random(prices, train_start, train_end, n_trials=50)
            if am_opt.best_params:
                current_params["American TF"] = am_opt.best_params

            # TSMOM optimization (simplified)
            ts_opt = optimize_tsmom_random(prices, train_start, train_end, n_trials=50)
            if ts_opt.best_params:
                current_params["TSMOM"] = ts_opt.best_params

        # Test optimized parameters on out-of-sample data
        test_prices = prices[test_start:test_end]

        # European TF
        try:
            eu_config = EuropeanTFConfig(**current_params["European TF"])
            eu_system = EuropeanTF(eu_config)
            eu_pnl, _, _, _ = eu_system.run_from_prices(test_prices)
            results["European TF"]["pnl"].extend(eu_pnl[~np.isnan(eu_pnl)])
            results["European TF"]["params_history"].append(current_params["European TF"].copy())
        except Exception as e:
            print(f"    European TF error: {e}")

        # American TF
        try:
            am_config = AmericanTFConfig(**current_params["American TF"])
            am_system = AmericanTF(am_config)
            # Generate synthetic OHLC for testing
            high = test_prices * 1.005
            low = test_prices * 0.995
            am_pnl, _ = am_system.run(test_prices, high, low)
            results["American TF"]["pnl"].extend(am_pnl[~np.isnan(am_pnl)])
            results["American TF"]["params_history"].append(current_params["American TF"].copy())
        except Exception as e:
            print(f"    American TF error: {e}")

        # TSMOM
        try:
            ts_config = TSMOMConfig(**current_params["TSMOM"])
            ts_system = TSMOM(ts_config)
            ts_pnl, _, _, _ = ts_system.run_from_prices(test_prices)
            results["TSMOM"]["pnl"].extend(ts_pnl[~np.isnan(ts_pnl)])
            results["TSMOM"]["params_history"].append(current_params["TSMOM"].copy())
        except Exception as e:
            print(f"    TSMOM error: {e}")

    return results


def create_optimization_plots(
    optimization_results: List[OptimizationResult], walkforward_results: Dict[str, Dict]
) -> None:
    """Create plots showing optimization results."""
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available for plotting.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Parameter Optimization Results", fontsize=16)

    # 1. Optimization scores comparison
    ax1 = axes[0, 0]
    systems = [result.system_name for result in optimization_results]
    scores = [result.best_score for result in optimization_results]
    times = [result.optimization_time for result in optimization_results]

    colors = ["blue", "red", "green"]
    ax1.bar(systems, scores, color=colors[: len(systems)])
    ax1.set_title("Best Optimization Scores")
    ax1.set_ylabel("Objective Score")
    ax1.grid(True, alpha=0.3)

    # Add optimization time as text
    for i, (system, score, opt_time) in enumerate(zip(systems, scores, times)):
        ax1.text(i, score + 0.01, f"{opt_time:.1f}s", ha="center", va="bottom")

    # 2. Walk-forward cumulative performance
    ax2 = axes[0, 1]
    for system_name, system_data in walkforward_results.items():
        if system_data["pnl"]:
            cumulative_pnl = np.cumsum(system_data["pnl"])
            ax2.plot(cumulative_pnl, label=system_name, linewidth=2)

    ax2.set_title("Walk-Forward Cumulative P&L")
    ax2.set_ylabel("Cumulative P&L")
    ax2.set_xlabel("Trading Days")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Parameter stability (example for European TF span_long)
    ax3 = axes[1, 0]
    eu_params = walkforward_results.get("European TF", {}).get("params_history", [])
    if eu_params:
        span_long_history = [p.get("span_long", 0) for p in eu_params]
        ax3.plot(span_long_history, "o-", linewidth=2, markersize=4)
        ax3.set_title("European TF: span_long Parameter Evolution")
        ax3.set_ylabel("span_long")
        ax3.set_xlabel("Reoptimization Period")
        ax3.grid(True, alpha=0.3)

    # 4. Walk-forward performance metrics
    ax4 = axes[1, 1]
    wf_sharpes = []
    wf_systems = []

    for system_name, system_data in walkforward_results.items():
        if system_data["pnl"]:
            pnl_array = np.array(system_data["pnl"])
            if len(pnl_array) > 0:
                annual_ret = np.mean(pnl_array) * 252
                annual_vol = np.std(pnl_array, ddof=0) * np.sqrt(252)
                sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
                wf_sharpes.append(sharpe)
                wf_systems.append(system_name)

    if wf_sharpes:
        ax4.bar(wf_systems, wf_sharpes, color=colors[: len(wf_systems)])
        ax4.set_title("Walk-Forward Sharpe Ratios")
        ax4.set_ylabel("Sharpe Ratio")
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="k", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("examples/parameter_optimization.png", dpi=150, bbox_inches="tight")
    plt.show()

    print(f"Optimization plots saved to: examples/parameter_optimization.png")


def print_optimization_summary(optimization_results: List[OptimizationResult]) -> None:
    """Print summary of optimization results."""
    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)

    print(f"{'System':<15} {'Method':<20} {'Best Score':<12} {'Time (s)':<10} {'Best Parameters'}")
    print("-" * 100)

    for result in optimization_results:
        method = (
            "Grid Search"
            if "European" in result.system_name
            else "Random Search"
            if "American" in result.system_name
            else "Differential Evolution"
        )

        # Format best parameters for display
        if result.best_params:
            param_str = (
                ", ".join([f"{k}={v}" for k, v in result.best_params.items() if k not in ["a"]])[
                    :50
                ]
                + "..."
            )
        else:
            param_str = "None"

        print(
            f"{result.system_name:<15} {method:<20} "
            f"{result.best_score:<12.3f} {result.optimization_time:<10.1f} {param_str}"
        )

    print("\nKey Findings:")

    # Find best overall system
    best_result = max(optimization_results, key=lambda x: x.best_score)
    print(
        f"- Best performing system: {best_result.system_name} (score: {best_result.best_score:.3f})"
    )

    # Compare optimization efficiency
    fastest = min(optimization_results, key=lambda x: x.optimization_time)
    print(f"- Fastest optimization: {fastest.system_name} ({fastest.optimization_time:.1f}s)")

    # Parameter insights
    for result in optimization_results:
        if result.best_params and result.system_name == "European TF":
            mode = result.best_params.get("mode", "unknown")
            span_ratio = result.best_params.get("span_long", 1) / result.best_params.get(
                "span_short", 1
            )
            print(f"- European TF optimal: {mode} mode, span ratio {span_ratio:.1f}")


def main():
    """Main function for parameter optimization example."""
    print("TFUNIFY PARAMETER OPTIMIZATION EXAMPLE")
    print("Comprehensive parameter optimization across multiple methods")
    print("=" * 70)

    # Step 1: Generate optimization data
    print("\nStep 1: Generating optimization dataset...")
    prices = generate_optimization_data(n_days=2500, seed=42)

    # Analyze the generated data
    returns = np.diff(np.log(prices))
    annual_ret = np.mean(returns) * 252
    annual_vol = np.std(returns) * np.sqrt(252)
    trend_strength = np.corrcoef(np.arange(len(prices)), prices)[0, 1]

    print(f"Generated {len(prices)} price observations")
    print(f"Annual return: {annual_ret:.2%}")
    print(f"Annual volatility: {annual_vol:.1%}")
    print(f"Trend strength (correlation): {trend_strength:.3f}")

    # Step 2: Run individual optimizations
    print("\nStep 2: Running system optimizations...")

    # Split data: 70% for optimization, 30% for validation
    split_point = int(len(prices) * 0.7)

    optimization_results = []

    # European TF - Grid Search
    print("\n2a. European TF Optimization")
    eu_result = optimize_european_tf_grid(prices, 0, split_point)
    optimization_results.append(eu_result)

    # American TF - Random Search
    print("\n2b. American TF Optimization")
    am_result = optimize_american_tf_random(prices, 0, split_point, n_trials=150)
    optimization_results.append(am_result)

    # TSMOM - Bayesian/Differential Evolution
    print("\n2c. TSMOM Optimization")
    ts_result = optimize_tsmom_bayesian(prices, 0, split_point, n_calls=80)
    optimization_results.append(ts_result)

    # Step 3: Print optimization summary
    print_optimization_summary(optimization_results)

    # Step 4: Out-of-sample validation
    print("\n" + "=" * 50)
    print("OUT-OF-SAMPLE VALIDATION")
    print("=" * 50)

    test_prices = prices[split_point:]
    print(f"Testing on {len(test_prices)} out-of-sample observations...")

    for result in optimization_results:
        if not result.best_params:
            continue

        try:
            if result.system_name == "European TF":
                config = EuropeanTFConfig(**result.best_params)
                system = EuropeanTF(config)
                pnl, _, _, _ = system.run_from_prices(test_prices)
            elif result.system_name == "American TF":
                config = AmericanTFConfig(**result.best_params)
                system = AmericanTF(config)
                high = test_prices * 1.005
                low = test_prices * 0.995
                pnl, _ = system.run(test_prices, high, low)
            elif result.system_name == "TSMOM":
                config = TSMOMConfig(**result.best_params)
                system = TSMOM(config)
                pnl, _, _, _ = system.run_from_prices(test_prices)

            valid_pnl = pnl[~np.isnan(pnl)]
            if len(valid_pnl) > 0:
                oos_annual_ret = np.mean(valid_pnl) * 252
                oos_annual_vol = np.std(valid_pnl, ddof=0) * np.sqrt(252)
                oos_sharpe = oos_annual_ret / oos_annual_vol if oos_annual_vol > 0 else 0

                print(
                    f"{result.system_name:<15}: "
                    f"Return={oos_annual_ret:6.2%}, "
                    f"Vol={oos_annual_vol:6.2%}, "
                    f"Sharpe={oos_sharpe:5.2f}"
                )

        except Exception as e:
            print(f"{result.system_name:<15}: Error - {e}")

    # Step 5: Walk-forward analysis
    print("\nStep 5: Walk-forward analysis...")

    # Setup default configurations for walk-forward
    default_configs = {
        "European TF": optimization_results[0].best_params
        or {
            "sigma_target_annual": 0.12,
            "a": 252,
            "span_sigma": 44,
            "mode": "longshort",
            "span_long": 200,
            "span_short": 40,
        },
        "American TF": optimization_results[1].best_params
        or {
            "span_long": 150,
            "span_short": 30,
            "atr_period": 30,
            "q": 2.5,
            "p": 4.0,
            "r_multiple": 0.015,
        },
        "TSMOM": optimization_results[2].best_params
        or {"sigma_target_annual": 0.10, "a": 252, "span_sigma": 44, "L": 15, "M": 8},
    }

    wf_results = walk_forward_analysis(
        prices, default_configs, train_window=800, test_window=200, reoptimize_freq=400
    )

    # Step 6: Create visualization
    print("\nStep 6: Creating optimization plots...")
    create_optimization_plots(optimization_results, wf_results)

    # Summary insights
    print("\n" + "=" * 50)
    print("KEY INSIGHTS")
    print("=" * 50)

    best_system = max(optimization_results, key=lambda x: x.best_score)

    print(f"1. Best optimized system: {best_system.system_name}")
    print("2. Grid search provides systematic exploration but is slow")
    print("3. Random search offers good efficiency for complex parameter spaces")
    print("4. Walk-forward analysis reveals parameter stability over time")
    print("5. Out-of-sample performance often differs from in-sample optimization")

    print("\nRecommendations:")
    print("- Use ensemble methods combining multiple parameter sets")
    print("- Regular reoptimization helps adapt to changing market conditions")
    print("- Consider robust optimization targeting multiple objectives")
    print("- Always validate on out-of-sample data before deployment")
    print("- Monitor parameter stability and performance decay over time")

    print(f"\nOptimization analysis complete!")
    print(f"Check 'examples/parameter_optimization.png' for detailed results.")


if __name__ == "__main__":
    main()
