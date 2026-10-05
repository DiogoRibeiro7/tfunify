"""A trend-following sleeve next to a 60/40 portfolio.

Three simulated assets (an equity index with crashes, a bond, a commodity) are
traded by the European system, one position per asset with an equal volatility
target. The script compares a 60/40 portfolio with the same portfolio plus a
trend-following sleeve (an allocation held on top of it, as a futures overlay
is), and reports how the sleeve did in the worst quarters of the 60/40
portfolio.

The assets are simulated with regimes that last about a year; the numbers
illustrate the mechanics, not an expected performance.

    python examples/portfolio_integration.py
"""

from __future__ import annotations

import argparse

import numpy as np

from tfunify import EuropeanTF, EuropeanTFConfig, performance_summary

A = 260

# name: (annual volatility, annual drift in a good regime, annual drift in a bad regime)
ASSETS = {
    "equity": (0.16, 0.12, -0.25),
    "bond": (0.06, 0.03, -0.02),
    "commodity": (0.20, 0.10, -0.10),
}


def simulate_assets(n_days: int, seed: int) -> dict[str, np.ndarray]:
    """Daily returns; each asset is in its bad regime about a quarter of the time."""
    rng = np.random.default_rng(seed)
    returns = {}
    for name, (volatility, good, bad) in ASSETS.items():
        regimes = rng.random(n_days // 130 + 1) < 0.25  # redrawn every half year
        in_bad_regime = np.repeat(regimes, 130)[:n_days]
        drift = np.where(in_bad_regime, bad, good) / A
        returns[name] = drift + volatility / np.sqrt(A) * rng.standard_normal(n_days)
    return returns


def main(argv: list[str] | None = None) -> dict[str, np.ndarray]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, default=30)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--sleeve", type=float, default=0.5, help="exposure to the trend sleeve")
    args = parser.parse_args(argv)

    returns = simulate_assets(args.years * A, args.seed)
    benchmark = 0.6 * returns["equity"] + 0.4 * returns["bond"]

    # One European system per asset. Each targets 10 % volatility; three of them
    # with little correlation give the sleeve a volatility of about 6 %.
    system = EuropeanTF(EuropeanTFConfig(sigma_target_annual=0.10, span_long=125, span_short=20))
    sleeve = np.mean([system.run_from_returns(r).pnl for r in returns.values()], axis=0)
    combined = benchmark + args.sleeve * sleeve

    portfolios = {
        "60/40": benchmark,
        "trend sleeve": sleeve,
        f"60/40 + {args.sleeve:g} x sleeve": combined,
    }
    print(f"{'portfolio':24s} {'return':>8s} {'vol':>8s} {'Sharpe':>7s} {'max DD':>8s}")
    for name, pnl in portfolios.items():
        stats = performance_summary(pnl)
        print(
            f"{name:24s} {stats.annual_return:8.1%} {stats.annual_volatility:8.1%} "
            f"{stats.sharpe_ratio:7.2f} {stats.max_drawdown:8.1%}"
        )

    quarters = len(benchmark) // 65
    by_quarter = benchmark[: quarters * 65].reshape(quarters, 65).sum(axis=1)
    sleeve_by_quarter = sleeve[: quarters * 65].reshape(quarters, 65).sum(axis=1)
    worst = by_quarter <= np.quantile(by_quarter, 0.16)
    print(
        f"\nCorrelation of daily returns, sleeve and 60/40: "
        f"{np.corrcoef(sleeve, benchmark)[0, 1]:.2f}"
    )
    print(
        f"In the worst 16 % of quarters of the 60/40 portfolio ({int(worst.sum())} quarters) it "
        f"returned {by_quarter[worst].mean():.1%} a quarter on average and the sleeve "
        f"{sleeve_by_quarter[worst].mean():.1%}; in the other quarters the sleeve returned "
        f"{sleeve_by_quarter[~worst].mean():.1%}."
    )
    return portfolios


if __name__ == "__main__":
    main()
