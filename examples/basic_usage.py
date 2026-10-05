"""The three systems on one simulated market.

Run it from any directory::

    python examples/basic_usage.py            # prints a table
    python examples/basic_usage.py --plot     # also saves basic_usage.png next to this file

The market is simulated: a drift that switches between rising and falling
regimes, plus noise. Nothing here says how the systems do on real data; the
point is to show how each one is configured, run and summarised.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tfunify import (
    TSMOM,
    AmericanTF,
    AmericanTFConfig,
    EuropeanTF,
    EuropeanTFConfig,
    TSMOMConfig,
    performance_summary,
)


def simulate_market(n_days: int, seed: int) -> dict[str, np.ndarray]:
    """Daily open-high-low-close prices with trending regimes of about a year."""
    rng = np.random.default_rng(seed)
    regime = np.repeat(rng.choice([-1.0, 1.0], size=n_days // 250 + 1), 250)[:n_days]
    returns = 0.0006 * regime + 0.01 * rng.standard_normal(n_days)
    close = 100.0 * np.cumprod(1.0 + returns)
    spread = 0.006 * rng.random((2, n_days))
    return {"close": close, "high": close * (1.0 + spread[0]), "low": close * (1.0 - spread[1])}


def run_systems(market: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Daily profit and loss of the three systems, each with the paper's parameters."""
    european = EuropeanTF(EuropeanTFConfig(span_long=250, span_short=20))
    american = AmericanTF(AmericanTFConfig(span_long=250, span_short=20, q=5.0, p=5.0))
    tsmom = TSMOM(TSMOMConfig(L=10, M=10))
    return {
        "European LS(250, 20)": european.run_from_prices(market["close"]).pnl,
        "American (250, 20)": american.run(market["close"], market["high"], market["low"]).pnl,
        "TSMOM (L=10, M=10)": tsmom.run_from_prices(market["close"]).pnl,
    }


def main(argv: list[str] | None = None) -> dict[str, np.ndarray]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--days", type=int, default=5200, help="length of the simulation")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--plot", action="store_true", help="save a chart (needs matplotlib)")
    args = parser.parse_args(argv)

    market = simulate_market(args.days, args.seed)
    results = run_systems(market)

    print(f"{'system':24s} {'return':>8s} {'vol':>8s} {'Sharpe':>7s} {'max DD':>8s}")
    for name, pnl in results.items():
        stats = performance_summary(pnl)
        print(
            f"{name:24s} {stats.annual_return:8.1%} {stats.annual_volatility:8.1%} "
            f"{stats.sharpe_ratio:7.2f} {stats.max_drawdown:8.1%}"
        )
    names = list(results)
    correlation = np.corrcoef([results[name] for name in names])
    print("\ncorrelation of daily returns")
    for i, name in enumerate(names):
        print(f"{name:24s} " + " ".join(f"{value:6.2f}" for value in correlation[i]))

    if args.plot:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (top, bottom) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
        top.plot(market["close"], color="black", linewidth=0.8)
        top.set_ylabel("price")
        for name, pnl in results.items():
            bottom.plot(np.cumsum(pnl), label=name, linewidth=1.0)
        bottom.set_ylabel("cumulative return")
        bottom.set_xlabel("day")
        bottom.legend(frameon=False)
        target = Path(__file__).with_suffix(".png")
        figure.savefig(target, dpi=120, bbox_inches="tight")
        plt.close(figure)
        print(f"\nchart saved to {target}")
    return results


if __name__ == "__main__":
    main()
