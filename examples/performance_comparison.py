"""What the three systems earn in four kinds of market, and what theory says.

Trend following is a bet on positive autocorrelation and on drift. This script
simulates returns with a known autocorrelation and drift, runs the three
systems on many paths, and prints the average Sharpe ratio of each with its
standard error. For the European system the closed form of the paper is shown
next to the simulation.

    python examples/performance_comparison.py
    python examples/performance_comparison.py --paths 200 --years 20
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from tfunify import (
    TSMOM,
    AmericanTF,
    AmericanTFConfig,
    EuropeanTF,
    EuropeanTFConfig,
    TSMOMConfig,
    theory,
)
from tfunify.metrics import sharpe_ratio

A = 260
SPAN = 21  # one month: short enough for the autocorrelation at short lags to matter

# name: (AR(1) coefficient of daily returns, annualised Sharpe ratio of the instrument)
MARKETS = {
    "random walk": (0.0, 0.0),
    "trending, phi = 0.10": (0.10, 0.0),
    "mean reverting, phi = -0.10": (-0.10, 0.0),
    "drift, Sharpe 1.0": (0.0, 1.0),
}


def simulate_returns(
    rng: np.random.Generator, n_days: int, phi: float, sharpe: float
) -> np.ndarray:
    """AR(1) returns with 1 % daily volatility and the given Sharpe ratio."""
    shocks = (math.sqrt(1.0 - phi * phi) * rng.standard_normal(n_days)).tolist()
    values = [shocks[0]]
    for shock in shocks[1:]:
        values.append(phi * values[-1] + shock)
    return 0.01 * (np.asarray(values) + sharpe / math.sqrt(A))


def sharpe_ratios(returns: np.ndarray) -> dict[str, float]:
    close = 100.0 * np.cumprod(1.0 + returns)
    european = EuropeanTF(EuropeanTFConfig(mode="single", span_long=SPAN))
    american = AmericanTF(
        AmericanTFConfig(span_long=SPAN, span_short=5, atr_period=20, q=1.0, p=3.0)
    )
    tsmom = TSMOM(TSMOMConfig(L=1, M=SPAN))
    burn_in = 100
    return {
        "European": sharpe_ratio(european.run_from_returns(returns).pnl[burn_in:], A),
        "American": sharpe_ratio(american.run(close).pnl[burn_in:], A),
        "TSMOM": sharpe_ratio(tsmom.run_from_returns(returns).pnl[burn_in:], A),
    }


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--paths", type=int, default=60, help="simulated paths per market")
    parser.add_argument("--years", type=int, default=20, help="length of each path")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n_days = args.years * A
    print(f"Sharpe ratios, mean of {args.paths} paths of {args.years} years (standard error)\n")
    print(f"{'market':30s} {'European':>14s} {'closed form':>12s} {'American':>14s} {'TSMOM':>14s}")
    table: dict[str, dict[str, float]] = {}
    for market, (phi, sharpe) in MARKETS.items():
        samples: dict[str, list[float]] = {"European": [], "American": [], "TSMOM": []}
        for _ in range(args.paths):
            for system, value in sharpe_ratios(simulate_returns(rng, n_days, phi, sharpe)).items():
                samples[system].append(value)
        closed_form = theory.european_sharpe_ratio(
            span_long=SPAN, acf=theory.ar1_acf(phi, 500) if phi else None, instrument_sharpe=sharpe
        )
        row = {"closed form": closed_form}
        cells = {}
        for system, values in samples.items():
            mean = float(np.mean(values))
            error = (
                float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            )
            row[system] = mean
            cells[system] = f"{mean:6.2f} ({error:.2f})"
        table[market] = row
        print(
            f"{market:30s} {cells['European']:>14s} {closed_form:12.2f} "
            f"{cells['American']:>14s} {cells['TSMOM']:>14s}"
        )
    return table


if __name__ == "__main__":
    main()
