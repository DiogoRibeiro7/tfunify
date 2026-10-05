"""Choosing the span: a grid search, its out-of-sample check, and the closed form.

The usual way to pick a filter span is to try several on history and keep the
best. This script does that on simulated returns and shows two things next to
it: how the span chosen in sample fares on fresh data, and the Sharpe ratio that
the closed form of the paper assigns to each span from the autocorrelation and
the drift alone. The data are AR(1) returns with drift, so the truth is known.

    python examples/parameter_optimization.py
    python examples/parameter_optimization.py --years 10 --phi 0.03 --sharpe 0.4
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from tfunify import EuropeanTF, EuropeanTFConfig, theory
from tfunify.metrics import sharpe_ratio

A = 260
SPANS = [5, 10, 21, 63, 125, 250]


def simulate_returns(
    rng: np.random.Generator, n_days: int, phi: float, sharpe: float
) -> np.ndarray:
    """AR(1) returns with 1 % daily volatility and the given Sharpe ratio."""
    shocks = (math.sqrt(1.0 - phi * phi) * rng.standard_normal(n_days)).tolist()
    values = [shocks[0]]
    for shock in shocks[1:]:
        values.append(phi * values[-1] + shock)
    return 0.01 * (np.asarray(values) + sharpe / math.sqrt(A))


def realised_sharpe(returns: np.ndarray, span: float) -> float:
    system = EuropeanTF(EuropeanTFConfig(mode="single", span_long=span))
    return sharpe_ratio(system.run_from_returns(returns).pnl[100:], A)


def main(argv: list[str] | None = None) -> dict[str, list[float]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, default=15, help="years in each half of the sample")
    parser.add_argument("--phi", type=float, default=0.05, help="AR(1) coefficient of the returns")
    parser.add_argument("--sharpe", type=float, default=0.5, help="Sharpe ratio of the instrument")
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    n_days = args.years * A
    in_sample = simulate_returns(rng, n_days, args.phi, args.sharpe)
    out_of_sample = simulate_returns(rng, n_days, args.phi, args.sharpe)
    acf = theory.ar1_acf(args.phi, 1000) if args.phi else None

    table: dict[str, list[float]] = {"in sample": [], "out of sample": [], "closed form": []}
    print(
        f"European system, single filter; AR(1) returns with phi = {args.phi}, "
        f"Sharpe ratio {args.sharpe}; {args.years} years in each sample\n"
    )
    print(f"{'span':>6s} {'in sample':>10s} {'out of sample':>14s} {'closed form':>12s}")
    for span in SPANS:
        row = (
            realised_sharpe(in_sample, span),
            realised_sharpe(out_of_sample, span),
            theory.european_sharpe_ratio(span_long=span, acf=acf, instrument_sharpe=args.sharpe),
        )
        for key, value in zip(table, row):
            table[key].append(value)
        print(f"{span:6d} {row[0]:10.2f} {row[1]:14.2f} {row[2]:12.2f}")

    best = int(np.argmax(table["in sample"]))
    # the Sharpe ratio estimated from n days has a standard error of about sqrt(a / n)
    error = math.sqrt(A / n_days)
    excess = (table["in sample"][best] - table["closed form"][best]) / error
    print(
        f"\nBest span in sample: {SPANS[best]} days, Sharpe ratio {table['in sample'][best]:.2f}; "
        f"the same span out of sample: {table['out of sample'][best]:.2f}; "
        f"closed form: {table['closed form'][best]:.2f}."
    )
    print(
        f"A Sharpe ratio measured on {args.years} years has a standard error of about {error:.2f}: "
        f"the in-sample figure of the chosen span is {excess:+.1f} standard errors from the value "
        "the process supports."
    )
    return table


if __name__ == "__main__":
    main()
