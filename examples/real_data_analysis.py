"""The three systems on a real price history.

With the Yahoo extra installed (``pip install "tfunify[yahoo]"``) the script
downloads a symbol; with ``--csv`` it reads a file you already have, in the
format described in the README (a ``close`` column, and ``high`` and ``low`` if
available)::

    python examples/real_data_analysis.py --ticker SPY --period 10y
    python examples/real_data_analysis.py --csv my_prices.csv

It prints, for each system, the annual return and volatility, the Sharpe ratio,
the maximum drawdown, the share of days with a position and the annual
volatility-weighted turnover. Results are gross of trading costs, and the
position is in one instrument: a single market says little about a method that
is normally run on dozens.

The figures are returns of the instrument times the weight, as for a futures
position, whose return is already in excess of the cost of financing. The price
of a share or of a fund such as SPY is not: holding it has to be financed, so a
long position earns less than shown here, and a short one more, by about the
short-term interest rate times the weight. Downloaded prices are adjusted for
dividends, so that the returns are total returns.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import numpy as np

from tfunify import (
    TSMOM,
    AmericanTF,
    AmericanTFConfig,
    EuropeanTF,
    EuropeanTFConfig,
    TSMOMConfig,
    ewma_volatility_from_returns,
    pct_returns_from_prices,
    performance_summary,
    span_to_nu,
    volatility_weighted_turnover,
)
from tfunify.data import download_csv, load_csv

A = 260


def analyse(data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    """Run the three systems with the paper's parameters and summarise them."""
    close, high, low = data["close"], data["high"], data["low"]
    european = EuropeanTF(EuropeanTFConfig()).run_from_prices(close)
    american = AmericanTF(AmericanTFConfig()).run(close, high, low)
    tsmom = TSMOM(TSMOMConfig()).run_from_prices(close)

    # the American system has no volatility estimate of its own: use the same
    # EWMA volatility as the other two to put its turnover on the same scale
    sigma = np.r_[
        np.nan, ewma_volatility_from_returns(pct_returns_from_prices(close)[1:], span_to_nu(33))
    ]
    runs = {
        "European": (european.pnl, european.weights, european.volatility),
        "American": (american.pnl, american.weights, sigma),
        "TSMOM": (tsmom.pnl, tsmom.weights, tsmom.volatility),
    }
    table = {}
    for name, (pnl, weights, volatility) in runs.items():
        stats = performance_summary(pnl, A)
        known = np.isfinite(volatility)
        turnover = volatility_weighted_turnover(weights[known], volatility[known], A)
        table[name] = {
            "return": stats.annual_return,
            "volatility": stats.annual_volatility,
            "sharpe": stats.sharpe_ratio,
            "drawdown": stats.max_drawdown,
            "in market": float(np.mean(weights != 0.0)),
            "turnover": float(A * turnover.mean()),
        }
    return table


def main(argv: list[str] | None = None) -> dict[str, dict[str, float]]:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--csv", help="read prices from this file instead of downloading")
    parser.add_argument("--ticker", default="SPY", help="Yahoo symbol to download")
    parser.add_argument("--period", default="10y", help="history to download")
    args = parser.parse_args(argv)

    try:
        if args.csv:
            source = args.csv
            data = load_csv(source)
        else:
            source = f"{args.ticker} from Yahoo Finance ({args.period})"
            with tempfile.TemporaryDirectory() as folder:
                path = download_csv(
                    args.ticker, Path(folder) / "prices.csv", period=args.period, auto_adjust=True
                )
                data = load_csv(path)
    except (ImportError, OSError, ValueError) as error:
        raise SystemExit(f"error: {error}") from None

    close = data["close"]
    returns = pct_returns_from_prices(close)[1:]
    span = f"{data['date'][0]} to {data['date'][-1]}, " if "date" in data else ""
    print(f"{source}: {span}{close.size} days")
    autocorrelation = np.corrcoef(returns[1:], returns[:-1])[0, 1]
    print(
        f"buy and hold: {performance_summary(returns, A)}\n"
        f"lag-1 autocorrelation of daily returns: {autocorrelation:+.3f}\n"
    )
    print("Returns of the instrument, not in excess of financing: see the note in this script.\n")
    table = analyse(data)
    print(
        f"{'system':10s} {'return':>8s} {'vol':>8s} {'Sharpe':>7s} {'max DD':>8s} "
        f"{'in market':>10s} {'turnover':>9s}"
    )
    for name, row in table.items():
        print(
            f"{name:10s} {row['return']:8.1%} {row['volatility']:8.1%} {row['sharpe']:7.2f} "
            f"{row['drawdown']:8.1%} {row['in market']:10.0%} {row['turnover']:9.2f}"
        )
    return table


if __name__ == "__main__":
    main()
