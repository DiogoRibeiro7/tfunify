"""Command line interface: `tfu european|american|tsmom|download`."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from . import __version__
from .american import AmericanTF, AmericanTFConfig
from .data import download_csv, load_csv
from .european import EuropeanTF, EuropeanTFConfig
from .metrics import performance_summary
from .tsmom import TSMOM, TSMOMConfig

__all__ = ["build_parser", "main"]

_EPILOG = """\
examples:
  tfu download SPY --out spy.csv --period 5y
  tfu european --csv spy.csv --target 0.15 --span-long 250 --span-short 20
  tfu american --csv spy.csv --q 5 --p 5
  tfu tsmom --csv spy.csv --L 10 --M 10

Each system prints a summary and writes its daily results to an .npz file.
"""


def _report(name: str, pnl: NDArray[np.float64], a: float, out: str, **arrays: Any) -> None:
    # through an open file: given a name, NumPy would append ".npz" to it and
    # the results would not be where the message below says
    with open(out, "wb") as handle:
        np.savez(handle, pnl=pnl, **arrays)
    print(f"{name}: {performance_summary(pnl, a)}")
    print(f"results written to {out}")


def _cmd_european(args: argparse.Namespace) -> None:
    data = load_csv(args.csv)
    cfg = EuropeanTFConfig(
        sigma_target_annual=args.target,
        a=args.a,
        span_sigma=args.span_sigma,
        mode=args.mode,
        span_long=args.span_long,
        span_short=args.span_short,
    )
    result = EuropeanTF(cfg).run_from_prices(data["close"])
    _report(
        "European TF",
        result.pnl,
        cfg.a,
        args.out,
        weights=result.weights,
        signal=result.signal,
        volatility=result.volatility,
    )


def _cmd_american(args: argparse.Namespace) -> None:
    data = load_csv(args.csv)
    cfg = AmericanTFConfig(
        span_long=args.span_long,
        span_short=args.span_short,
        atr_period=args.atr_period,
        q=args.q,
        p=args.p,
        r_multiple=args.r_multiple,
    )
    result = AmericanTF(cfg).run(data["close"], data["high"], data["low"])
    _report(
        "American TF",
        result.pnl,
        args.a,
        args.out,
        weights=result.weights,
        position=result.position,
        stop=result.stop,
        atr=result.atr,
        fast=result.fast,
        slow=result.slow,
    )


def _cmd_tsmom(args: argparse.Namespace) -> None:
    data = load_csv(args.csv)
    cfg = TSMOMConfig(
        sigma_target_annual=args.target,
        a=args.a,
        span_sigma=args.span_sigma,
        L=args.L,
        M=args.M,
        signs=args.signs,
    )
    result = TSMOM(cfg).run_from_prices(data["close"])
    _report(
        "TSMOM",
        result.pnl,
        cfg.a,
        args.out,
        weights=result.weights,
        signal=result.signal,
        volatility=result.volatility,
        rebalance=result.rebalance,
    )


def _cmd_download(args: argparse.Namespace) -> None:
    path = download_csv(
        args.ticker, args.out, period=args.period, interval=args.interval, auto_adjust=args.adjusted
    )
    print(f"{args.ticker} written to {path}")


def build_parser() -> argparse.ArgumentParser:
    """The argument parser of the `tfu` command."""
    parser = argparse.ArgumentParser(
        prog="tfu",
        description="Trend-following systems: European, American and time-series momentum.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    parser.add_argument("--version", action="version", version=f"tfunify {__version__}")
    commands = parser.add_subparsers(dest="command", required=True, metavar="command")
    shown = " (default: %(default)s)"

    def add_system(name: str, help_: str, default_out: str) -> argparse.ArgumentParser:
        sub = commands.add_parser(name, help=help_, description=help_)
        sub.add_argument(
            "--csv", required=True, help="CSV file with a 'close' column (and high, low)"
        )
        sub.add_argument("--out", default=default_out, help="results file" + shown)
        sub.add_argument("-a", type=float, default=260, help="trading days per year" + shown)
        return sub

    def add_target(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--target", type=float, default=0.15, help="annual volatility target" + shown
        )
        sub.add_argument(
            "--span-sigma",
            type=float,
            default=33,
            help="span, in days, of the volatility estimate" + shown,
        )

    def add_spans(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--span-long", type=float, default=250, help="span, in days, of the slow filter" + shown
        )
        sub.add_argument(
            "--span-short", type=float, default=20, help="span, in days, of the fast filter" + shown
        )

    european = add_system("european", "run the European system", "european_results.npz")
    add_target(european)
    add_spans(european)
    european.add_argument(
        "--mode",
        choices=["longshort", "single"],
        default="longshort",
        help="long-short filter, or a single filter of span --span-long" + shown,
    )
    european.add_argument(
        "--longshort",
        action="store_const",
        const="longshort",
        dest="mode",
        help="same as --mode longshort",
    )
    european.set_defaults(run=_cmd_european)

    american = add_system("american", "run the American system", "american_results.npz")
    add_spans(american)
    american.add_argument(
        "--atr-period", type=int, default=33, help="days in the average true range" + shown
    )
    american.add_argument(
        "--q", type=float, default=5.0, help="entry buffer, in average true ranges" + shown
    )
    american.add_argument(
        "--p", type=float, default=5.0, help="stop distance, in average true ranges" + shown
    )
    american.add_argument(
        "--r-multiple",
        type=float,
        default=0.01,
        help="fraction of the capital an adverse move of one average true range costs" + shown,
    )
    american.set_defaults(run=_cmd_american)

    tsmom = add_system("tsmom", "run the time-series momentum system", "tsmom_results.npz")
    add_target(tsmom)
    tsmom.add_argument("--L", type=int, default=10, help="days in a period" + shown)
    tsmom.add_argument("--M", type=int, default=10, help="periods in the lookback" + shown)
    tsmom.add_argument(
        "--signs",
        choices=["daily", "period"],
        default="daily",
        help="signs of daily returns, or one sign per period" + shown,
    )
    tsmom.set_defaults(run=_cmd_tsmom)

    download = commands.add_parser(
        "download",
        help="download prices from Yahoo Finance",
        description='Download prices from Yahoo Finance; needs pip install "tfunify[yahoo]".',
    )
    download.add_argument("ticker", help="Yahoo symbol, for example SPY or ES=F")
    download.add_argument("--out", default="data.csv", help="CSV file to write" + shown)
    download.add_argument(
        "--period", default="5y", help="length of the history: 1y, 5y, max, ..." + shown
    )
    download.add_argument(
        "--interval",
        choices=["1d", "1wk", "1mo"],
        default="1d",
        help="spacing of the prices: daily, weekly or monthly" + shown,
    )
    download.add_argument(
        "--adjusted", action="store_true", help="adjust prices for dividends as well as splits"
    )
    download.set_defaults(run=_cmd_download)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments without the program name; `sys.argv[1:]` by default.

    Returns
    -------
    int
        0 on success, 1 when the command failed on its input (a message is
        printed to standard error). Usage errors exit with status 2, as usual
        for `argparse`.
    """
    args = build_parser().parse_args(argv)
    # a path the console cannot encode must not turn a finished run into an error
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(errors="backslashreplace")
    try:
        args.run(args)
    except (ValueError, OSError, ImportError) as error:
        print(f"tfu: error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
