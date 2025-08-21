from __future__ import annotations

import argparse
import sys
import numpy as np
from numpy.typing import NDArray

from .european import EuropeanTF, EuropeanTFConfig
from .american import AmericanTF, AmericanTFConfig
from .tsmom import TSMOM, TSMOMConfig


def _load_csv(path: str) -> dict[str, NDArray[np.floating]]:
    """
    Load a CSV with at least 'close' column; 'high'/'low' are optional and
    default to 'close' if missing.
    """
    import csv
    cols: dict[str, list[float]] = {"close": [], "high": [], "low": []}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "close" not in reader.fieldnames:
            raise ValueError("CSV must contain a 'close' column.")
        for row in reader:
            cols["close"].append(float(row["close"]))
            # Fall back to 'close' if high/low are absent
            cols["high"].append(float(row.get("high", row["close"])))
            cols["low"].append(float(row.get("low", row["close"])))
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _summarise(daily: NDArray[np.floating], trading_days: int = 260) -> str:
    """
    Print annualised mean/vol and Sharpe from a vector of daily PnL/returns.
    """
    mu = float(np.nanmean(daily))
    sd = float(np.nanstd(daily, ddof=0))
    ann_mu = mu * trading_days
    ann_sd = sd * (trading_days ** 0.5)
    sharpe = ann_mu / ann_sd if ann_sd > 0 else float("nan")
    return f"Ann μ={ann_mu:.3%}  Ann σ={ann_sd:.3%}  SR={sharpe:.2f}"


# ---------------------------
# Systems subcommands
# ---------------------------

def cmd_european(args: argparse.Namespace) -> int:
    data = _load_csv(args.csv)
    cfg = EuropeanTFConfig(
        sigma_target_annual=args.target,
        a=args.a,
        span_sigma=args.span_sigma,
        mode="longshort" if args.longshort else "single",
        span_long=args.span_long,
        span_short=args.span_short,
    )
    sysm = EuropeanTF(cfg)
    f, w, s, sigma = sysm.run_from_prices(data["close"])
    np.savez("european_results.npz", f=f, w=w, s=s, sigma=sigma)
    print(_summarise(f, cfg.a))
    return 0


def cmd_american(args: argparse.Namespace) -> int:
    data = _load_csv(args.csv)
    cfg = AmericanTFConfig(
        span_long=args.span_long,
        span_short=args.span_short,
        atr_period=args.atr_period,
        q=args.q,
        p=args.p,
        r_multiple=args.r_multiple,
    )
    sysm = AmericanTF(cfg)
    pnl, units = sysm.run(data["close"], data["high"], data["low"])
    np.savez("american_results.npz", pnl=pnl, units=units)
    print(_summarise(pnl, args.a))
    return 0


def cmd_tsmom(args: argparse.Namespace) -> int:
    data = _load_csv(args.csv)
    cfg = TSMOMConfig(
        sigma_target_annual=args.target,
        a=args.a,
        span_sigma=args.span_sigma,
        L=args.L,
        M=args.M,
    )
    sysm = TSMOM(cfg)
    f, w, sgrid, sigma = sysm.run_from_prices(data["close"])
    np.savez("tsmom_results.npz", f=f, w=w, sgrid=sgrid, sigma=sigma)
    print(_summarise(f, cfg.a))
    return 0


# ---------------------------
# Downloader subcommand
# ---------------------------

def _download_csv_yahoo(ticker: str, out_path: str, period: str, interval: str) -> str:
    """
    Minimal Yahoo Finance → CSV downloader without pandas.

    Writes a CSV with headers:
      date,open,high,low,close,volume
    """
    try:
        import yfinance as yf
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "The 'download' command requires yfinance. Install extra:\n"
            "  pip install tfunify[yahoo]\n"
            "or with Poetry:\n"
            "  poetry add tfunify[yahoo]"
        ) from e

    # Fetch dataframe-like object; we'll only read columns we need.
    df = yf.download(
        ticker, period=period, interval=interval, auto_adjust=False, progress=False
    )
    # Robust emptiness check (yfinance returns an empty DataFrame on errors)
    if getattr(df, "empty", True):
        raise SystemExit(f"No data returned for {ticker} (period={period}, interval={interval}).")

    # Write a tiny CSV with only the necessary columns
    import csv
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "open", "high", "low", "close", "volume"])
        # df.iterrows() yields (Timestamp, row)
        for dt, row in df.iterrows():
            w.writerow([
                dt.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
            ])
    return out_path


def cmd_download(args: argparse.Namespace) -> int:
    out = _download_csv_yahoo(
        ticker=args.ticker,
        out_path=args.out,
        period=args.period,
        interval=args.interval,
    )
    print(f"Saved to {out}")
    return 0


# ---------------------------
# Entry point
# ---------------------------

def main() -> int:
    p = argparse.ArgumentParser(prog="tfu", description="Unified trend-following runners")
    sub = p.add_subparsers(dest="cmd", required=True)

    # european
    p_eu = sub.add_parser("european")
    p_eu.add_argument("--csv", required=True)
    p_eu.add_argument("--target", type=float, default=0.15)
    p_eu.add_argument("-a", type=int, default=260, help="trading days per year")
    p_eu.add_argument("--span-sigma", type=int, default=33)
    p_eu.add_argument("--span-long", type=int, default=250)
    p_eu.add_argument("--span-short", type=int, default=20)
    p_eu.add_argument("--longshort", action="store_true", help="use long–short filter")
    p_eu.set_defaults(func=cmd_european)

    # american
    p_am = sub.add_parser("american")
    p_am.add_argument("--csv", required=True)
    p_am.add_argument("-a", type=int, default=260)
    p_am.add_argument("--span-long", type=int, default=250)
    p_am.add_argument("--span-short", type=int, default=20)
    p_am.add_argument("--atr-period", type=int, default=33)
    p_am.add_argument("--q", type=float, default=5.0)
    p_am.add_argument("--p", type=float, default=5.0)
    p_am.add_argument("--r-multiple", type=float, default=0.01)
    p_am.set_defaults(func=cmd_american)

    # tsmom
    p_tm = sub.add_parser("tsmom")
    p_tm.add_argument("--csv", required=True)
    p_tm.add_argument("--target", type=float, default=0.15)
    p_tm.add_argument("-a", type=int, default=260)
    p_tm.add_argument("--span-sigma", type=int, default=33)
    p_tm.add_argument("--L", type=int, default=10)
    p_tm.add_argument("--M", type=int, default=10)
    p_tm.set_defaults(func=cmd_tsmom)

    # download
    p_dl = sub.add_parser("download", help="Download OHLCV CSV from Yahoo Finance")
    p_dl.add_argument("ticker", help="Yahoo ticker (e.g. SPY, ES=F)")
    p_dl.add_argument("--out", default="data.csv", help="Output CSV path")
    p_dl.add_argument("--period", default="5y", help="1y, 3y, 5y, max, ...")
    p_dl.add_argument("--interval", choices=["1d", "1wk", "1mo"], default="1d")
    p_dl.set_defaults(func=cmd_download)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
