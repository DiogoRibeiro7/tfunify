from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

from .european import EuropeanTF, EuropeanTFConfig
from .american import AmericanTF, AmericanTFConfig
from .tsmom import TSMOM, TSMOMConfig

if TYPE_CHECKING:
    import yfinance as yf


def _load_csv(path: str) -> dict[str, NDArray[np.floating]]:
    """
    Load a CSV with at least 'close' column; 'high'/'low' are optional and
    default to 'close' if missing.
    
    Parameters
    ----------
    path : str
        Path to CSV file
        
    Returns
    -------
    dict[str, NDArray[np.floating]]
        Dictionary with 'close', 'high', 'low' arrays
        
    Raises
    ------
    FileNotFoundError
        If CSV file doesn't exist
    ValueError
        If CSV doesn't contain required 'close' column
    """
    import csv
    from pathlib import Path
    
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    cols: dict[str, list[float]] = {"close": [], "high": [], "low": []}
    
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames or "close" not in reader.fieldnames:
                raise ValueError("CSV must contain a 'close' column.")
            
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    cols["close"].append(float(row["close"]))
                    # Fall back to 'close' if high/low are absent
                    cols["high"].append(float(row.get("high", row["close"])))
                    cols["low"].append(float(row.get("low", row["close"])))
                except (ValueError, KeyError) as e:
                    raise ValueError(f"Error parsing row {row_count}: {e}") from e
                    
            if row_count == 0:
                raise ValueError("CSV file is empty")
                
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}") from e
        
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def _summarise(daily: NDArray[np.floating], trading_days: int = 260) -> str:
    """
    Generate summary statistics from daily PnL/returns.
    
    Parameters
    ----------
    daily : NDArray[np.floating]
        Daily PnL or returns
    trading_days : int, default=260
        Trading days per year for annualization
        
    Returns
    -------
    str
        Formatted summary string with annualized metrics
    """
    if len(daily) == 0:
        return "No data to summarize"
        
    # Filter out non-finite values for statistics
    valid_data = daily[np.isfinite(daily)]
    if len(valid_data) == 0:
        return "No valid data to summarize"
        
    mu = float(np.mean(valid_data))
    sd = float(np.std(valid_data, ddof=0))
    ann_mu = mu * trading_days
    ann_sd = sd * (trading_days ** 0.5)
    sharpe = ann_mu / ann_sd if ann_sd > 0 else float("nan")
    
    return f"Ann μ={ann_mu:.3%}  Ann σ={ann_sd:.3%}  SR={sharpe:.2f}  (n={len(valid_data)})"


# ---------------------------
# Systems subcommands
# ---------------------------

def cmd_european(args: argparse.Namespace) -> int:
    """Run European TF system."""
    try:
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
        
        output_file = "european_results.npz"
        np.savez(output_file, f=f, w=w, s=s, sigma=sigma)
        
        print("European TF Results:")
        print(_summarise(f, cfg.a))
        print(f"Results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error running European TF: {e}", file=sys.stderr)
        return 1


def cmd_american(args: argparse.Namespace) -> int:
    """Run American TF system."""
    try:
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
        
        output_file = "american_results.npz"
        np.savez(output_file, pnl=pnl, units=units)
        
        print("American TF Results:")
        print(_summarise(pnl, args.a))
        print(f"Results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error running American TF: {e}", file=sys.stderr)
        return 1


def cmd_tsmom(args: argparse.Namespace) -> int:
    """Run TSMOM system."""
    try:
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
        
        output_file = "tsmom_results.npz"
        np.savez(output_file, f=f, w=w, sgrid=sgrid, sigma=sigma)
        
        print("TSMOM Results:")
        print(_summarise(f, cfg.a))
        print(f"Results saved to {output_file}")
        
        return 0
    except Exception as e:
        print(f"Error running TSMOM: {e}", file=sys.stderr)
        return 1


# ---------------------------
# Downloader subcommand
# ---------------------------

def _download_csv_yahoo(ticker: str, out_path: str, period: str, interval: str) -> str:
    """
    Minimal Yahoo Finance → CSV downloader without pandas.

    Parameters
    ----------
    ticker : str
        Yahoo Finance ticker symbol
    out_path : str
        Output CSV path
    period : str
        Data period (e.g., "1y", "5y")
    interval : str
        Data interval (e.g., "1d", "1wk")
        
    Returns
    -------
    str
        Path to output file
        
    Raises
    ------
    ImportError
        If yfinance is not installed
    SystemExit
        If no data is returned
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit(
            "The 'download' command requires yfinance. Install extra:\n"
            "  pip install tfunify[yahoo]\n"
            "or with Poetry:\n"
            "  poetry add tfunify[yahoo]"
        ) from e

    # Fetch dataframe-like object; we'll only read columns we need.
    print(f"Downloading {ticker} data...")
    df = yf.download(
        ticker, period=period, interval=interval, auto_adjust=False, progress=False
    )
    
    # Robust emptiness check (yfinance returns an empty DataFrame on errors)
    if getattr(df, "empty", True):
        raise SystemExit(f"No data returned for {ticker} (period={period}, interval={interval}).")

    # Write a CSV with the necessary columns
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
    """Download data from Yahoo Finance."""
    try:
        out = _download_csv_yahoo(
            ticker=args.ticker,
            out_path=args.out,
            period=args.period,
            interval=args.interval,
        )
        print(f"Successfully saved {args.ticker} data to {out}")
        return 0
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error downloading data: {e}", file=sys.stderr)
        return 1


# ---------------------------
# Entry point
# ---------------------------

def main() -> int:
    """Main CLI entry point."""
    p = argparse.ArgumentParser(
        prog="tfu", 
        description="Unified trend-following systems CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  tfu european --csv data.csv --target 0.15 --longshort
  tfu american --csv data.csv --q 2.0 --p 3.0
  tfu tsmom --csv data.csv --L 5 --M 12
  tfu download SPY --out spy_data.csv --period 2y
        """
    )
    sub = p.add_subparsers(dest="cmd", required=True, help="Available commands")

    # european
    p_eu = sub.add_parser("european", help="Run European TF system")
    p_eu.add_argument("--csv", required=True, help="Input CSV file with OHLC data")
    p_eu.add_argument("--target", type=float, default=0.15, help="Target annual volatility")
    p_eu.add_argument("-a", type=int, default=260, help="Trading days per year")
    p_eu.add_argument("--span-sigma", type=int, default=33, help="Volatility estimation span")
    p_eu.add_argument("--span-long", type=int, default=250, help="Long moving average span")
    p_eu.add_argument("--span-short", type=int, default=20, help="Short moving average span")
    p_eu.add_argument("--longshort", action="store_true", help="Use long–short filter")
    p_eu.set_defaults(func=cmd_european)

    # american
    p_am = sub.add_parser("american", help="Run American TF system")
    p_am.add_argument("--csv", required=True, help="Input CSV file with OHLC data")
    p_am.add_argument("-a", type=int, default=260, help="Trading days per year")
    p_am.add_argument("--span-long", type=int, default=250, help="Long moving average span")
    p_am.add_argument("--span-short", type=int, default=20, help="Short moving average span")
    p_am.add_argument("--atr-period", type=int, default=33, help="ATR calculation period")
    p_am.add_argument("--q", type=float, default=5.0, help="Entry threshold multiplier")
    p_am.add_argument("--p", type=float, default=5.0, help="Stop loss multiplier")
    p_am.add_argument("--r-multiple", type=float, default=0.01, help="Risk multiple for position sizing")
    p_am.set_defaults(func=cmd_american)

    # tsmom
    p_tm = sub.add_parser("tsmom", help="Run TSMOM system")
    p_tm.add_argument("--csv", required=True, help="Input CSV file with OHLC data")
    p_tm.add_argument("--target", type=float, default=0.15, help="Target annual volatility")
    p_tm.add_argument("-a", type=int, default=260, help="Trading days per year")
    p_tm.add_argument("--span-sigma", type=int, default=33, help="Volatility estimation span")
    p_tm.add_argument("--L", type=int, default=10, help="Block length")
    p_tm.add_argument("--M", type=int, default=10, help="Number of blocks")
    p_tm.set_defaults(func=cmd_tsmom)

    # download
    p_dl = sub.add_parser("download", help="Download OHLCV CSV from Yahoo Finance")
    p_dl.add_argument("ticker", help="Yahoo ticker (e.g. SPY, ES=F)")
    p_dl.add_argument("--out", default="data.csv", help="Output CSV path")
    p_dl.add_argument("--period", default="5y", help="Data period (1y, 3y, 5y, max, ...)")
    p_dl.add_argument("--interval", choices=["1d", "1wk", "1mo"], default="1d", help="Data interval")
    p_dl.set_defaults(func=cmd_download)

    try:
        args = p.parse_args()
        return args.func(args)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
