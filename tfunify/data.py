from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal

import numpy as np

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "yfinance is required for tfunify.data. "
        "Install with `pip install tfunify[yahoo]`."
    ) from e


def download_csv(
    ticker: str,
    path: str | Path,
    period: str = "5y",
    interval: Literal["1d", "1wk", "1mo"] = "1d",
) -> Path:
    """
    Download OHLCV data from Yahoo Finance and save to CSV.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., "SPY", "ES=F").
    path : str | Path
        Destination CSV file.
    period : str, default="5y"
        History length (e.g., "1y", "5y", "max").
    interval : {"1d","1wk","1mo"}, default="1d"
        Data interval.

    Returns
    -------
    Path
        Path to the written CSV file.

    CSV format
    ----------
    date,open,high,low,close,volume
    2020-01-02,323.8,325.0,322.1,324.1,28000000
    ...
    """
    path = Path(path)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for date, row in df.iterrows():
            writer.writerow([
                date.strftime("%Y-%m-%d"),
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
                int(row["Volume"]),
            ])
    return path


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a CSV saved by `download_csv` into numpy arrays.

    Returns
    -------
    dict with keys: "close", "high", "low", "open", "volume"
    """
    import csv

    data = {k: [] for k in ["open", "high", "low", "close", "volume"]}
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k in data:
                data[k].append(float(row[k]))
    return {k: np.asarray(v, dtype=float) for k, v in data.items()}
