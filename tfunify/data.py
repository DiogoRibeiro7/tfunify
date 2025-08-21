from __future__ import annotations

import csv
from pathlib import Path
from typing import Literal, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import yfinance as yf


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

    Raises
    ------
    ImportError
        If yfinance is not installed
    ValueError
        If no data is returned for the ticker

    CSV format
    ----------
    date,open,high,low,close,volume
    2020-01-02,323.8,325.0,322.1,324.1,28000000
    ...
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required for tfunify.data. " "Install with `pip install tfunify[yahoo]`."
        ) from e

    path = Path(path)
    df = yf.download(ticker, period=period, interval=interval, auto_adjust=False, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for date, row in df.iterrows():
            writer.writerow(
                [
                    date.strftime("%Y-%m-%d"),
                    float(row["Open"]),
                    float(row["High"]),
                    float(row["Low"]),
                    float(row["Close"]),
                    int(row["Volume"]),
                ]
            )
    return path


def load_csv(path: str | Path) -> dict[str, np.ndarray]:
    """
    Load a CSV saved by `download_csv` into numpy arrays.

    Parameters
    ----------
    path : str | Path
        Path to CSV file

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with keys: "close", "high", "low", "open", "volume"

    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist
    ValueError
        If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    data = {k: [] for k in ["open", "high", "low", "close", "volume"]}

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV file appears to be empty or malformed")

        missing_cols = set(data.keys()) - set(reader.fieldnames)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        for row in reader:
            try:
                for k in data:
                    data[k].append(float(row[k]))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Error parsing row {reader.line_num}: {e}") from e

    return {k: np.asarray(v, dtype=float) for k, v in data.items()}
