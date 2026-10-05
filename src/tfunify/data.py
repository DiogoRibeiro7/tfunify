"""Reading price files, and downloading them from Yahoo Finance."""

from __future__ import annotations

import csv
import datetime as dt
import math
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = ["download_csv", "load_csv"]

_NUMERIC_COLUMNS = ("open", "high", "low", "close", "volume")
# The systems use these three; a value that is not a positive number is an error.
# open and volume are passed through, with NaN where the file has no number.
_REQUIRED_WHEN_PRESENT = ("high", "low", "close")


def load_csv(path: str | Path) -> dict[str, NDArray[Any]]:
    """Read daily prices from a CSV file.

    The file needs a header row with a `close` column. The columns `open`,
    `high`, `low`, `volume` and `date` are read when present; any other
    column is ignored. Column names are matched without regard to case or
    surrounding spaces, so a file exported from Yahoo Finance works as it is.

    Parameters
    ----------
    path : str or pathlib.Path
        The file, with rows in chronological order.

    Returns
    -------
    dict of numpy.ndarray
        Always `"close"`, `"high"` and `"low"`; when the file has neither a
        high nor a low column, both are copies of the close. `"open"` and
        `"volume"` are present when the file has them, with NaN where a cell
        is not a number, and so is `"date"`, as an array of strings.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the file is empty, is not UTF-8 text, has no `close` column, has
        only one of `high` and `low`, or has a column twice; if a close, high
        or low is not a positive number, or a high is below its low (the
        message names the line of the file); or if the dates are ISO dates
        or timestamps (`YYYY-MM-DD`, optionally with a time) that are not
        strictly increasing.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as handle:
            reader = csv.reader(handle)
            try:
                columns, dates = _read_rows(reader, csv_path)
            except csv.Error as error:
                raise ValueError(f"{csv_path}, line {reader.line_num}: {error}") from error
    except UnicodeDecodeError as error:
        raise ValueError(f"{csv_path} is not a UTF-8 text file: {error}") from error

    if not columns["close"]:
        raise ValueError(f"{csv_path} has a header but no rows")
    if dates:
        _check_chronological(dates, csv_path)

    data: dict[str, NDArray[Any]] = {
        name: np.asarray(values, dtype=np.float64) for name, values in columns.items()
    }
    for name in ("high", "low"):
        if name not in data:
            data[name] = data["close"].copy()
    if dates:
        data["date"] = np.asarray(dates, dtype=np.str_)
    return data


def _read_rows(reader: Any, csv_path: Path) -> tuple[dict[str, list[float]], list[str]]:
    """Parse the header and the rows of a price file."""
    header = next(reader, None)
    if header is None:
        raise ValueError(f"{csv_path} is empty")
    names = [name.strip().lower() for name in header]
    wanted = (*_NUMERIC_COLUMNS, "date")
    if "close" not in names:
        hint = ""
        if len(header) == 1 and any(separator in header[0] for separator in ";\t"):
            hint = " (the file must be comma-separated)"
        raise ValueError(
            f"{csv_path} has no 'close' column; its columns are "
            f"{', '.join(header) or '(none)'}{hint}"
        )
    repeated = sorted({name for name in names if name in wanted and names.count(name) > 1})
    if repeated:
        raise ValueError(f"{csv_path} has more than one column named {', '.join(repeated)}")
    if ("high" in names) != ("low" in names):
        present, absent = ("high", "low") if "high" in names else ("low", "high")
        raise ValueError(
            f"{csv_path} has a '{present}' column but no '{absent}' column; give both or neither"
        )
    index = {name: names.index(name) for name in wanted if name in names}
    columns: dict[str, list[float]] = {name: [] for name in _NUMERIC_COLUMNS if name in index}
    dates: list[str] = []

    for row in reader:
        if not any(cell.strip() for cell in row):
            continue  # blank line
        where = f"{csv_path}, line {reader.line_num}"
        parsed: dict[str, float] = {}
        for name in columns:
            position = index[name]
            text = row[position].strip() if position < len(row) else ""
            try:
                value = float(text)
            except ValueError:
                value = math.nan
            if name in _REQUIRED_WHEN_PRESENT and not (math.isfinite(value) and value > 0.0):
                shown = repr(text) if text else "nothing"
                raise ValueError(f"{where}: column '{name}' holds {shown}, not a positive number")
            parsed[name] = value
        if parsed.get("high", math.inf) < parsed.get("low", 0.0):
            raise ValueError(f"{where}: high {parsed['high']:g} is below low {parsed['low']:g}")
        for name, value in parsed.items():
            columns[name].append(value)
        if "date" in index:
            position = index["date"]
            dates.append(row[position].strip() if position < len(row) else "")
    return columns, dates


def download_csv(
    ticker: str,
    path: str | Path,
    period: str = "5y",
    interval: str = "1d",
    *,
    auto_adjust: bool = False,
) -> Path:
    """Download prices from Yahoo Finance into a CSV file.

    Needs the optional dependency: `pip install "tfunify[yahoo]"`.

    Parameters
    ----------
    ticker : str
        Yahoo symbol, for example `"SPY"` or `"ES=F"`.
    path : str or pathlib.Path
        File to write. Its columns are `date, open, high, low, close, volume`,
        the format `load_csv` reads.
    period : str, default "5y"
        Length of history, for example `"1y"`, `"5y"` or `"max"`.
    interval : str, default "1d"
        Sampling interval: `"1d"`, `"1wk"` or `"1mo"`.
    auto_adjust : bool, default False
        Passed to Yahoo: `True` adjusts all four prices for dividends as well
        as for splits, so that returns are total returns.

    Returns
    -------
    pathlib.Path
        The file written.

    Raises
    ------
    ImportError
        If `yfinance` is not installed.
    ValueError
        If Yahoo returns no usable rows for the symbol.
    """
    if interval not in ("1d", "1wk", "1mo"):
        raise ValueError(f"interval must be '1d', '1wk' or '1mo', got {interval!r}")
    try:
        import yfinance
    except ImportError as error:
        raise ImportError(
            'downloading needs the optional dependency yfinance: pip install "tfunify[yahoo]"'
        ) from error

    frame = yfinance.download(
        ticker, period=period, interval=interval, auto_adjust=auto_adjust, progress=False
    )
    if frame is None or len(frame) == 0:
        raise ValueError(f"Yahoo returned no data for {ticker!r} (period={period}, {interval})")

    # Recent yfinance versions return two column levels (price, ticker) even
    # for one symbol; keep the level that holds the price names.
    columns = frame.columns
    if getattr(columns, "nlevels", 1) > 1:
        for level in range(columns.nlevels):
            if "Close" in columns.get_level_values(level):
                frame = frame.copy()
                frame.columns = columns.get_level_values(level)
                break
    missing = [name for name in ("Open", "High", "Low", "Close") if name not in frame.columns]
    if missing:
        raise ValueError(f"Yahoo data for {ticker!r} lacks the column(s) {', '.join(missing)}")

    opens, highs, lows, closes = (
        np.asarray(frame[name], dtype=np.float64).reshape(-1)
        for name in ("Open", "High", "Low", "Close")
    )
    if "Volume" in frame.columns:
        volumes = np.nan_to_num(np.asarray(frame["Volume"], dtype=np.float64).reshape(-1))
    else:
        volumes = np.zeros(closes.size)
    prices = np.column_stack((opens, highs, lows, closes))
    usable = np.all(np.isfinite(prices) & (prices > 0.0), axis=1)  # skip incomplete rows
    if not np.any(usable):
        raise ValueError(f"Yahoo returned no complete rows for {ticker!r}")

    out_path = Path(path)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "high", "low", "close", "volume"])
        for keep, stamp, row, volume in zip(usable, frame.index, prices.tolist(), volumes.tolist()):
            if keep:
                writer.writerow([stamp.strftime("%Y-%m-%d"), *row, int(volume)])
    return out_path


def _parse_stamp(text: str) -> dt.datetime:
    """An ISO date or timestamp as a datetime (a date counts as its midnight)."""
    try:
        return dt.datetime.fromisoformat(text)
    except ValueError:  # a suffix this Python does not read: go by the date
        return dt.datetime.combine(dt.date.fromisoformat(text[:10]), dt.time())


def _check_chronological(dates: list[str], csv_path: Path) -> None:
    """Raise if the dates are ISO dates or timestamps that are not strictly increasing."""
    try:
        parsed = [_parse_stamp(text) for text in dates]
        late = [current <= previous for previous, current in pairwise(parsed)]
    except (ValueError, TypeError):
        # another date format, or stamps with and without a time zone that
        # cannot be compared: the order is not checked
        return
    if any(late):
        row = late.index(True) + 1
        raise ValueError(
            f"{csv_path}: the rows must be in chronological order, but data row {row + 1} "
            f"({dates[row]}) does not come after the row before it ({dates[row - 1]})"
        )
