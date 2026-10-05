# Command line

Installing tfunify adds the `tfu` program; `python -m tfunify` is the same thing.

```bash
tfu european --csv prices.csv --target 0.15 --span-long 250 --span-short 20
tfu european --csv prices.csv --mode single --span-long 63
tfu american --csv prices.csv --q 5 --p 5 --atr-period 33
tfu tsmom    --csv prices.csv --L 10 --M 10
tfu download SPY --out spy.csv --period 5y
```

Each system reads a price file, prints one line of statistics and writes its daily arrays to an `.npz` file:

```text
$ tfu european --csv prices.csv
European TF: annual return ..., annual volatility ..., Sharpe ratio ..., maximum drawdown ... (N days)
results written to european_results.npz
```

## Options

Common to the three systems:

| Option | Default | Meaning |
|---|---|---|
| `--csv FILE` | required | the price file |
| `--out FILE` | `<system>_results.npz` | where to write the arrays |
| `-a DAYS` | 260 | trading days per year |

`tfu european`

| Option | Default | Meaning |
|---|---|---|
| `--target` | 0.15 | annual volatility target |
| `--span-sigma` | 33 | span of the volatility estimate, in days |
| `--span-long` | 250 | span of the slow filter, in days |
| `--span-short` | 20 | span of the fast filter, in days |
| `--mode` | `longshort` | `longshort`, or `single` for one filter of span `--span-long` |
| `--longshort` | | the same as `--mode longshort`; kept from version 0.1 |

`tfu american`

| Option | Default | Meaning |
|---|---|---|
| `--span-long` | 250 | span of the slow filter of the close, in days |
| `--span-short` | 20 | span of the fast filter, in days |
| `--atr-period` | 33 | days in the average true range (ATR) |
| `--q` | 5.0 | entry buffer, in ATRs |
| `--p` | 5.0 | distance of the stop, in ATRs |
| `--r-multiple` | 0.01 | risk multiple: the share of the capital that a move of one ATR against the position costs |

`tfu tsmom`

| Option | Default | Meaning |
|---|---|---|
| `--target` | 0.15 | annual volatility target |
| `--span-sigma` | 33 | span of the volatility estimate, in days |
| `--L` | 10 | days in a period |
| `--M` | 10 | periods in the lookback |
| `--signs` | `daily` | `daily` or `period` |

`tfu download TICKER` (needs `pip install "tfunify[yahoo]"`)

| Option | Default | Meaning |
|---|---|---|
| `--out` | `data.csv` | the file to write |
| `--period` | `5y` | length of history: `1y`, `10y`, `max`, ... |
| `--interval` | `1d` | spacing of the prices: `1d`, `1wk` or `1mo` |
| `--adjusted` | off | adjust prices for dividends as well as for splits |

`tfu <command> --help` prints the same list, and `tfu --version` the version.

The systems count in observations. On weekly or monthly prices a span of 250 is 250 weeks or months, and `-a` should be 52 or 12 for the annual figures to come out right.

The results go to the file named by `--out`, exactly as it is given.

## Results file

```python
import numpy as np

with np.load("european_results.npz") as results:
    pnl = results["pnl"]
    weights = results["weights"]
```

| System | Arrays |
|---|---|
| European | `pnl`, `weights`, `signal`, `volatility` |
| American | `pnl`, `weights`, `position`, `stop`, `atr`, `fast`, `slow` |
| TSMOM | `pnl`, `weights`, `signal`, `volatility`, `rebalance` |

## Exit status

0 on success. 1 when the command failed on its input: a missing or malformed file, parameters that do not make sense, a series too short. The reason is printed to standard error. 2 for a usage error, as usual.

## Data format

A CSV file with a header row and a `close` column:

```csv
date,open,high,low,close,volume
2020-01-02,323.8,325.0,322.1,324.1,28000000
2020-01-03,325.2,327.1,324.8,326.9,31000000
```

- `high` and `low` are used by the American system. Without them the range of a day is the absolute change of the close. A file with only one of the two is an error.
- `open`, `volume` and `date` are read if present; other columns are ignored.
- The separator is a comma and the encoding UTF-8.
- Column names are matched without regard to case, so a file saved from Yahoo Finance works as it is.
- Rows must be in chronological order, oldest first. If the `date` column holds ISO dates (`2020-01-02`), the order is checked.
- A close, high or low that is not a positive number is an error, reported with its line. So is a high below its low.

From Python, `tfunify.data.load_csv` reads the same format and `tfunify.data.download_csv` writes it:

```python
from tfunify.data import load_csv

data = load_csv("prices.csv")
close, high, low = data["close"], data["high"], data["low"]
```
