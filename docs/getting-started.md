# Getting started

## Installation

Install it from GitHub:

```bash
# the latest release
pip install "tfunify @ git+https://github.com/DiogoRibeiro7/tfunify@main"

# with the Yahoo Finance downloader
pip install "tfunify[yahoo] @ git+https://github.com/DiogoRibeiro7/tfunify@main"

# one version
pip install "tfunify @ git+https://github.com/DiogoRibeiro7/tfunify@v0.2.0"

# the development branch
pip install "tfunify @ git+https://github.com/DiogoRibeiro7/tfunify"
```

Python 3.10 or later and NumPy 1.24 or later.

## A price series

The systems take daily prices as a one-dimensional array, oldest first, all strictly positive. For the examples on this page, a simulated series will do:

```python
import numpy as np

rng = np.random.default_rng(1)
close = 100 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(3000)))
high = close * (1 + 0.005 * rng.random(3000))
low = close * (1 - 0.005 * rng.random(3000))
```

To read your own data, see [the data format](cli.md#data-format); `tfunify.data.load_csv` returns the arrays.

## The European system

```python
from tfunify import EuropeanTF, EuropeanTFConfig

cfg = EuropeanTFConfig(
    sigma_target_annual=0.15,  # volatility target of the position
    span_sigma=33,  # span of the volatility estimate, in days
    mode="longshort",  # "longshort" or "single"
    span_long=250,  # slow filter
    span_short=20,  # fast filter
)
result = EuropeanTF(cfg).run_from_prices(close)
```

`result` holds four arrays aligned with `close`:

| Attribute | Meaning |
|---|---|
| `pnl` | daily return of the position per unit of capital |
| `weights` | exposure per unit of capital, decided at the close of each day |
| `signal` | the trend signal; unit variance for serially independent returns |
| `volatility` | the daily volatility estimate; NaN during the warm-up |

It also unpacks: `pnl, weights, signal, volatility = result`.

## The American system

```python
from tfunify import AmericanTF, AmericanTFConfig

cfg = AmericanTFConfig(
    span_long=250,  # slow filter of the price
    span_short=20,  # fast filter
    atr_period=33,  # days in the average true range (ATR)
    q=5.0,  # entry buffer, in ATRs
    p=5.0,  # width of the trailing stop, in ATRs
    r_multiple=0.01,  # weight of a new position = r_multiple * price / ATR
)
result = AmericanTF(cfg).run(close, high, low)
```

`high` and `low` are optional (give both or neither); without them the range of a day is the absolute change of the close. Besides `pnl` and `weights`, the result has `position` (+1, 0 or -1), `stop`, `atr`, `fast` and `slow`.

## Time-series momentum

```python
from tfunify import TSMOM, TSMOMConfig

cfg = TSMOMConfig(
    sigma_target_annual=0.15,
    L=10,  # days in a period: the position is rebalanced every L days
    M=10,  # periods in the lookback
)
result = TSMOM(cfg).run_from_prices(close)
```

The result has `pnl`, `weights`, `signal`, `volatility` and `rebalance`, a boolean array that marks the rebalancing days.

## Summarising a result

```python
from tfunify import performance_summary

stats = performance_summary(result.pnl, a=260)
print(stats)  # one line with all five figures
print(stats.sharpe_ratio)  # or each figure on its own
```

The summary gives the annual return and volatility, the Sharpe ratio, the maximum drawdown and the number of days. All of them treat `pnl` as a return per unit of capital that is not compounded; see [Conventions](systems/conventions.md).

## Next

- [What each system computes](systems/index.md), with the formulas.
- [Closed forms](theory.md): what a filter should earn on returns with a given autocorrelation.
- [The command line](cli.md), to run a system on a CSV file.
