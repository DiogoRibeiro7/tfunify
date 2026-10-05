# tfunify

[![CI](https://github.com/DiogoRibeiro7/tfunify/actions/workflows/ci.yml/badge.svg)](https://github.com/DiogoRibeiro7/tfunify/actions/workflows/ci.yml)
[![Docs](https://github.com/DiogoRibeiro7/tfunify/actions/workflows/docs.yml/badge.svg)](https://diogoribeiro7.github.io/tfunify/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/DiogoRibeiro7/tfunify/blob/develop/LICENSE)

Three trend-following systems in NumPy, as defined by Sepp and Lucic in
*The Science and Practice of Trend-Following Systems*
([arXiv:2607.19497](https://arxiv.org/abs/2607.19497),
[SSRN 3167787](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3167787)):

| System | Signal | Position |
|---|---|---|
| **European** | exponentially weighted moving average (EWMA), single or long-short, of volatility-normalised returns | proportional to the signal, sized to a volatility target; changes every day |
| **American** | fast EWMA of the price crosses the slow one by more than a buffer of average true ranges (ATR) | fixed when opened, closed by a trailing stop |
| **TSMOM** (time-series momentum) | sum of the signs of past returns over `M` periods of `L` days | sized to a volatility target; rebalanced every `L` days |

It also evaluates the paper's closed forms for the European system: expected
return, volatility and Sharpe ratio from the autocorrelation and the drift of
the returns, and the turnover of the signal. A backtest can then be compared
with what the data imply.

The only dependency is NumPy. The package is typed, and every formula is tested
against an independent evaluation of the paper's definition.

This is an independent implementation. The authors' own package is
[trendfollowing](https://github.com/ArturSepp/TrendFollowingSystems) (GPL-3.0);
[Relation to the paper](#relation-to-the-paper) lists where the two differ.

## Installation

Install it from GitHub:

```bash
# the latest release
pip install "tfunify @ git+https://github.com/DiogoRibeiro7/tfunify@main"

# with the Yahoo Finance downloader
pip install "tfunify[yahoo] @ git+https://github.com/DiogoRibeiro7/tfunify@main"
```

`@main` is the released code; a tag such as `@v0.2.0` pins one version, and
without either you get the development branch. Python 3.10 or later, NumPy 1.24
or later.

## Quick start

```python
import numpy as np
from tfunify import EuropeanTF, EuropeanTFConfig, performance_summary

# 20 years of simulated daily prices
rng = np.random.default_rng(0)
prices = 100 * np.cumprod(1 + 0.0003 + 0.01 * rng.standard_normal(5200))

cfg = EuropeanTFConfig(
    sigma_target_annual=0.15,  # volatility target of the position
    a=260,  # trading days per year
    span_sigma=33,  # span of the volatility estimate, in days
    mode="longshort",  # "longshort" or "single"
    span_long=250,  # slow filter
    span_short=20,  # fast filter
)
result = EuropeanTF(cfg).run_from_prices(prices)

print(result.pnl.shape)  # (5200,): daily return per unit of capital
print(performance_summary(result.pnl))  # annual return, volatility, Sharpe ratio, drawdown
pnl, weights, signal, volatility = result  # the result also unpacks, in this order
```

### American system

```python
import numpy as np
from tfunify import AmericanTF, AmericanTFConfig

rng = np.random.default_rng(1)
close = 100 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(3000)))
high = close * (1 + 0.005 * rng.random(3000))
low = close * (1 - 0.005 * rng.random(3000))

cfg = AmericanTFConfig(
    span_long=250,  # slow filter of the price
    span_short=20,  # fast filter
    atr_period=33,  # days in the average true range
    q=5.0,  # entry buffer, in ATRs
    p=5.0,  # width of the trailing stop, in ATRs
    r_multiple=0.01,  # weight of a new position = r_multiple * price / ATR
)
result = AmericanTF(cfg).run(close, high, low)  # high and low are optional

print(sorted(set(result.position.tolist())))  # [-1, 0, 1]: short, flat, long
print(np.isnan(result.stop[result.position == 0]).all())  # True: no stop when flat
```

`result` carries `pnl`, `weights`, `position`, `stop`, `atr`, `fast` and `slow`,
and unpacks into `pnl, weights`.

### Time-series momentum

```python
import numpy as np
from tfunify import TSMOM, TSMOMConfig

rng = np.random.default_rng(2)
prices = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(3000)))

cfg = TSMOMConfig(
    sigma_target_annual=0.15,
    span_sigma=33,
    L=10,  # days in a period: the position is rebalanced every L days
    M=10,  # periods in the lookback
    signs="daily",  # the paper's definition; "period" takes one sign per period
)
result = TSMOM(cfg).run_from_prices(prices)

print(int(result.rebalance.sum()))  # 290 rebalancing days
print(np.flatnonzero(result.rebalance)[:3].tolist())  # [100, 110, 120]
```

## What the systems compute

In the notation below `r` is the daily return `s[t] / s[t-1] - 1`, `EWMA(y, span)`
is the filter `out[t] = (1 - nu) * y[t] + nu * out[t-1]` with
`nu = 1 - 2 / (span + 1)`, and `a` is the number of trading days in a year.
The [documentation](https://diogoribeiro7.github.io/tfunify/) has the formulas
typeset.

**Volatility and normalised returns**, shared by the European and TSMOM systems:

```text
sigma[t] = sqrt(EWMA(r**2, span_sigma)[t])      daily volatility, no mean subtracted
z[t]     = r[t] / sigma[t-1]                    the lag keeps a return from scaling itself
```

**European.** A signal with unit variance for serially independent returns,
times the volatility-target weight:

```text
single:      S[t] = sqrt((1 + nu) / (1 - nu)) * EWMA(z, span_long)[t]
long-short:  S[t] = l1 * EWMA(z, span_long)[t] - l2 * EWMA(z, span_short)[t]
             l1 = q / (1 - nu1),  l2 = q / (1 - nu2)
             q  = (1/(1 - nu1**2) + 1/(1 - nu2**2) - 2/(1 - nu1*nu2)) ** -0.5

w[t]   = S[t] * sigma_target / (sqrt(a) * sigma[t])
pnl[t] = w[t-1] * r[t]
```

**American.** With `fast` and `slow` the EWMA filters of the close and `ATR` the
average true range over `atr_period` days:

```text
from flat:   long  if fast[t] > slow[t] + q * ATR[t]
             short if fast[t] < slow[t] - q * ATR[t]
             weight = +/- r_multiple * close[t] / ATR[t]
             stop   = close[t] -/+ p * ATR[t]

when long:   exit if close[t] < stop[t-1] and not (fast[t] > slow[t] + q * ATR[t])
             otherwise stop[t] = max(stop[t-1], close[t] - p * ATR[t])
```

Short positions mirror the long ones. The size stays as it was at entry, and
the exit is a trailing stop, not a reversal of the signal.

**TSMOM.** On every `L`-th day, over the last `M * L` days:

```text
S[t] = sum(sign(r)) / sqrt(M * L)
w[t] = S[t] * sigma_target / (sqrt(a) * sigma[t-1])      held until the next rebalancing day
```

With `signs="period"` the signal takes one sign per period, that of the sum of
its normalised returns, and divides by `sqrt(M)`. This is the construction of
tfunify 0.1, kept as an option; it is not the one of Moskowitz, Ooi and
Pedersen, who take the sign of the return over the trailing twelve months.

### Conventions

- **Timing.** An array entry dated `t` uses data up to the close of day `t`.
  A weight dated `t` earns the return of day `t + 1`: `pnl[t] = weights[t-1] * r[t]`.
  The tests check that no output changes when later data are appended.
- **Units.** `pnl` is a return per unit of capital, not compounded. Weights are
  exposures per unit of capital; with futures they can exceed one.
- **Prices or returns.** `run_from_prices` uses simple returns and returns
  arrays aligned with the prices (index 0 has no return). `run_from_returns`
  treats every entry as an observed return.
- **Warm-up.** The volatility estimate needs data before it can be used. Until
  `warmup` returns have been seen (by default `span_sigma` of them) `volatility`
  is NaN and the weight is zero, and no return enters the European signal.
  Filters of the signal start from zero, so the position builds up as evidence
  accumulates.
- **Where a series starts.** The volatility estimate and the ATR start at the
  first return and the first true range that are not zero. Unchanged prices in
  front of a series, the padding of an instrument that started trading later,
  are not observations of a volatility of zero.
- **No hidden limits.** Volatility is not clipped and weights are not capped
  unless you ask: `sigma_floor_annual` puts a floor under the volatility
  estimate and `weight_cap` bounds the absolute weight. Without them a market
  that goes quiet and then wakes up can produce very large weights; that is the
  definition, and the options exist for that case.

## Closed forms

Given the autocorrelations of the returns and the Sharpe ratio of the
instrument, the paper gives the expected return, the variance and the Sharpe
ratio of the European system. `tfunify.theory` evaluates them:

```python
from tfunify import theory

acf = theory.ar1_acf(0.05, 500)  # first-order autoregressive, AR(1), returns: coefficient 0.05

print(round(theory.european_sharpe_ratio(span_long=21, acf=acf), 2))  # 0.34
print(round(theory.european_sharpe_ratio(span_long=250, acf=acf), 2))  # 0.1
print(round(theory.european_sharpe_ratio(span_long=250, instrument_sharpe=0.5), 2))  # 0.22

# annual turnover of the signal on white noise: single filter, and long-short LS(250, 20)
print(round(theory.expected_signal_turnover(span_long=250), 2))  # 3.93
print(round(theory.expected_signal_turnover(span_long=250, span_short=20), 2))  # 0.88
```

A short filter earns from autocorrelation at short lags and a long one from
drift. The expected return holds for any distribution; the variance and the
Sharpe ratio are for Gaussian returns (the paper's correction for excess
kurtosis is not included).

The two turnover figures are the paper's proxy: the turnover of the signal
alone, for Gaussian white noise and a 15 % target, in units of
volatility-weighted exposure. A system also trades the changes of its
volatility estimate. In simulation the single 250-day filter turns over 4.1 a
year and the long-short filter 1.4, about a third as much rather than the
quarter the proxy suggests.

## Summary statistics

```python
import numpy as np
from tfunify import performance_summary
from tfunify.metrics import max_drawdown, sharpe_ratio

pnl = np.array([0.01, -0.02, 0.03, 0.00])
print(performance_summary(pnl, a=260))
# annual return 130.00%, annual volatility 33.57%, Sharpe ratio 3.87, maximum drawdown 2.00% (4 days)
print(round(sharpe_ratio(pnl), 2), round(max_drawdown(pnl), 2))  # 3.87 0.02
```

Annual figures are arithmetic and the Sharpe ratio is the ratio of the moments
of daily returns, without a risk-free rate: futures returns are excess returns
already. Returns of a share or a fund are not; subtract the financing rate from
them first if that matters for your comparison.

## Command line

```bash
tfu european --csv prices.csv --target 0.15 --span-long 250 --span-short 20
tfu european --csv prices.csv --mode single --span-long 63
tfu american --csv prices.csv --q 5 --p 5 --atr-period 33
tfu tsmom    --csv prices.csv --L 10 --M 10
tfu download SPY --out spy.csv --period 5y     # needs tfunify[yahoo]
```

Each system prints its summary statistics and writes the daily arrays to an
`.npz` file (`--out`, by default `european_results.npz` and so on, in the
current directory). `tfu <command> --help` lists the options; `python -m tfunify`
is the same program.

### Data format

A CSV file with a header and a `close` column; `high` and `low` (both or
neither) are used by the American system, and `open`, `volume` and `date` are
read if they are there. Column names are matched without regard to case, so a
file saved from Yahoo Finance works as it is. Rows must be in chronological
order and prices strictly positive.

```csv
date,open,high,low,close,volume
2020-01-02,323.8,325.0,322.1,324.1,28000000
2020-01-03,325.2,327.1,324.8,326.9,31000000
```

`tfunify.data.load_csv` reads such a file into arrays and reports the line of
any value it cannot use.

## Examples

The scripts in [`examples/`](https://github.com/DiogoRibeiro7/tfunify/tree/develop/examples) run from any directory:

| Script | What it shows |
|---|---|
| `basic_usage.py` | the three systems on one simulated market, with a chart (`--plot`) |
| `performance_comparison.py` | Sharpe ratios in trending, mean-reverting and drifting markets, next to the closed form |
| `parameter_optimization.py` | a span chosen in sample, checked out of sample and against the closed form |
| `portfolio_integration.py` | a trend-following allocation (a "sleeve") held on top of a 60/40 portfolio |
| `real_data_analysis.py` | the three systems on a Yahoo Finance symbol or on your own CSV file |

## Relation to the paper

Definitions and equation numbers follow arXiv:2607.19497v1: Section 2 for the
filters and the volatility, Definition 4.1 for the European system, Appendix A
for the American and TSMOM systems, Sections 4 and 5 for the closed forms.

Where a finite sample forces a choice the paper leaves open, tfunify does this:

- The volatility estimate rescales the weights of the returns seen so far to
  sum to one, which makes it unbiased from the first day and uses no data from
  the future. `bias_correction=False` in `ewma_volatility_from_returns` gives the
  plain recursion started at the first squared return.
- The volatility estimate and the ATR start at the first return and the first
  true range that are not zero.
- Signal filters start from zero; price filters of the American system start at
  the first price.
- The American system uses the true range of the day for the entry of that day,
  stays flat on the day of an exit, and accrues profit as weight times simple
  return.
- TSMOM periods are counted from the first return, and the weight uses the
  volatility of the day before the rebalancing day, as in equation (A.16).

The authors' `trendfollowing` package makes other choices in places (log returns
and caps on signal and weight in its European runner, a close-to-close range in
its American runner), so the two will not agree to the last digit. The
[conventions page](https://diogoribeiro7.github.io/tfunify/systems/conventions/#differences-from-the-authors-package)
lists the differences.

## Development

```bash
git clone https://github.com/DiogoRibeiro7/tfunify.git
cd tfunify
python -m pip install -e ".[dev]"

python -m pytest --cov     # tests
ruff check . && ruff format --check .
mypy
```

To build the documentation: `pip install -e ".[docs]"`, then `mkdocs serve`.

Work happens on `develop`; `main` holds the released code. A release is made by
GitHub Actions when a commit reaches `main` whose version in `pyproject.toml`
has no tag yet: the workflow runs the tests, creates the tag and the GitHub
release with the notes from `CHANGELOG.md`, and publishes to PyPI if the
repository variable `PUBLISH_TO_PYPI` is `true`.
[DEVELOPMENT_WORKFLOW.md](https://github.com/DiogoRibeiro7/tfunify/blob/develop/DEVELOPMENT_WORKFLOW.md) has the details.

## Citation

```bibtex
@software{tfunify,
  author = {Ribeiro, Diogo},
  title = {tfunify: trend-following systems in NumPy},
  year = {2026},
  url = {https://github.com/DiogoRibeiro7/tfunify}
}

@article{sepp2026trend,
  author = {Sepp, Artur and Lucic, Vladimir},
  title = {The Science and Practice of Trend-Following Systems},
  year = {2026},
  eprint = {2607.19497},
  archivePrefix = {arXiv}
}
```

## License

MIT. See [LICENSE](https://github.com/DiogoRibeiro7/tfunify/blob/develop/LICENSE).
