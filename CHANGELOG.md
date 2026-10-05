# Changelog

All notable changes to this project are documented in this file. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-05

The three systems now compute what the paper defines. Every numerical defect
below was reproduced on 0.1.3 and has a test. Results of all three systems
change; the American system and the long-short filter were wrong on every
input.

Definitions and equation numbers refer to Sepp and Lucic, *The Science and
Practice of Trend-Following Systems*, arXiv:2607.19497v1.

### Fixed

- **American system: always short.** The price filters were the
  variance-preserving ones, which multiply a filter by the square root of its
  span, so the slow filter was about 3.5 times the fast one and
  `fast < slow - q * ATR` held on every day: 0 % of days long and 99 % short on
  rising and on falling prices alike. The filters are now plain EWMAs of the
  close (eq. A.5).
- **American system: profit in mixed units.** The dimensionless weight
  `R * price / ATR` was multiplied by a price change. The profit is now the
  weight times the simple return, a return on capital like that of the other
  two systems.
- **Long-short filter: normalising constant inverted.** `q` was the square
  root of the bracket of equation (2.10) instead of the reciprocal of that
  root, which makes the output too large by the value of the bracket: 49 times
  for spans (250, 20), a standard deviation of 49 on unit white noise instead
  of 1.
- **European system: signal squashed.** The signal was passed through
  `tanh(s / 3)`, which is not part of the method. With the inflated long-short
  filter, the signal was beyond ±0.99 on 88 % of days, a binary position; with
  a single filter the system ran at a third of its volatility target (4.8 % for
  a target of 15 %). The signal is now the filter output.
- **Volatility clipped and floored.** The estimate was clipped to between
  0.05 % and 15 % a day, and the weights used `max(sigma, 0.5 %)`. An instrument
  with 0.3 % daily volatility (4.8 % a year) got 60 % of the exposure its target
  required. The estimate is no longer altered; see `sigma_floor_annual` and
  `weight_cap` below for explicit limits.
- **Volatility started on a placeholder.** The variance filter was started on
  the zero that stands for the missing first return, so the first normalised
  returns of a simulated series with 1 % daily volatility were -2.6, 12.8, 0.7
  and -3.4; the largest of the first five is typically 18. A 250-day filter
  carries such values for months. The estimate now starts on the first real
  return, is unbiased from the first day, and is not used until `warmup`
  returns have been seen.
- **TSMOM: a zero signal kept the old position.** The forward fill treated a
  weight of zero as "no rebalancing", so whenever the signs cancelled the
  previous weight stayed (with the default `M = 10`, on about a quarter of the
  rebalancing days).
- **`pct_returns_from_prices` returned log returns.** It now returns simple
  returns, as its name and the paper say; `log_returns_from_prices` is new.
- `ewma_variance_preserving` with `nu = 0` returned a list unchanged instead
  of an array, and raised on a tuple.
- `tfunify.__version__` was `"0.1.0"` while the package was 0.1.3. It is now
  read from the installed metadata.
- The command line printed its summary with the Greek letters μ and σ, which
  the default code page of Windows cannot encode when the output is redirected
  to a file. The summary is plain ASCII.
- `tfu european` without `--longshort` ran the single filter although the
  library default is the long-short one. The default is now the same,
  `--mode longshort`; `--mode single` selects the other.
- `download_csv` raised a `TypeError` with pandas 3 on the two levels of
  column names that current versions of yfinance return, and a `ValueError` on
  a missing volume. Both are handled, and rows with missing prices are skipped.
- The release workflow was not valid YAML, and no run of it ever produced a
  release.

### Changed

These changes are incompatible with 0.1.

- **TSMOM signal.** The default is now the definition of the paper (eq. A.16):
  the sum of the signs of the `M * L` daily returns of the lookback, divided by
  `sqrt(M * L)`. The construction of 0.1, one sign per period, is available as
  `TSMOMConfig(signs="period")`. Periods are counted from the first return.
- **Results are objects.** `run_from_prices`, `run_from_returns` and `run`
  return `EuropeanTFResult`, `TSMOMResult` and `AmericanTFResult` with named
  arrays. They still unpack as before: `pnl, weights, signal, volatility` for
  the European and TSMOM systems and `pnl, weights` for the American one (whose
  second array was called `units`).
- **TSMOM `signal`** holds the signal in force on every day; before, it was
  non-zero on rebalancing days only. `TSMOMResult.rebalance` marks those days.
- **Warm-up.** Until `warmup` returns have been observed (default:
  `span_sigma`, rounded up), counted from the first return that is not zero,
  the volatility is NaN and the weight is zero. The European signal and the
  TSMOM signal with `signs="period"` are zero as well; the TSMOM signal with
  daily signs does not use the volatility and counts every return of its
  lookback.
- **Where a series starts.** The volatility estimate and the ATR start at the
  first return and the first true range that are not zero. Unchanged prices in
  front of a series, the padding of an instrument that started trading later
  than the others, are not counted as observations of a volatility of zero, so
  a padded series gives the results of the series without the padding.
- **Prices must be strictly positive**, in all three systems and in
  `load_csv`. That excludes back-adjusted futures series that pass through
  zero: they have no returns. Compute the returns from the unadjusted contracts
  and use `run_from_returns`.
- **NaN is rejected, not propagated**: in prices and returns, and in the
  returns and weights given to `vol_normalised_returns` and
  `volatility_weighted_turnover`. NaN in a volatility array still means "no
  estimate".
- **TSMOM needs a full lookback**: at least `M * L` returns (`M * L + 1`
  prices); a shorter series is an error.
- **`run_from_returns`** treats every entry as an observed return. Do not pass
  the placeholder that `pct_returns_from_prices` puts in front.
- **Filters.** `ewma(x, nu, x0=...)` becomes `ewma(x, nu, initial=...)`, where
  `initial` is the state before the first observation (0.1 replaced the first
  output by `x0`). The variance-preserving filters start from zero by default.
- **`ewma_volatility_from_returns`** loses its `eps` argument and gains
  `min_periods` and `bias_correction`.
- **Configurations are immutable**, and they reject invalid values that 0.1
  accepted (booleans, non-finite numbers).
- **American system:** the system stays flat on the day of an exit, and giving
  only one of `high` and `low` is an error (0.1 silently ignored it).
- **Command line:** results files hold `pnl`, `weights`, `signal`,
  `volatility` (and so on) instead of `f`, `w`, `s`, `sgrid`, `sigma`, `units`.
- **`load_csv`** needs only a `close` column, matches column names without
  regard to case, returns the `date` column when there is one, and rejects ISO
  dates that are not strictly increasing, repeated dates included. A file with
  only one of `high` and `low`, or with one of the recognised columns twice, is
  an error.
- `tfunify.__author__` and `tfunify.__email__` are gone; the package metadata
  has them.
- Spans may be real numbers, not only integers.
- NumPy 1.24 or later is enough (0.1 required NumPy 2), and Python 3.13 and
  3.14 are supported.

### Added

- `tfunify.theory`: the paper's closed forms for the European system.
  `european_expected_return`, `european_volatility`, `european_sharpe_ratio`
  and `european_daily_moments` take the autocorrelations of the normalised
  returns and the Sharpe ratio of the instrument (Corollary 4.10,
  Proposition 5.3, Corollaries 5.4 and 5.5, for Gaussian returns);
  `expected_signal_turnover` gives the turnover of the signal (Propositions 4.6
  and 4.7). They reproduce the figures quoted in the paper: a turnover of 393 %
  for the single 250-day filter and 88 % for LS(250, 20), and a break-even cost
  (the cost per unit of volatility-normalised turnover that uses up the
  expected return) of 37 to 41 basis points for an AR(1) coefficient of 0.05.
- `tfunify.metrics`: `performance_summary`, `annualised_return`,
  `annualised_volatility`, `sharpe_ratio` and `max_drawdown`.
- `sigma_floor_annual` and `weight_cap` in the configurations: explicit,
  optional limits in place of the hidden ones. Both are off by default.
- `long_short_loadings`, `nu_to_span`, `log_returns_from_prices`, `true_range`
  and `average_true_range`.
- `AmericanTFResult` exposes the position, the stop, the ATR and the two
  filters.
- Command line: `--out`, `--mode`, `--signs`, `--version`, `download
  --adjusted`, and `python -m tfunify`. A command that fails on its input
  prints the reason to standard error and exits with status 1.
- Type information (`py.typed`).
- A documentation site built with MkDocs and published to GitHub Pages.

### Tests and infrastructure

- The tests of 0.1, 205 of them, all passed with the defects above in place:
  they checked shapes, finiteness and agreement of the code with itself. They
  are replaced by tests against independent references:
  the definitions evaluated with explicit sums, a hand-worked path and the
  rules of the American system checked day by day, the paper's special cases,
  direct evaluation of the moments, and simulation.
- Packaging moves from Poetry to a standard `pyproject.toml` built with
  Hatchling, in a `src` layout. The lock file is gone: a library does not pin
  its dependencies.
- CI runs on Python 3.10 to 3.14 on Linux, on macOS and Windows, and with the
  oldest supported NumPy; it type-checks with `mypy --strict`, builds the
  distribution and runs the command line from the installed wheel.
- Releases are made by GitHub Actions when a new version reaches `main`: the
  tag, the GitHub release with its notes from this file, and, once switched on,
  the upload to PyPI.
- The example scripts are rewritten, run from any directory and are tested.
- The Sphinx configuration, which did not build, is replaced by the MkDocs site.

## [0.1.3] - 2025-08-22

First version. It was never published to PyPI.

### Added

- European, American and TSMOM systems with their configuration classes.
- Core functions: `span_to_nu`, `ewma`, `ewma_variance_preserving`,
  `long_short_variance_preserving`, `pct_returns_from_prices`,
  `ewma_volatility_from_returns`, `vol_normalised_returns`,
  `volatility_target_weights` and `volatility_weighted_turnover`.
- The `tfu` command line with the `european`, `american`, `tsmom` and
  `download` commands.
- CSV loading and an optional Yahoo Finance downloader.
