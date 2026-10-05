# Reference

Everything in the tables below can be imported from `tfunify` directly, except where a module is named.

## Systems

| Name | |
|---|---|
| [`EuropeanTF`][tfunify.EuropeanTF], [`EuropeanTFConfig`][tfunify.EuropeanTFConfig], [`EuropeanTFResult`][tfunify.EuropeanTFResult] | continuous signal with volatility targeting |
| [`AmericanTF`][tfunify.AmericanTF], [`AmericanTFConfig`][tfunify.AmericanTFConfig], [`AmericanTFResult`][tfunify.AmericanTFResult] | breakouts with trailing stops |
| [`TSMOM`][tfunify.TSMOM], [`TSMOMConfig`][tfunify.TSMOMConfig], [`TSMOMResult`][tfunify.TSMOMResult] | time-series momentum |
| [`true_range`][tfunify.true_range], [`average_true_range`][tfunify.average_true_range] | the range measure of the American system |

## Filters and volatility

| Name | |
|---|---|
| [`span_to_nu`][tfunify.span_to_nu], [`nu_to_span`][tfunify.nu_to_span] | spans and smoothing parameters |
| [`ewma`][tfunify.ewma] | exponentially weighted moving average |
| [`ewma_variance_preserving`][tfunify.ewma_variance_preserving] | the single signal filter |
| [`long_short_variance_preserving`][tfunify.long_short_variance_preserving], [`long_short_loadings`][tfunify.long_short_loadings] | the long-short signal filter |
| [`pct_returns_from_prices`][tfunify.pct_returns_from_prices], [`log_returns_from_prices`][tfunify.log_returns_from_prices] | returns |
| [`ewma_volatility_from_returns`][tfunify.ewma_volatility_from_returns] | volatility estimate |
| [`vol_normalised_returns`][tfunify.vol_normalised_returns] | normalised returns |
| [`volatility_target_weights`][tfunify.volatility_target_weights] | volatility-target weight |
| [`volatility_weighted_turnover`][tfunify.volatility_weighted_turnover] | turnover |

## Closed forms: `tfunify.theory`

| Name | |
|---|---|
| [`european_expected_return`][tfunify.theory.european_expected_return] | expected annual return |
| [`european_volatility`][tfunify.theory.european_volatility] | annual volatility |
| [`european_sharpe_ratio`][tfunify.theory.european_sharpe_ratio] | Sharpe ratio |
| [`european_daily_moments`][tfunify.theory.european_daily_moments] | mean and variance of the daily return |
| [`expected_signal_turnover`][tfunify.theory.expected_signal_turnover] | turnover of the signal |
| [`acf_generating_function`][tfunify.theory.acf_generating_function], [`ar1_acf`][tfunify.theory.ar1_acf] | autocorrelation inputs |

## Statistics: `tfunify.metrics`

| Name | |
|---|---|
| [`performance_summary`][tfunify.metrics.performance_summary], [`PerformanceSummary`][tfunify.metrics.PerformanceSummary] | all figures at once |
| [`annualised_return`][tfunify.metrics.annualised_return], [`annualised_volatility`][tfunify.metrics.annualised_volatility], [`sharpe_ratio`][tfunify.metrics.sharpe_ratio], [`max_drawdown`][tfunify.metrics.max_drawdown] | one figure each |

## Data: `tfunify.data`

| Name | |
|---|---|
| [`load_csv`][tfunify.data.load_csv] | read a price file |
| [`download_csv`][tfunify.data.download_csv] | download prices from Yahoo Finance |
