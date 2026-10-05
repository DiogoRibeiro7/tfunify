# Conventions

What the three systems have in common, and the choices tfunify makes where the paper leaves one open.

## Timing

An array entry dated `t` holds what is known at the close of day `t`. A weight dated `t` earns the return of day `t + 1`:

```text
pnl[t] = weights[t - 1] * r[t]
```

Nothing dated `t` uses data from after `t`. The tests check this directly: running a system on the first `k` days of a series gives exactly the first `k` entries of the full run.

## Units

- `pnl` is a return per unit of capital. It is not compounded: the cumulative return is the running sum, as in the paper.
- `weights` are exposures per unit of capital. Futures need little margin, so weights above one are normal; an instrument with 5 % annual volatility needs a weight of three to reach a 15 % target.
- `volatility` is daily. Multiply by $\sqrt{a}$ to annualise.
- Annual figures use $a = 260$ trading days by default, the weekday count of the paper. Pass `a=252` if you prefer.

## Prices and returns

`run_from_prices` computes simple returns $s_t/s_{t-1}-1$ and returns arrays aligned with the prices. Index 0 has no return: its profit, weight and signal are zero and its volatility is NaN.

`run_from_returns` takes the returns themselves, and treats every entry as an observed return. Do not pass the placeholder zero that `pct_returns_from_prices` puts in front; pass `r[1:]`.

For log returns, compute them with `log_returns_from_prices(prices)[1:]` and call `run_from_returns`.

## Warm-up

The volatility estimate needs data before it means anything, and the first normalised returns would otherwise be divided by an estimate made from one or two observations. Until `warmup` returns have been observed (by default `span_sigma` rounded up, 33 days):

- `volatility` is NaN;
- returns are not normalised, so they do not enter the European signal, nor the TSMOM signal with `signs="period"`;
- weights are zero.

The first estimate is the one that includes return number `warmup`. It normalises the next return, and sizes the position of its own day. The TSMOM signal with daily signs needs no volatility and uses every return of its lookback from the start; only its weight waits.

The count starts at the first return that is not zero. Unchanged prices in front of a series are how a data vendor pads an instrument that started trading later than the others, and they say nothing about its volatility: counted as observations they would give an estimate of zero, and the first real return, divided by it, a position of any size. A padded series therefore gives the same results as the series without the padding. If all returns are zero there is no estimate at all, and no position. The average true range of the American system follows the same rule.

The rule is about prices that are exactly unchanged. A stretch of prices that hardly move is a market that has gone quiet, and is treated as one: see the [optional limits](#optional-limits).

The estimate itself gives the returns seen so far weights that sum to one:

$$
\sigma_t^2 = \frac{\sum_{k=0}^{t} \nu^k r_{t-k}^2}{\sum_{k=0}^{t} \nu^k} .
$$

This is unbiased from the first day under a constant variance and converges to the recursion of the paper. It uses no data from the future (seeding the recursion with the variance of the whole sample, a common shortcut, does).

Signal filters start from zero. The signal therefore builds up as evidence accumulates, and a filter with a span of 250 days needs a few hundred days to reach its full variance. If you compare systems, discard the first year or two.

The American system has no volatility estimate. It waits for its ATR (`atr_period` true ranges), and its price filters start at the first price.

## Optional limits

The paper's systems have no limit on the weight. Two options add one; both are off by default.

| Option | Systems | Effect |
|---|---|---|
| `sigma_floor_annual` | European, TSMOM | the volatility estimate is not allowed below this annualised value, for normalising returns and for sizing |
| `weight_cap` | all three | the absolute weight is not allowed above this value |

The case they exist for is an instrument that stops moving and then moves again. While it is quiet the volatility estimate decays towards zero, the weight grows as its reciprocal, and the first normal return meets that weight. An extreme example, which the test suite runs with filter spans of 20 and 4 days: daily moves of 1 % that shrink to 0.0001 % for 300 days and then return. The estimate falls to 0.00015 % a day, the weight reaches 5,600 times the capital, and the first normal day changes the capital by 56 times its size. With a floor of 5 % a year the largest weight in the same series is below 4.

```python
from tfunify import EuropeanTFConfig

cfg = EuropeanTFConfig(sigma_floor_annual=0.01, weight_cap=10.0)
```

Choose the floor for the instrument: well below its normal volatility, a fifth or a tenth of it, so that it binds only when something is wrong. A floor of 3 % a year, harmless for an equity index, would be in force all the time on a short-term interest rate future with a volatility of 2 %, and would hold that position at two thirds of its target.

## Input checks

Prices must be finite and strictly positive; returns must be finite. Missing values are not filled: clean the data first, so that the choice of how is yours. A wrong argument raises a `ValueError` that names it.

Strictly positive prices rule out back-adjusted futures series that pass through zero or below it, which have no returns. Compute the returns from the unadjusted contracts, roll by roll, and give them to `run_from_returns`. The American system works on price levels and needs a positive series, for example one adjusted by ratios instead of differences.

## Differences from the authors' package

tfunify follows the definitions of the paper. The runners of the authors' [trendfollowing](https://github.com/ArturSepp/TrendFollowingSystems) package depart from them in a few places, so the two will not agree to the last digit. As read from the source of that package in September 2026:

| | tfunify | trendfollowing runners |
|---|---|---|
| European returns | simple | log |
| Start of the volatility estimate | weights of the observed returns rescaled | recursion seeded with the variance of the whole sample |
| Warm-up | no weight for the first `warmup` returns, which do not enter the signal | all returns enter the signal; weights of the first 250 days are discarded |
| Caps | off unless set | European: signal at ±3 and weight at ±5; American: weight at ±10 |
| American range | true range of the day, averaged over `atr_period` days | mean absolute change of the close, as of the day before |
| TSMOM volatility in the weight | day before the rebalancing day | rebalancing day |

How much this matters differs by row. The first three change only how a series starts. Both volatility estimates follow the same recursion afterwards, so with log returns given to `run_from_returns`, and while no cap binds, the weights of the two European systems approach each other at the rate at which the slowest filter forgets: for LS(250, 20) the difference shrinks by a factor of $e$ every 125 days. Log and simple returns differ by half the squared return: 0.005 % for a move of 1 %. The caps change the result whenever they bind: the signal cap seldom, at three standard deviations, and the weight cap regularly on instruments with a low volatility. The American range is a different quantity: without the highs and the lows it is smaller, so entries come sooner and positions are larger.
