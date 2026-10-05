# Time-series momentum

Positions from the signs of past returns, rebalanced on a grid. The paper generalises the construction of Moskowitz, Ooi and Pedersen (2012) so that the signal has unit variance, like the European one (Definition A.4).

## Definition

Fix a period of $L$ days and a lookback of $M$ periods. On the rebalancing days $t'$, the last day of every period:

$$
S_{t'} = \frac{1}{\sqrt{ML}} \sum_{k=0}^{ML-1} \operatorname{sgn}(r_{t'-k}),
\qquad
w_{t'} = \frac{\sigma_{\mathrm{target}}}{\sqrt{a}\,\sigma_{t'-1}}\,S_{t'} .
$$

The weight is held until the next rebalancing day and earns from day $t'+1$.

The sum holds $ML$ signs. When the returns are serially independent and as likely to be positive as negative (a median of zero, and no return of exactly zero), the signs are independent with unit variance. $S$ then has unit variance and the position runs close to its volatility target whatever $L$ and $M$ are (about 3 % above it with the default volatility span, for the reason given for the [European system](european.md#definition)). A drift gives the signs a mean, and unchanged prices make some of them zero; the variance of $S$ is then below one. The signal depends on $L$ and $M$ only through their product; $L$ sets how often the position changes.

## One sign per period

With `signs="period"` the signal takes one sign per period, the sign of the sum of its normalised returns:

$$
S_{t'} = \frac{1}{\sqrt{M}} \sum_{m=0}^{M-1} \operatorname{sgn}\Big(\sum_{k=0}^{L-1} z_{t'-mL-k}\Big),
\qquad z_t = \frac{r_t}{\sigma_{t-1}} .
$$

This is the construction of tfunify 0.1, kept as an option. It is not the construction of Moskowitz, Ooi and Pedersen, who take the sign of the return over the trailing twelve months; here each period contributes the sign of its own return. It has unit variance under the same condition as the daily signs. The two variants differ in what they respond to: a period with nine small gains and one large loss counts as $+8$ with daily signs and as $-1$ with a period sign.

## Usage

```python
import numpy as np
from tfunify import TSMOM, TSMOMConfig

rng = np.random.default_rng(2)
prices = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(3000)))

result = TSMOM(TSMOMConfig(L=10, M=10)).run_from_prices(prices)
monthly = TSMOM(TSMOMConfig(L=21, M=12, signs="period")).run_from_prices(prices)

print(np.flatnonzero(result.rebalance)[:3])  # [100 110 120]
```

| Parameter | Default | Meaning |
|---|---|---|
| `sigma_target_annual` | 0.15 | annual volatility target of the position |
| `a` | 260 | trading days per year |
| `span_sigma` | 33 | span of the volatility estimate, in days |
| `L` | 10 | days in a period |
| `M` | 10 | periods in the lookback |
| `signs` | `"daily"` | `"daily"` (the paper) or `"period"` |
| `warmup` | `span_sigma` | returns observed before the volatility estimate is used |
| `sigma_floor_annual` | 0 | optional floor under the annualised volatility estimate |
| `weight_cap` | none | optional bound on the absolute weight |

$L = M = 10$ is where the paper finds the best results on its grid.

The result has `pnl`, `weights`, `signal` (the signal in force on each day), `volatility` and `rebalance`.

## Details of this implementation

- Periods are counted from the first return: with prices indexed from zero, the rebalancing days are the multiples of $L$, starting at $ML$.
- The weight uses the volatility of the day before the rebalancing day, as the definition is written.
- A signal of zero closes the position. With an even number of signs the sum can be exactly zero: for the default lookback of 100 days, on about 8 % of the rebalancing days.
- The series must hold at least $ML$ returns, that is, $ML + 1$ prices.
- The weight is zero until the volatility of the day before the rebalancing day exists, which takes `warmup` returns. The signal with daily signs does not wait: it needs no volatility and counts every return of its lookback.
- With `signs="period"` the first signal waits until every return of the lookback can be normalised, that is, until the lookback lies after the warm-up.
- A period whose normalised returns cancel has no sign: a sum that differs from zero only by rounding counts as zero.

Reference: [`TSMOM`](../reference/systems.md#tfunify.TSMOM).
