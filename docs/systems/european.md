# European system

Continuous positions proportional to a trend signal, sized to a volatility target. This is the design of the large European managers, and the one the paper analyses in closed form (Definition 4.1).

## Definition

With normalised returns $z_t = r_t/\sigma_{t-1}$:

1. **Signal.** Either the variance-preserving filter

    $$
    S_t = \sqrt{\frac{1+\nu}{1-\nu}}\;\mathcal{L}^{(\nu)}(z_t)
    $$

    or the long-short filter

    $$
    S_t = l_1\,\mathcal{L}^{(\nu_1)}(z_t) - l_2\,\mathcal{L}^{(\nu_2)}(z_t),
    $$

    $$
    l_i = \frac{q}{1-\nu_i}, \qquad
    q = \left(\frac{1}{1-\nu_1^2} + \frac{1}{1-\nu_2^2} - \frac{2}{1-\nu_1\nu_2}\right)^{-1/2} .
    $$

2. **Weight.** $w_t = S_t\,\dfrac{\sigma_{\mathrm{target}}}{\sqrt{a}\,\sigma_t}$.

3. **Daily return.** $f_t = w_{t-1}\,r_t = \dfrac{\sigma_{\mathrm{target}}}{\sqrt{a}}\,S_{t-1}\,z_t$.

Both filters are scaled so that the signal has unit variance when the normalised returns are serially independent with unit variance. The position then runs close to its volatility target whatever the spans. In practice it runs a little above: the volatility is estimated, and a return divided by an estimate varies more than one divided by the true value. On simulated white noise, with the default 33-day volatility span, the realised volatility comes out 3 % to 7 % above the target. A shorter span makes a noisier estimate and a larger excess: 15 % to 20 % for a span of 10 days, against 1 % to 2 % for 100 days.

## The long-short filter

The two loadings are chosen so that $l_i(1-\nu_i) = q$ for both filters. The weight of the return $k$ days back is therefore

$$
q\,(\nu_1^k - \nu_2^k),
$$

which is zero for the latest return, rises to a peak and then decays at the slow rate. The return of a day does not move the signal on that day. This is why the long-short filter trades much less than a single filter with the same slow span. For Gaussian white noise the signal of the filter with spans 250 and 20, LS(250, 20), turns over 88 % of volatility-weighted exposure a year, against 393 % for a single 250-day filter. Those are the paper's figures for the signal alone; with the trades that the volatility estimate causes, the long-short system turns over about a third as much as the single one (see [turnover](../theory.md#turnover)).

## Usage

```python
import numpy as np
from tfunify import EuropeanTF, EuropeanTFConfig

rng = np.random.default_rng(0)
prices = 100 * np.cumprod(1 + 0.0003 + 0.01 * rng.standard_normal(5200))

long_short = EuropeanTF(EuropeanTFConfig(span_long=250, span_short=20)).run_from_prices(prices)
single = EuropeanTF(EuropeanTFConfig(mode="single", span_long=63)).run_from_prices(prices)
```

| Parameter | Default | Meaning |
|---|---|---|
| `sigma_target_annual` | 0.15 | annual volatility target of the position |
| `a` | 260 | trading days per year |
| `span_sigma` | 33 | span of the volatility estimate, in days |
| `mode` | `"longshort"` | `"longshort"` or `"single"` |
| `span_long` | 250 | span of the slow filter (the only one in `"single"` mode) |
| `span_short` | 20 | span of the fast filter |
| `warmup` | `span_sigma` | returns observed before the volatility estimate is used |
| `sigma_floor_annual` | 0 | optional floor under the annualised volatility estimate |
| `weight_cap` | none | optional bound on the absolute weight |

The spans of 250 and 20 days are the ones the paper settles on in its empirical section, 33 days is its baseline for the volatility, and 15 % is the target of its exhibits.

## Reading the result

- `signal` is dimensionless: +1 is one standard deviation of trend. `weights` is the signal times the volatility-target weight, so an instrument with 5 % annual volatility at a signal of one carries a weight of three for a 15 % target.
- When an instrument goes quiet, its volatility estimate falls and the weights grow. That is the definition. To bound them, set `sigma_floor_annual` or `weight_cap`; see [Conventions](conventions.md#optional-limits).
- The first `warmup` returns only feed the volatility estimate: no weight, and nothing in the signal. After that the signal builds up from zero; see [Conventions](conventions.md#warm-up).
- The [closed forms](../theory.md) give the expected return and the Sharpe ratio of this system from the autocorrelations of the returns.

Reference: [`EuropeanTF`](../reference/systems.md#tfunify.EuropeanTF).
