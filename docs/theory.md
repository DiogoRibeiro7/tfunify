# Closed forms

The paper's main result is that the performance of the European system can be read off the autocorrelation function of the returns it trades. `tfunify.theory` evaluates those formulas.

## Setting

The volatility-normalised returns $z_t$ are assumed stationary with unit variance, autocorrelations $\rho(m)$ and an annualised mean $\mu_{an}$, which is the Sharpe ratio of the instrument. Everything depends on the autocorrelations through one number per filter,

$$
\Psi_\nu = \sum_{m=1}^{\infty} \nu^m \rho(m) ,
$$

which is zero for white noise and $\nu\phi/(1-\nu\phi)$ for a first-order autoregressive process, AR(1), with coefficient $\phi$.

The functions take the autocorrelations from lag 1 on. An array that starts with the lag-0 value of one, as most estimators return it, is rejected rather than read one lag off.

## Expected return

For a single filter with $l = \sqrt{(1+\nu)/(1-\nu)}$, the expected annual return is

$$
\bar F_{1y} = l\,\sigma_{\mathrm{target}}\sqrt{a}\,\frac{1-\nu}{\nu}\,\Psi_\nu
\;+\; \frac{l\,\sigma_{\mathrm{target}}}{\sqrt{a}}\,\mu_{an}^2 .
$$

The first term is earned from autocorrelation and the second from drift. The drift enters squared: the system profits from a trend in either direction. This formula holds for any distribution of the returns.

## Volatility and Sharpe ratio

For Gaussian returns the daily variance is also closed-form, and with it the Sharpe ratio:

$$
SR = \frac{\sqrt{a}\,A_\nu + \mu_{an}^2/\sqrt{a}}
          {\sqrt{B_\nu + A_\nu^2 + (\mu_{an}^2/a)(1 + B_\nu + 2A_\nu)}},
$$

$$
A_\nu = \frac{1-\nu}{\nu}\,\Psi_\nu, \qquad
B_\nu = \frac{1-\nu}{1+\nu}\,(1 + 2\Psi_\nu) .
$$

The Sharpe ratio is the ratio of the moments of daily returns, and does not depend on the volatility target. The long-short filter has the same kind of formula with the moments of its signal (Corollary 5.4).

!!! note "What is not included"
    The paper adds a term for the excess kurtosis of the returns to the variance. It is not implemented: the functions give the Gaussian case. The term vanishes for serially independent returns. For an AR(1) coefficient of 0.05 and an excess kurtosis of 3, the paper reports that it lowers the Sharpe ratio by at most 0.2 % in relative terms, for spans from 5 to 250 days; with stronger autocorrelation or heavier tails it is larger.

## Using it

```python
from tfunify import theory

acf = theory.ar1_acf(0.05, 500)  # autocorrelations at lags 1..500

for span in (5, 21, 63, 250):
    print(span, round(theory.european_sharpe_ratio(span_long=span, acf=acf), 2))
# 5 0.6
# 21 0.34
# 63 0.2
# 250 0.1

print(round(theory.european_sharpe_ratio(span_long=250, instrument_sharpe=0.5), 2))  # 0.22
print(round(theory.european_expected_return(span_long=21, acf=acf), 3))  # 0.053
print(round(theory.european_volatility(span_long=21, acf=acf), 3))  # 0.157
```

Autocorrelation at short lags is worth more to a short filter, and drift is worth more to a long one. For a process with both, the two pull in opposite directions and the Sharpe ratio need not rise or fall steadily with the span; in the table of the [span example](examples.md#choosing-a-span) it falls up to a span of 125 days and then turns. The formula gives the whole curve without a backtest.

With estimates from data, pass the sample autocorrelations of the normalised returns as `acf`, with enough lags for $\sum \nu^m \rho(m)$ to converge at the slow span. Bear in mind that sample autocorrelations at hundreds of lags are noisy, and that the noise goes straight into the result.

## Turnover

The volatility-normalised turnover of a sequence of weights is

$$
U_t = \sqrt{a}\,\sigma_t\,\lvert w_t - w_{t-1}\rvert ,
$$

computed by `volatility_weighted_turnover`. Notional turnover is dominated by low-volatility contracts, whose weights are large; multiplying by the volatility makes trades comparable across contracts.

For Gaussian white noise the part of the turnover that comes from the signal has a closed form:

$$
\mathbb{E}\big[a\,\sigma_{\mathrm{target}}\,\lvert S_t - S_{t-1}\rvert\big]
= \frac{2a}{\sqrt{\pi}}\,\sigma_{\mathrm{target}}\sqrt{1-\nu}
$$

for the single filter, and the same with $\sqrt{\zeta}$, $\zeta = (1-\nu_1)(1-\nu_2)/(1+\nu_1\nu_2)$, in place of $\sqrt{1-\nu}$ for the long-short filter.

```python
from tfunify import theory

print(round(theory.expected_signal_turnover(span_long=250), 2))  # 3.93
print(round(theory.expected_signal_turnover(span_long=250, span_short=20), 2))  # 0.88
```

These are the 393 % and 88 % a year quoted in the paper, for a target of 15 %; the unit is that of the target, so 0.88 is 5.9 times the target.

They are a proxy for the turnover of the system: the signal alone, on Gaussian white noise. A system also trades every change of its volatility estimate. With a 33-day volatility span the paper puts the full turnover about 4 % above the proxy for a single filter, and at 1.6 to 2.3 times the proxy for the long-short filter, whose signal moves so little from day to day that the volatility updates dominate. Simulated with this package on white noise:

| | proxy | system |
|---|---|---|
| single filter, 250 days | 3.93 | 4.1 |
| long-short filter with spans 250 and 20, LS(250, 20) | 0.88 | 1.4 |

The long-short system trades about a third as much as the single one, not the quarter the proxy suggests. `volatility_weighted_turnover` measures the turnover of the weights a backtest produced.

## How closely does a backtest follow?

The closed forms describe the signal applied to returns normalised by their true volatility. A system has to estimate it, with two effects:

- The autocorrelation of the normalised returns is dampened a little, and the Sharpe ratio with it: by 2 % to 8 % in relative terms in the paper's simulations of an AR(1) process, and by about 3 % in the test suite's (filter span of 21 days, coefficient 0.1).
- A return divided by an estimate varies more than one divided by the true value. On white noise, with the default 33-day volatility span, the realised volatility is 3 % to 7 % above the target, more for long filters than for short ones, and more for a shorter volatility span.

A third thing is not an effect of the system at all: a Sharpe ratio measured on $n$ days has a standard error of about $\sqrt{a/n}$, which is 0.22 for twenty years. A single backtest cannot confirm or refute a closed-form value more closely than that. `examples/performance_comparison.py` averages over paths to make the comparison:

```text
market                               European  closed form       American          TSMOM
random walk                       0.00 (0.03)         0.00    0.04 (0.03)    0.05 (0.03)
trending, phi = 0.10              0.63 (0.03)         0.67    0.19 (0.03)    0.24 (0.03)
mean reverting, phi = -0.10      -0.66 (0.03)        -0.67   -0.13 (0.03)   -0.25 (0.03)
drift, Sharpe 1.0                 0.27 (0.03)         0.27    0.27 (0.03)    0.22 (0.03)
```

Mean Sharpe ratios of 60 simulated paths of 20 years, with standard errors; filters with a span of 21 days, one month.

Reference: [`tfunify.theory`](reference/theory.md).
