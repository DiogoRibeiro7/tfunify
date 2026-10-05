# The three systems

The paper sorts trend-following systems into three families by how they turn prices into positions.

| | European | American | Time-series momentum |
|---|---|---|---|
| Signal | exponentially weighted moving average (EWMA) of volatility-normalised returns | fast EWMA of the price against a slow one, with a buffer | sum of the signs of past returns |
| Position | proportional to the signal | fixed when opened | proportional to the signal |
| Sizing | volatility target | risk multiple over the average true range (ATR) | volatility target |
| Changes | every day | at entries and exits | every `L` days |
| Exit | the signal changes sign | trailing stop | the signal changes sign |
| Class | [`EuropeanTF`](european.md) | [`AmericanTF`](american.md) | [`TSMOM`](tsmom.md) |

All three monetise the same thing, persistence in returns. In the paper's backtests on 84 futures the European and American systems with the same spans correlate at 95 %, and the American system has about half the turnover of the other two. They also differ in how much of the theory applies: the European signal is linear in past returns, which is what makes the [closed forms](../theory.md) possible.

## Shared building blocks

With daily prices $s_t$:

- **Return.** $r_t = s_t/s_{t-1} - 1$.
- **EWMA filter.** With $\nu = 1 - 2/(\mathrm{span}+1)$,

    $$
    \mathcal{L}^{(\nu)}(y_t) = (1-\nu)\,y_t + \nu\,\mathcal{L}^{(\nu)}(y_{t-1}) .
    $$

    For white noise the filter divides the variance by its span.
- **Volatility.** $\sigma_t = \sqrt{\mathcal{L}^{(\nu_\sigma)}(r_t^2)}$: daily, not annualised, with no mean subtracted. It starts at the first return that is not zero.
- **Normalised return.** $z_t = r_t/\sigma_{t-1}$. The lag keeps the return of a day from scaling itself.
- **Volatility target.** A weight of $\sigma_{\mathrm{target}}/(\sqrt{a}\,\sigma_t)$ brings a position to an annual volatility of $\sigma_{\mathrm{target}}$, with $a$ trading days in a year.

These are functions of their own, documented in the [reference](../reference/core.md): `ewma`, `ewma_volatility_from_returns`, `vol_normalised_returns`, `volatility_target_weights`.

[Conventions](conventions.md) covers what is common to the three systems: timing, units, the warm-up and the optional limits.
