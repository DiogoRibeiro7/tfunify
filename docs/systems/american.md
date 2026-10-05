# American system

Breakouts from a range, positions of fixed size and trailing stops: the design that descends from the turtle traders (Definition A.3 of the paper).

## Definition

Two EWMA filters of the close, slow and fast, and the average true range (ATR) over $N$ days:

$$
tr_t = \max\{\lvert s^{high}_t - s^{low}_t\rvert,\ \lvert s^{high}_t - s^{close}_{t-1}\rvert,\ \lvert s^{low}_t - s^{close}_{t-1}\rvert\},
$$

$$
ATR_t = \frac{1}{N}\sum_{n=0}^{N-1} tr_{t-n} .
$$

The first day has no previous close, so its true range is its high minus its low.

**Entry.** From a flat position:

- long when $\text{fast}_t > \text{slow}_t + q\,ATR_t$, with weight $R\,s_t/ATR_t$ and stop $s_t - p\,ATR_t$;
- short when $\text{fast}_t < \text{slow}_t - q\,ATR_t$, with weight $-R\,s_t/ATR_t$ and stop $s_t + p\,ATR_t$.

**Exit.** A long position is closed when the close is below the stop of the previous day *and* the long entry condition no longer holds. Otherwise the stop trails:

$$
\text{stop}_t = \max\{\text{stop}_{t-1},\ s_t - p\,ATR_t\} .
$$

Short positions mirror this with the minimum. The exit needs both conditions because a position stopped out while the entry signal is still on would be reopened the next day.

The weight is fixed when the position is opened. The distance from the entry price to the initial stop, as a return, is $p\,ATR_t/s_t$; times the weight, that is $R\,p$ of capital at risk in each trade, whatever the volatility of the instrument.

## Usage

```python
import numpy as np
from tfunify import AmericanTF, AmericanTFConfig

rng = np.random.default_rng(1)
close = 100 * np.exp(np.cumsum(0.0004 + 0.01 * rng.standard_normal(3000)))
high = close * (1 + 0.005 * rng.random(3000))
low = close * (1 - 0.005 * rng.random(3000))

result = AmericanTF(AmericanTFConfig(q=5.0, p=5.0)).run(close, high, low)

entries = np.flatnonzero((result.position[1:] != 0) & (result.position[:-1] == 0)) + 1
print(len(entries), "trades")
```

| Parameter | Default | Meaning |
|---|---|---|
| `span_long` | 250 | span of the slow filter of the close |
| `span_short` | 20 | span of the fast filter |
| `atr_period` | 33 | days in the average true range |
| `q` | 5.0 | entry buffer, in ATRs (the paper's $\omega$) |
| `p` | 5.0 | width of the trailing stop, in ATRs |
| `r_multiple` | 0.01 | risk multiple $R$ |
| `weight_cap` | none | optional bound on the weight of a new position |

The spans and $q = p = 5$ are the paper's choices. The paper sets $R$ so that the system matches the volatility of the other two and does not quote it; 0.01 is the default of the authors' code.

The result has `pnl`, `weights`, `position` (+1, 0, -1), `stop`, `atr`, `fast` and `slow`.

## Details of this implementation

- Everything is decided on closing prices. The stop is compared with the close, not with the low or the high of the day, and the position decided at the close of day $t$ earns the return of day $t+1$.
- The entry of day $t$ uses the ATR of day $t$.
- After an exit the system is flat for at least that day; a new position, in either direction, can open on the next day.
- The weight is an exposure per unit of capital, held constant over the trade and applied to the simple return of each day. That is a position rebalanced daily to constant exposure, and it is how the paper's backtest accrues profit. It is not the profit of a fixed number of contracts, which the paper's equation (A.14) states as the weight times the return from entry to exit: over a trade in which the price rises by half, the daily returns add up to about 41 %, not 50 %. For moves of a few per cent the two agree closely.
- The ATR starts at the first true range that is not zero, and needs `atr_period` of them: with highs and lows it is available from day `atr_period - 1`, counted from zero, and with closes only one day later, because the first day then has no range. Unchanged prices in front of a series are skipped.
- No position is opened before the ATR is available, or on a day on which the ATR is zero.
- The filters start at the first price. A slow filter needs a few multiples of its span to forget that starting point, so the first trades of a short history reflect it.

With the default buffer of five ATRs the system is flat a good part of the time. On a simulated random walk without drift it is flat on about four days in ten when it is given closes only, and on more than half of the days when it is given highs and lows, whose true range is wider.

Reference: [`AmericanTF`](../reference/systems.md#tfunify.AmericanTF).
