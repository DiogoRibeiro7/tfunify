---
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# Three trend-following systems, as the paper defines them

European, American and time-series momentum systems in NumPy, with the closed forms that say what a trend filter should earn from the autocorrelation and the drift of the returns.

[Get started](getting-started.md){ .md-button .md-button--primary }
[The systems](systems/index.md){ .md-button }

</div>

tfunify implements the three systems of Sepp and Lucic, *The Science and Practice of Trend-Following Systems* ([arXiv:2607.19497](https://arxiv.org/abs/2607.19497)). Each one takes a price series and returns the daily weights and the daily profit and loss of a position in that instrument. The only dependency is NumPy.

<div class="grid cards" markdown>

-   :material-sine-wave:{ .lg .middle } **Three systems, one interface**

    ---

    A continuous signal with volatility targeting, breakouts with trailing stops, and signs of past returns: configured with a dataclass, run with one call, returned as named arrays.

    [:octicons-arrow-right-24: The systems](systems/index.md)

-   :material-sigma:{ .lg .middle } **Closed forms**

    ---

    Expected return, volatility and Sharpe ratio of the European system from the autocorrelations and the drift of the returns, and the turnover of its signal, to set against a backtest.

    [:octicons-arrow-right-24: Closed forms](theory.md)

-   :material-check-decagram:{ .lg .middle } **Checked against the definitions**

    ---

    Every formula is tested against an independent evaluation of the paper's definition, and no output depends on data from the future.

    [:octicons-arrow-right-24: Conventions](systems/conventions.md)

</div>

## Install

Install it from GitHub:

```bash
pip install "tfunify @ git+https://github.com/DiogoRibeiro7/tfunify@main"
```

`@main` is the released code. [Getting started](getting-started.md#installation) has the other ways.

## In a few lines

```python
import numpy as np
from tfunify import EuropeanTF, EuropeanTFConfig, performance_summary

rng = np.random.default_rng(0)
prices = 100 * np.cumprod(1 + 0.0003 + 0.01 * rng.standard_normal(5200))

result = EuropeanTF(EuropeanTFConfig(span_long=250, span_short=20)).run_from_prices(prices)
print(performance_summary(result.pnl))
```

`result.pnl`, `result.weights`, `result.signal` and `result.volatility` are arrays aligned with the prices. [Getting started](getting-started.md) goes through the three systems.

This is an independent implementation. The authors' own package is [trendfollowing](https://github.com/ArturSepp/TrendFollowingSystems).
