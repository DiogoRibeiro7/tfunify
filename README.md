# tfunify

Unified trend-following systems in pure NumPy:

- **European TF**: variance-preserving EWMA on vol-normalised returns with volatility targeting.
- **American TF**: fast/slow price filters with ATR buffers + trailing stops.
- **TSMOM**: block-averaged sign of cumulative vol-normalised returns.

## Install

```bash
# with Poetry (recommended)
poetry add tfunify

# or pip
pip install tfunify
```

## Quickstart

```python
import numpy as np
from tfunify.european import EuropeanTF, EuropeanTFConfig

# toy price series
np.random.seed(0)
n = 1500
a = 260
r = 0.05/a + 0.01*np.random.randn(n)
prices = 100*np.cumprod(1 + np.r_[0.0, r[1:]])

cfg = EuropeanTFConfig(sigma_target_annual=0.15, a=a, span_sigma=33,
                       mode="longshort", span_long=250, span_short=20)
sys = EuropeanTF(cfg)
f, w, s, sigma = sys.run_from_prices(prices)
print(f"Annualised mean: {np.mean(f)*a:.3%}")
```

## CLI

```bash
# CSV must have a 'close' column; optional 'high','low'
tfu european --csv data.csv --target 0.15 --span-sigma 33 --span-long 250 --span-short 20
tfu american --csv data.csv --atr-period 33 --q 5 --p 5 --r-multiple 0.01
tfu tsmom    --csv data.csv --target 0.15 --span-sigma 33 --L 10 --M 10
```

Outputs simple summary stats to stdout and a '*_results.npz' with arrays.

## License

MIT License
