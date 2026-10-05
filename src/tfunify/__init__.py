"""tfunify: three trend-following systems in NumPy.

* `EuropeanTF`: a continuous signal from a variance-preserving EWMA
  filter of volatility-normalised returns, with volatility targeting.
* `AmericanTF`: breakouts of two price filters with an ATR buffer,
  positions of fixed size and trailing stops.
* `TSMOM`: time-series momentum, the normalised sum of the signs of
  past returns, rebalanced on a grid.

The definitions follow Sepp and Lucic, "The Science and Practice of
Trend-Following Systems" (arXiv:2607.19497). `tfunify.theory` evaluates
the paper's closed forms for the European system, `tfunify.metrics`
summarises the results and `tfunify.data` reads price files.
"""

from importlib.metadata import PackageNotFoundError, version

from . import data, metrics, theory
from .american import (
    AmericanTF,
    AmericanTFConfig,
    AmericanTFResult,
    average_true_range,
    true_range,
)
from .core import (
    ewma,
    ewma_variance_preserving,
    ewma_volatility_from_returns,
    log_returns_from_prices,
    long_short_loadings,
    long_short_variance_preserving,
    nu_to_span,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
    volatility_weighted_turnover,
)
from .european import EuropeanTF, EuropeanTFConfig, EuropeanTFResult
from .metrics import PerformanceSummary, performance_summary
from .tsmom import TSMOM, TSMOMConfig, TSMOMResult

try:
    __version__ = version("tfunify")
except PackageNotFoundError:  # pragma: no cover - a source tree that is not installed
    __version__ = "0+unknown"

__all__ = [
    "TSMOM",
    "AmericanTF",
    "AmericanTFConfig",
    "AmericanTFResult",
    "EuropeanTF",
    "EuropeanTFConfig",
    "EuropeanTFResult",
    "PerformanceSummary",
    "TSMOMConfig",
    "TSMOMResult",
    "__version__",
    "average_true_range",
    "data",
    "ewma",
    "ewma_variance_preserving",
    "ewma_volatility_from_returns",
    "log_returns_from_prices",
    "long_short_loadings",
    "long_short_variance_preserving",
    "metrics",
    "nu_to_span",
    "pct_returns_from_prices",
    "performance_summary",
    "span_to_nu",
    "theory",
    "true_range",
    "vol_normalised_returns",
    "volatility_target_weights",
    "volatility_weighted_turnover",
]
