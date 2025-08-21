"""
tfunify: Unified trend-following systems in pure NumPy.

This package provides three main trend-following systems:
- European TF: Variance-preserving EWMA with volatility targeting
- American TF: Breakout system with ATR buffers and trailing stops
- TSMOM: Time series momentum with block-averaged signals

All systems include comprehensive input validation and are optimized for performance.
"""

from .core import (
    ewma,
    ewma_variance_preserving,
    ewma_volatility_from_returns,
    long_short_variance_preserving,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
    volatility_weighted_turnover,
)
from .american import AmericanTF, AmericanTFConfig
from .european import EuropeanTF, EuropeanTFConfig
from .tsmom import TSMOM, TSMOMConfig

__version__ = "0.1.0"
__author__ = "Diogo Ribeiro"
__email__ = "diogo.debastos.ribeiro@gmail.com"

__all__ = [
    # Core functions
    "span_to_nu",
    "ewma",
    "ewma_variance_preserving",
    "long_short_variance_preserving",
    "pct_returns_from_prices",
    "ewma_volatility_from_returns",
    "vol_normalised_returns",
    "volatility_target_weights",
    "volatility_weighted_turnover",
    # Trading systems
    "EuropeanTF",
    "EuropeanTFConfig",
    "AmericanTF",
    "AmericanTFConfig",
    "TSMOM",
    "TSMOMConfig",
]
