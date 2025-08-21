from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .core import (
    ewma_variance_preserving,
    ewma_volatility_from_returns,
    long_short_variance_preserving,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
)

FloatArray = NDArray[np.floating]


@dataclass
class EuropeanTFConfig:
    """Configuration for European trend-following system."""

    sigma_target_annual: float = 0.15
    a: int = 260
    span_sigma: int = 33
    mode: str = "longshort"  # "single" or "longshort"
    span_long: int = 250
    span_short: int = 20

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sigma_target_annual <= 0:
            raise ValueError("sigma_target_annual must be positive")
        if self.a <= 0:
            raise ValueError("a (trading days per year) must be positive")
        if self.span_sigma < 1:
            raise ValueError("span_sigma must be >= 1")
        if self.span_long < 1:
            raise ValueError("span_long must be >= 1")
        if self.span_short < 1:
            raise ValueError("span_short must be >= 1")
        if self.mode not in ("single", "longshort"):
            raise ValueError("mode must be 'single' or 'longshort'")
        if self.mode == "longshort" and self.span_short >= self.span_long:
            raise ValueError("span_short must be less than span_long in longshort mode")


class EuropeanTF:
    """
    European TF: variance-preserving EWMA on z_t with vol targeting.

    This system applies exponentially weighted moving averages to volatility-normalized
    returns, with optional long-short filtering and volatility targeting.

    Parameters
    ----------
    cfg : EuropeanTFConfig
        Configuration object with system parameters

    Examples
    --------
    >>> import numpy as np
    >>> from tfunify.european import EuropeanTF, EuropeanTFConfig
    >>>
    >>> # Generate sample price data
    >>> np.random.seed(0)
    >>> n = 1000
    >>> returns = 0.0001 + 0.02 * np.random.randn(n)
    >>> prices = 100 * np.cumprod(1 + np.r_[0.0, returns[1:]])
    >>>
    >>> # Configure and run system
    >>> cfg = EuropeanTFConfig(
    ...     sigma_target_annual=0.15,
    ...     span_sigma=33,
    ...     mode="longshort",
    ...     span_long=250,
    ...     span_short=20
    ... )
    >>> system = EuropeanTF(cfg)
    >>> pnl, weights, signal, volatility = system.run_from_prices(prices)
    """

    def __init__(self, cfg: EuropeanTFConfig) -> None:
        self.cfg = cfg

    def run_from_prices(
        self, prices: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Run the European TF system from price data.

        Parameters
        ----------
        prices : FloatArray
            Price time series

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray, FloatArray]
            - pnl: Daily P&L
            - weights: Position weights
            - signal: Trend signal
            - volatility: Volatility estimates
        """
        r = pct_returns_from_prices(prices)
        return self.run_from_returns(r)

    def run_from_returns(
        self, r: FloatArray
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """
        Run the European TF system from return data.

        Parameters
        ----------
        r : FloatArray
            Return time series

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray, FloatArray]
            - pnl: Daily P&L
            - weights: Position weights
            - signal: Trend signal
            - volatility: Volatility estimates
        """
        if len(r) == 0:
            raise ValueError("Returns array cannot be empty")

        cfg = self.cfg
        nu_sigma = span_to_nu(cfg.span_sigma)
        sigma = ewma_volatility_from_returns(r, nu_sigma)
        z = vol_normalised_returns(r, sigma)

        if cfg.mode == "single":
            s = ewma_variance_preserving(z, span_to_nu(cfg.span_long))
        else:
            s = long_short_variance_preserving(
                z, span_to_nu(cfg.span_long), span_to_nu(cfg.span_short)
            )

        v = volatility_target_weights(sigma, cfg.sigma_target_annual, cfg.a)
        w = s * v
        f = np.zeros_like(r)
        f[1:] = w[:-1] * r[1:]
        return f, w, s, sigma
