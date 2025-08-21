from __future__ import annotations

import math
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .core import (
    span_to_nu, ewma_volatility_from_returns, vol_normalised_returns,
    volatility_target_weights, pct_returns_from_prices,
    ewma_variance_preserving, long_short_variance_preserving
)

FloatArray = NDArray[np.floating]


@dataclass
class EuropeanTFConfig:
    sigma_target_annual: float = 0.15
    a: int = 260
    span_sigma: int = 33
    mode: str = "longshort"      # "single" or "longshort"
    span_long: int = 250
    span_short: int = 20


class EuropeanTF:
    """European TF: variance-preserving EWMA on z_t with vol targeting."""

    def __init__(self, cfg: EuropeanTFConfig):
        self.cfg = cfg

    def run_from_prices(self, prices: FloatArray):
        r = pct_returns_from_prices(prices)
        return self.run_from_returns(r)

    def run_from_returns(self, r: FloatArray):
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
