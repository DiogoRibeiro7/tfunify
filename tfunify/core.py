from __future__ import annotations

import math
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def span_to_nu(span: int) -> float:
    """Convert EWMA span to ν using ν = 1 - 2/(span+1)."""
    if not isinstance(span, int) or span < 1:
        raise ValueError("span must be an integer >= 1.")
    return 1.0 - 2.0 / (span + 1.0)


def ewma(x: FloatArray, nu: float, *, x0: Optional[float] = None) -> FloatArray:
    """EWMA recursion y_t = (1-ν)x_t + ν y_{t-1}."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D.")
    if not (0.0 < nu < 1.0):
        raise ValueError("nu must be in (0,1).")
    y = np.empty_like(x)
    y[0] = x[0] if x0 is None else float(x0)
    one_minus = 1.0 - nu
    for t in range(1, x.size):
        y[t] = one_minus * x[t] + nu * y[t - 1]
    return y


def ewma_variance_preserving(x: FloatArray, nu: float) -> FloatArray:
    """Variance-preserving EWMA: sqrt((1+ν)/(1-ν)) * EWMA(x)."""
    alpha = math.sqrt((1.0 + nu) / (1.0 - nu))
    return alpha * ewma(x, nu)


def long_short_variance_preserving(x: FloatArray, nu_long: float, nu_short: float) -> FloatArray:
    """Variance-preserving long–short EWMA with correct loadings."""
    if not (0.0 < nu_short < nu_long < 1.0):
        raise ValueError("Require 0 < nu_short < nu_long < 1.")
    q = math.sqrt(
        1.0 / (1.0 - nu_long * nu_long)
        + 1.0 / (1.0 - nu_short * nu_short)
        - 2.0 / (1.0 - nu_long * nu_short)
    )
    ltilde1 = q / math.sqrt(1.0 - nu_long * nu_long)
    ltilde2 = q / math.sqrt(1.0 - nu_short * nu_short)
    return ltilde1 * ewma_variance_preserving(x, nu_long) - ltilde2 * ewma_variance_preserving(x, nu_short)


def pct_returns_from_prices(prices: FloatArray) -> FloatArray:
    """Relative returns r_t = s_t/s_{t-1} - 1, with r_0 = 0."""
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1 or prices.size < 2 or np.any(prices <= 0.0):
        raise ValueError("prices must be 1-D, positive, and length >= 2.")
    r = np.empty_like(prices)
    r[0] = 0.0
    r[1:] = prices[1:] / prices[:-1] - 1.0
    return r


def ewma_volatility_from_returns(r: FloatArray, nu_sigma: float, eps: float = 1e-12) -> FloatArray:
    """Daily (non-annualised) sigma_t = sqrt(EWMA(r^2))."""
    r2 = np.square(np.asarray(r, dtype=float))
    s2 = ewma(r2, nu_sigma)
    sigma = np.sqrt(np.maximum(s2, eps))
    return sigma


def vol_normalised_returns(r: FloatArray, sigma: FloatArray) -> FloatArray:
    """z_t = r_t / sigma_{t-1}."""
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if r.shape != sigma.shape:
        raise ValueError("r and sigma must have same shape.")
    z = np.empty_like(r)
    z[0] = 0.0
    z[1:] = r[1:] / np.where(sigma[:-1] > 0.0, sigma[:-1], np.nan)
    z[~np.isfinite(z)] = 0.0
    return z


def volatility_target_weights(sigma: FloatArray, sigma_target_annual: float, a: int) -> FloatArray:
    """v_t = sigma_target / (sqrt(a) * sigma_t)."""
    sigma = np.asarray(sigma, dtype=float)
    return sigma_target_annual / (math.sqrt(a) * np.maximum(sigma, 1e-12))


def volatility_weighted_turnover(w: FloatArray, sigma: FloatArray, a: int) -> FloatArray:
    """U_t = sqrt(a) * sigma_t * |w_t - w_{t-1}|."""
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if w.shape != sigma.shape:
        raise ValueError("w and sigma must have same shape.")
    dw = np.zeros_like(w)
    dw[1:] = np.abs(w[1:] - w[:-1])
    return math.sqrt(a) * sigma * dw
