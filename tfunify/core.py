from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]


def span_to_nu(span: int) -> float:
    """Convert EWMA span to ν using ν = 1 - 2/(span+1)."""
    if not isinstance(span, int) or span < 1:
        raise ValueError("span must be an integer >= 1.")
    
    nu = 1.0 - 2.0 / (span + 1.0)
    
    # Handle edge case where span=1 gives nu=0
    if nu <= 0.0:
        raise ValueError(f"span={span} results in invalid nu={nu}. Use span >= 2.")
    
    return nu


def ewma(x: FloatArray, nu: float, *, x0: float | None = None) -> FloatArray:
    """
    EWMA recursion y_t = (1-ν)x_t + ν y_{t-1}.

    Parameters
    ----------
    x : FloatArray
        Input array
    nu : float
        Smoothing parameter in (0,1)
    x0 : float, optional
        Initial value. If None, uses x[0]

    Returns
    -------
    FloatArray
        EWMA of input series

    Raises
    ------
    ValueError
        If x is empty, not 1-D, contains non-finite values, or nu not in (0,1)
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be 1-D.")
    if x.size == 0:
        raise ValueError("Input array cannot be empty.")
    if np.any(~np.isfinite(x)):
        raise ValueError("Input contains non-finite values.")
    if not (0.0 < nu < 1.0):
        raise ValueError("nu must be in (0,1).")

    y = np.empty_like(x)
    y[0] = x[0] if x0 is None else float(x0)
    one_minus = 1.0 - nu
    for t in range(1, x.size):
        y[t] = one_minus * x[t] + nu * y[t - 1]
    return y


def ewma_variance_preserving(x: FloatArray, nu: float) -> FloatArray:
    """
    Variance-preserving EWMA: sqrt((1+ν)/(1-ν)) * EWMA(x).

    Parameters
    ----------
    x : FloatArray
        Input array
    nu : float
        Smoothing parameter in (0,1)

    Returns
    -------
    FloatArray
        Variance-preserving EWMA
    """
    if not (0.0 < nu < 1.0):
        raise ValueError("nu must be in (0,1).")
    alpha = math.sqrt((1.0 + nu) / (1.0 - nu))
    return alpha * ewma(x, nu)


def long_short_variance_preserving(x: FloatArray, nu_long: float, nu_short: float) -> FloatArray:
    """
    Variance-preserving long–short EWMA with correct loadings.

    Parameters
    ----------
    x : FloatArray
        Input array
    nu_long : float
        Long smoothing parameter
    nu_short : float
        Short smoothing parameter

    Returns
    -------
    FloatArray
        Long-short variance-preserving EWMA

    Raises
    ------
    ValueError
        If parameters don't satisfy 0 < nu_short < nu_long < 1
    """
    if not (0.0 < nu_short < nu_long < 1.0):
        raise ValueError("Require 0 < nu_short < nu_long < 1.")
    q = math.sqrt(
        1.0 / (1.0 - nu_long * nu_long)
        + 1.0 / (1.0 - nu_short * nu_short)
        - 2.0 / (1.0 - nu_long * nu_short)
    )
    ltilde1 = q / math.sqrt(1.0 - nu_long * nu_long)
    ltilde2 = q / math.sqrt(1.0 - nu_short * nu_short)
    return ltilde1 * ewma_variance_preserving(x, nu_long) - ltilde2 * ewma_variance_preserving(
        x, nu_short
    )


def pct_returns_from_prices(prices: FloatArray) -> FloatArray:
    """
    Relative returns r_t = s_t/s_{t-1} - 1, with r_0 = 0.

    Parameters
    ----------
    prices : FloatArray
        Price series

    Returns
    -------
    FloatArray
        Percentage returns

    Raises
    ------
    ValueError
        If prices are not 1-D, positive, or length >= 2
    """
    prices = np.asarray(prices, dtype=float)
    if prices.ndim != 1:
        raise ValueError("prices must be 1-D.")
    if prices.size < 2:
        raise ValueError("prices must have length >= 2.")
    if np.any(prices <= 0.0):
        raise ValueError("prices must be positive.")
    if np.any(~np.isfinite(prices)):
        raise ValueError("prices contain non-finite values.")

    r = np.empty_like(prices)
    r[0] = 0.0
    r[1:] = prices[1:] / prices[:-1] - 1.0
    return r


def ewma_volatility_from_returns(r: FloatArray, nu_sigma: float, eps: float = 1e-12) -> FloatArray:
    """
    Daily (non-annualised) sigma_t = sqrt(EWMA(r^2)).

    Parameters
    ----------
    r : FloatArray
        Return series
    nu_sigma : float
        Smoothing parameter for volatility
    eps : float, default=1e-12
        Minimum volatility floor

    Returns
    -------
    FloatArray
        Volatility estimates
    """
    if not (0.0 < nu_sigma < 1.0):
        raise ValueError("nu_sigma must be in (0,1).")
    if eps <= 0.0:
        raise ValueError("eps must be positive.")

    r = np.asarray(r, dtype=float)
    r2 = np.square(r)
    s2 = ewma(r2, nu_sigma)
    sigma = np.sqrt(np.maximum(s2, eps))
    
    # Ensure first value is exactly 0 for zero first return
    if len(r) > 0 and r[0] == 0.0:
        sigma[0] = eps  # Use minimum floor, not 0
    
    return sigma


def vol_normalised_returns(r: FloatArray, sigma: FloatArray) -> FloatArray:
    """
    z_t = r_t / sigma_{t-1}.

    Parameters
    ----------
    r : FloatArray
        Return series
    sigma : FloatArray
        Volatility series

    Returns
    -------
    FloatArray
        Vol-normalized returns

    Raises
    ------
    ValueError
        If r and sigma have different shapes
    """
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
    """
    v_t = sigma_target / (sqrt(a) * sigma_t).

    Parameters
    ----------
    sigma : FloatArray
        Volatility series
    sigma_target_annual : float
        Target annualized volatility
    a : int
        Trading days per year

    Returns
    -------
    FloatArray
        Volatility target weights

    Raises
    ------
    ValueError
        If parameters are invalid
    """
    if sigma_target_annual <= 0.0:
        raise ValueError("sigma_target_annual must be positive.")
    if a <= 0:
        raise ValueError("a must be positive.")

    sigma = np.asarray(sigma, dtype=float)
    return sigma_target_annual / (math.sqrt(a) * np.maximum(sigma, 1e-12))


def volatility_weighted_turnover(w: FloatArray, sigma: FloatArray, a: int) -> FloatArray:
    """
    U_t = sqrt(a) * sigma_t * |w_t - w_{t-1}|.

    Parameters
    ----------
    w : FloatArray
        Weight series
    sigma : FloatArray
        Volatility series
    a : int
        Trading days per year

    Returns
    -------
    FloatArray
        Volatility-weighted turnover

    Raises
    ------
    ValueError
        If w and sigma have different shapes or a is invalid
    """
    w = np.asarray(w, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if w.shape != sigma.shape:
        raise ValueError("w and sigma must have same shape.")
    if a <= 0:
        raise ValueError("a must be positive.")

    dw = np.zeros_like(w)
    dw[1:] = np.abs(w[1:] - w[:-1])
    return math.sqrt(a) * sigma * dw
