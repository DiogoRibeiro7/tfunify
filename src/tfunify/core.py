"""Filters, returns and volatility: the building blocks of the three systems.

The definitions follow Section 2 of Sepp and Lucic, "The Science and Practice of
Trend-Following Systems" (arXiv:2607.19497v1); equation numbers refer to that
version.

Time convention used throughout the package: an array indexed by `t` holds
what is known at the close of day `t`. Nothing dated `t` uses data from
after `t`, and a weight dated `t` earns the return of day `t + 1`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ._validation import (
    FloatArray,
    as_prices,
    as_real_array,
    as_series,
    check_integer,
    check_nu,
    check_positive,
    check_real,
    nu_from_span,
)

__all__ = [
    "FloatArray",
    "ewma",
    "ewma_variance_preserving",
    "ewma_volatility_from_returns",
    "log_returns_from_prices",
    "long_short_loadings",
    "long_short_variance_preserving",
    "nu_to_span",
    "pct_returns_from_prices",
    "span_to_nu",
    "vol_normalised_returns",
    "volatility_target_weights",
    "volatility_weighted_turnover",
]


def span_to_nu(span: float) -> float:
    r"""Smoothing parameter of an EWMA filter with the given span.

    $$
    \nu = 1 - \frac{2}{\mathrm{span} + 1} \qquad \text{(eq. 2.5)}
    $$

    Parameters
    ----------
    span : float
        Lookback of the filter in observations, at least 1. A span of 1 gives
        `nu = 0`, a filter that returns its input. Spans so large that `nu`
        would round to one (about 1e16) are rejected.

    Returns
    -------
    float
        `nu` in `[0, 1)`.
    """
    return nu_from_span(span)


def nu_to_span(nu: float) -> float:
    r"""Span of an EWMA filter with smoothing parameter `nu`: $(1+\nu)/(1-\nu)$."""
    nu = check_nu(nu)
    return (1.0 + nu) / (1.0 - nu)


def ewma(x: ArrayLike, nu: float, *, initial: float | None = None) -> FloatArray:
    r"""Exponentially weighted moving average.

    $$
    \mathcal{L}^{(\nu)}(y_t) = (1-\nu)\,y_t + \nu\,\mathcal{L}^{(\nu)}(y_{t-1})
    \qquad \text{(eq. 2.4)}
    $$

    Parameters
    ----------
    x : array_like
        One-dimensional sequence of finite values.
    nu : float
        Smoothing parameter, `0 <= nu < 1`. See `span_to_nu`.
    initial : float, optional
        Value of the filter before the first observation. By default the filter
        starts at the first observation, `out[0] = x[0]`, which is the natural
        choice for a level such as a price. With `initial=0.0` the output is
        the infinite sum of equation (2.4) applied to a sequence that is zero
        before the sample.

    Returns
    -------
    numpy.ndarray
        The filtered sequence, same length as `x`.
    """
    values = as_series(x, "x")
    nu = check_nu(nu)
    one_minus = 1.0 - nu
    samples = values.tolist()
    if initial is None:
        state = samples[0]
    else:
        state = one_minus * samples[0] + nu * check_real(initial, "initial")
    out = [state]
    append = out.append
    # A plain loop over Python floats: about ten times faster than indexing the
    # array, and exactly the recursion of the definition.
    for sample in samples[1:]:
        state = one_minus * sample + nu * state
        append(state)
    return np.asarray(out, dtype=np.float64)


def ewma_variance_preserving(x: ArrayLike, nu: float, *, initial: float | None = 0.0) -> FloatArray:
    r"""EWMA rescaled so that white noise keeps its variance.

    $$
    \widetilde{\mathcal{L}}^{(\nu)}(y_t) =
    \sqrt{\frac{1+\nu}{1-\nu}}\;\mathcal{L}^{(\nu)}(y_t) \qquad \text{(eq. 2.7)}
    $$

    For serially independent `y` with variance $\vartheta_0$ the plain
    filter has variance $\vartheta_0 (1-\nu)/(1+\nu)$; the factor undoes
    that (Proposition 2.4).

    Parameters
    ----------
    x : array_like
        One-dimensional sequence of finite values.
    nu : float
        Smoothing parameter, `0 <= nu < 1`.
    initial : float or None, default 0.0
        State of the plain filter before the first observation. The default
        starts from zero, so the variance of the output rises to
        $\vartheta_0$ from below, as
        $\vartheta_0\,(1-\nu^{2(t+1)})$. `None` starts at the first
        observation, whose weight is then $\sqrt{(1+\nu)/(1-\nu)}$.

    Returns
    -------
    numpy.ndarray
        The filtered sequence, same length as `x`.
    """
    nu = check_nu(nu)
    return np.asarray(
        math.sqrt((1.0 + nu) / (1.0 - nu)) * ewma(x, nu, initial=initial), dtype=np.float64
    )


def long_short_loadings(nu_long: float, nu_short: float) -> tuple[float, float]:
    r"""Loadings of the two plain EWMA filters in the long-short filter.

    $$
    l_1 = \frac{q}{1-\nu_1}, \qquad l_2 = \frac{q}{1-\nu_2},
    $$

    $$
    q = \left(\frac{1}{1-\nu_1^2} + \frac{1}{1-\nu_2^2}
    - \frac{2}{1-\nu_1\nu_2}\right)^{-1/2} \qquad \text{(eq. 2.10)}
    $$

    The bracket equals
    $(\nu_1-\nu_2)^2 (1+\nu_1\nu_2) / \big((1-\nu_1^2)(1-\nu_2^2)(1-\nu_1\nu_2)\big)$;
    that form is evaluated here because the three terms of the bracket nearly
    cancel when the two spans are close.

    Parameters
    ----------
    nu_long, nu_short : float
        Smoothing parameters of the slow and of the fast filter,
        `0 <= nu_short < nu_long < 1`.

    Returns
    -------
    tuple of float
        `(l_1, l_2)`, the loadings of the slow and of the fast filter.
    """
    nu_long, nu_short = _check_long_short(nu_long, nu_short)
    q = _long_short_scale(nu_long, nu_short) / (nu_long - nu_short)
    return q / (1.0 - nu_long), q / (1.0 - nu_short)


def long_short_variance_preserving(
    x: ArrayLike, nu_long: float, nu_short: float, *, initial: float | None = 0.0
) -> FloatArray:
    r"""Difference of a slow and a fast EWMA, scaled to keep the variance of white noise.

    $$
    \widetilde{\mathcal{LS}}^{(\nu_1,\nu_2)}(y_t) =
    l_1\,\mathcal{L}^{(\nu_1)}(y_t) - l_2\,\mathcal{L}^{(\nu_2)}(y_t)
    \qquad \text{(eq. 2.9)}
    $$

    with the loadings of `long_short_loadings`. For serially independent
    `y` the output has the variance of `y` (Proposition 2.6).

    Both loadings times their $(1-\nu)$ equal $q$, so the weight of
    the observation `k` days back is $q\,(\nu_1^k - \nu_2^k)$: zero for
    the latest observation, rising to a peak and decaying at the slow rate. One
    day's return does not move the output on that day, which is why this filter
    trades much less than a single filter of the same slow span.

    The output is computed as the slow filter of the lagged fast filter,
    $S_t = \nu_1 S_{t-1} + (1-\nu_1)(l_1 - l_2)\,\mathcal{L}^{(\nu_2)}(y_{t-1})$,
    which is the same sequence without the subtraction of two large terms.

    Parameters
    ----------
    x : array_like
        One-dimensional sequence of finite values.
    nu_long, nu_short : float
        Smoothing parameters, `0 <= nu_short < nu_long < 1`.
    initial : float or None, default 0.0
        State of both plain filters before the first observation; see
        `ewma_variance_preserving`.

    Returns
    -------
    numpy.ndarray
        The filtered sequence, same length as `x`.
    """
    nu_long, nu_short = _check_long_short(nu_long, nu_short)
    values = as_series(x, "x")
    # l_1 - l_2 = q (nu_1 - nu_2) / ((1 - nu_1)(1 - nu_2)), without the division by
    # the difference of the two parameters
    gain = _long_short_scale(nu_long, nu_short) / ((1.0 - nu_long) * (1.0 - nu_short))
    fast = ewma(values, nu_short, initial=initial)
    if initial is None:
        out = np.empty_like(values)
        out[0] = gain * values[0]
        if values.size > 1:
            out[1:] = ewma(gain * fast[:-1], nu_long, initial=float(out[0]))
        return out
    start = check_real(initial, "initial")
    lagged = np.concatenate(([start], fast[:-1]))
    return ewma(gain * lagged, nu_long, initial=gain * start)


def _check_long_short(nu_long: float, nu_short: float) -> tuple[float, float]:
    nu_long = check_nu(nu_long, "nu_long")
    nu_short = check_nu(nu_short, "nu_short")
    if not nu_short < nu_long:
        raise ValueError(
            f"nu_short must be smaller than nu_long, got nu_short={nu_short!r}, nu_long={nu_long!r}"
        )
    return nu_long, nu_short


def _long_short_scale(nu_long: float, nu_short: float) -> float:
    r"""$q\,(\nu_1 - \nu_2)$, which stays finite and accurate as the spans approach.

    $\sqrt{(1-\nu_1^2)(1-\nu_2^2)(1-\nu_1\nu_2)/(1+\nu_1\nu_2)}$, with every
    factor written in terms of $1-\nu_1$ and $1-\nu_2$: for a long span
    $1 - \nu^2$ computed as it stands would lose the digits that $\nu$
    shares with one.
    """
    gap_long, gap_short = 1.0 - nu_long, 1.0 - nu_short
    product = nu_long * nu_short
    return math.sqrt(
        gap_long * (1.0 + nu_long)
        * gap_short * (1.0 + nu_short)
        * (gap_long + nu_long * gap_short)  # 1 - nu_1 nu_2
        / (1.0 + product)
    )  # fmt: skip


def pct_returns_from_prices(prices: ArrayLike) -> FloatArray:
    r"""Simple returns $r_t = s_t/s_{t-1} - 1$ (eq. 2.2), with `r[0] = 0`.

    The first entry is a placeholder that keeps the result aligned with
    `prices`: there is no return on the first day. The systems drop it; if
    you pass returns to `run_from_returns` yourself, pass `r[1:]`.

    Parameters
    ----------
    prices : array_like
        At least two finite, strictly positive prices.

    Returns
    -------
    numpy.ndarray
        Returns, same length as `prices`.
    """
    values = as_prices(prices, min_length=2)
    out = np.zeros_like(values)
    # (s_t - s_{t-1}) / s_{t-1}: the difference of two nearby prices is exact, so
    # small returns keep their precision, which s_t / s_{t-1} - 1 would lose
    out[1:] = np.diff(values) / values[:-1]
    return out


def log_returns_from_prices(prices: ArrayLike) -> FloatArray:
    r"""Log returns $\log(s_t/s_{t-1})$, with `r[0] = 0`.

    See `pct_returns_from_prices` for the placeholder in the first entry.
    """
    values = as_prices(prices, min_length=2)
    out = np.zeros_like(values)
    # log(1 + r) from the simple return, for the reason given there: the
    # difference of two nearby logarithms keeps few digits of a small return
    out[1:] = np.log1p(np.diff(values) / values[:-1])
    return out


def ewma_volatility_from_returns(
    r: ArrayLike,
    nu_sigma: float,
    *,
    min_periods: int = 1,
    bias_correction: bool = True,
) -> FloatArray:
    r"""Daily (not annualised) volatility: root of an EWMA of squared returns.

    $$
    \sigma_t = \sqrt{\mathcal{L}^{(\nu_\sigma)}(r_t^2)} \qquad \text{(eq. 2.13)}
    $$

    No mean is subtracted. `sigma[t]` uses the returns up to and including
    day `t`.

    The estimate starts at the first return that is not zero. Zero returns
    before it, as in a series padded with its first price, say nothing about
    the volatility and are not counted as observations.

    Parameters
    ----------
    r : array_like
        Daily returns. Every entry is treated as an observed return: do not
        include the placeholder of `pct_returns_from_prices`.
    nu_sigma : float
        Smoothing parameter of the variance filter, `0 <= nu_sigma < 1`.
    min_periods : int, default 1
        Number of returns required for an estimate, counted from the first
        return that is not zero. Earlier entries are NaN.
    bias_correction : bool, default True
        Equation (2.13) is an infinite sum; a finite sample has to start it
        somewhere. With `True` the weights of the returns observed so far are
        rescaled to sum to one,

        $$
        \sigma_t^2 = \frac{\sum_{k=0}^{t} \nu^k r_{t-k}^2}{\sum_{k=0}^{t} \nu^k},
        $$

        which is unbiased from the first observation under a constant variance
        and converges to the infinite sum. With `False` the recursion starts
        at the first squared return, which then carries the weight
        $\nu^t$ instead of $(1-\nu)\nu^t$.

    Returns
    -------
    numpy.ndarray
        Volatility estimates, same length as `r`; NaN where fewer than
        `min_periods` returns were available, and everywhere if all returns
        are zero.
    """
    returns = as_series(r, "r")
    nu = check_nu(nu_sigma, "nu_sigma")
    min_periods = check_integer(min_periods, "min_periods")
    sigma = np.full(returns.size, np.nan)
    moved = np.flatnonzero(returns)
    if moved.size == 0:
        return sigma
    first = int(moved[0])
    sizes = np.abs(returns[first:]).tolist()
    # The recursion is run on the root: s_t = hypot(sqrt(nu) s_{t-1}, c |r_t|) is
    # s_t**2 = nu s_{t-1}**2 + c**2 r_t**2 without ever squaring a return, so a
    # return of 1e-170 or of 1e170 is neither lost nor infinite, and each value
    # depends on the returns up to its own day and on nothing else.
    decay = math.sqrt(nu)
    # with the correction s**2 is the plain sum of nu**k r**2, normalised below
    gain = 1.0 if bias_correction else math.sqrt(1.0 - nu)
    state = sizes[0]
    roots = [state]
    append = roots.append
    hypot = math.hypot
    for size in sizes[1:]:
        state = hypot(decay * state, gain * size)
        append(state)
    estimate = np.asarray(roots, dtype=np.float64)
    if bias_correction and nu > 0.0:
        # The weights nu**k of the k + 1 returns seen so far sum to
        # (1 - nu**(k+1)) / (1 - nu); expm1 keeps both differences accurate when
        # nu is close to one, and the ratio is exactly one for the first return.
        counts = np.arange(1, estimate.size + 1, dtype=np.float64)
        shortfall = np.expm1(counts * math.log(nu))  # nu**(k+1) - 1
        estimate = estimate * np.sqrt(shortfall[0] / shortfall)
    estimate[: min_periods - 1] = np.nan
    sigma[first:] = estimate
    return sigma


def vol_normalised_returns(r: ArrayLike, sigma: ArrayLike) -> FloatArray:
    r"""Returns divided by the volatility estimate of the previous day.

    $$
    z_t = \frac{r_t}{\sigma_{t-1}} \qquad \text{(eq. 2.14)}
    $$

    The lag makes `z` point in time: the return of day `t` never scales
    itself.

    Parameters
    ----------
    r : array_like
        Daily returns.
    sigma : array_like
        Daily volatility estimates aligned with `r`. NaN marks days without
        an estimate.

    Returns
    -------
    numpy.ndarray
        Normalised returns. `z[t]` is zero where `sigma[t-1]` is missing or
        zero, and for `t = 0`: a return that cannot be normalised carries no
        information for the signal.
    """
    returns = as_series(r, "r")
    vol = _as_volatility(sigma, "sigma")
    if returns.shape != vol.shape:
        raise ValueError(
            f"r and sigma must have the same length, got {returns.size} and {vol.size}"
        )
    z = np.zeros_like(returns)
    lagged = vol[:-1]
    usable = np.isfinite(lagged) & (lagged > 0.0)
    np.divide(returns[1:], lagged, out=z[1:], where=usable)
    return z


def volatility_target_weights(sigma: ArrayLike, sigma_target_annual: float, a: float) -> FloatArray:
    r"""Exposure that brings an asset with daily volatility `sigma` to the target.

    $$
    w^{vt}_t = \frac{\sigma_{\mathrm{target}}}{\sqrt{a}\,\sigma_t} \qquad \text{(eq. 4.2)}
    $$

    Parameters
    ----------
    sigma : array_like
        Daily volatility estimates. NaN marks days without an estimate.
    sigma_target_annual : float
        Annualised volatility target, for example `0.15`.
    a : float
        Annualisation factor: number of trading days per year.

    Returns
    -------
    numpy.ndarray
        Weights; zero where `sigma` is missing or zero (no estimate of the
        risk, no position).
    """
    vol = _as_volatility(sigma, "sigma")
    target = check_positive(sigma_target_annual, "sigma_target_annual")
    factor = check_positive(a, "a")
    weights = np.zeros_like(vol)
    usable = np.isfinite(vol) & (vol > 0.0)
    np.divide(target / math.sqrt(factor), vol, out=weights, where=usable)
    return weights


def volatility_weighted_turnover(w: ArrayLike, sigma: ArrayLike, a: float) -> FloatArray:
    r"""Volatility-normalised turnover of a sequence of weights.

    $$
    U_t = \sqrt{a}\,\sigma_t\,\lvert w_t - w_{t-1}\rvert \qquad \text{(eq. 4.15)}
    $$

    Notional turnover is dominated by low-volatility contracts, whose weights
    are large; multiplying by the volatility makes trades comparable across
    contracts.

    Parameters
    ----------
    w : array_like
        Weights.
    sigma : array_like
        Daily volatility estimates aligned with `w`. NaN is allowed on days
        without a trade.
    a : float
        Annualisation factor.

    Returns
    -------
    numpy.ndarray
        Turnover per day, with `U[0] = 0`. The annual turnover is `a` times
        the mean.
    """
    weights = as_series(w, "w")
    vol = _as_volatility(sigma, "sigma")
    if weights.shape != vol.shape:
        raise ValueError(
            f"w and sigma must have the same length, got {weights.size} and {vol.size}"
        )
    factor = check_positive(a, "a")
    trade = np.zeros_like(weights)
    trade[1:] = np.abs(np.diff(weights))
    traded = trade > 0.0
    if np.any(traded & ~np.isfinite(vol)):
        raise ValueError("sigma is missing on a day on which the weight changes")
    turnover = np.zeros_like(weights)
    np.multiply(math.sqrt(factor) * trade, vol, out=turnover, where=traded)
    return turnover


def _as_volatility(values: ArrayLike, name: str) -> FloatArray:
    """One-dimensional non-negative floats; NaN (no estimate) is allowed."""
    array = as_real_array(values, name)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if np.any(np.isinf(array)):
        raise ValueError(f"{name} contains infinite values")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must not be negative")
    return array
