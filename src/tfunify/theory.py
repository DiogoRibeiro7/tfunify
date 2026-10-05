"""Closed-form results for the European system.

Sections 4 and 5 of Sepp and Lucic (arXiv:2607.19497v1) express the expected
return, the variance and the Sharpe ratio of the European system through the
autocorrelation function of the volatility-normalised returns `z` and their
drift. The functions here evaluate those formulas, so that a backtest can be
compared with what the autocorrelation of the data implies.

Assumptions, as in the paper: `z` is stationary with unit variance and
autocorrelations `rho(m)`; its annualised mean is the Sharpe ratio of the
instrument. The expected return holds for any such process. The variance, and
therefore the volatility and the Sharpe ratio, is given for Gaussian `z`. For
serially dependent returns that are heavy-tailed or skewed the variance
differs: the paper's term for excess kurtosis (the one with K in
Proposition 5.3) is not included, and the paper itself assumes a zero third
moment. For serially independent returns the variance holds whatever their
distribution.

These are population values. A backtest estimates them with an error that
shrinks slowly: the Sharpe ratio of `n` days has a standard error of about
`sqrt(a / n)`.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from ._validation import (
    FloatArray,
    as_real_array,
    check_integer,
    check_nu,
    check_positive,
    check_real,
    check_span_order,
    nu_from_span,
)

__all__ = [
    "acf_generating_function",
    "ar1_acf",
    "european_daily_moments",
    "european_expected_return",
    "european_sharpe_ratio",
    "european_volatility",
    "expected_signal_turnover",
]


def ar1_acf(phi: float, n_lags: int) -> FloatArray:
    r"""Autocorrelations $\rho(m) = \phi^m$, `m = 1..n_lags`, of an AR(1) process."""
    phi = check_real(phi, "phi")
    if not -1.0 < phi < 1.0:
        raise ValueError(f"phi must satisfy -1 < phi < 1, got {phi!r}")
    n_lags = check_integer(n_lags, "n_lags")
    return np.asarray(phi ** np.arange(1, n_lags + 1, dtype=np.float64), dtype=np.float64)


def acf_generating_function(acf: ArrayLike | None, nu: float) -> float:
    r"""Centred autocorrelation generating function.

    $$
    \Psi_\nu = \sum_{m=1}^{\infty} \nu^m \rho(m) \qquad \text{(eq. 5.3)}
    $$

    Parameters
    ----------
    acf : array_like or None
        Autocorrelations at lags `1, 2, ...`; lags beyond the end of the
        array count as zero. `None` or an empty array is white noise. The
        array must not start with the lag-0 value of one.
    nu : float
        Smoothing parameter of the filter, `0 <= nu < 1`.

    Returns
    -------
    float
        $\Psi_\nu$. It is zero for white noise and
        $\nu\phi/(1-\nu\phi)$ for an AR(1) process.
    """
    nu = check_nu(nu)
    return nu * _psi_over_nu(_as_acf(acf), nu)


def european_daily_moments(
    *,
    span_long: float,
    span_short: float | None = None,
    acf: ArrayLike | None = None,
    instrument_sharpe: float = 0.0,
    sigma_target_annual: float = 0.15,
    a: float = 260,
) -> tuple[float, float]:
    r"""Mean and variance of the daily return of the European system.

    For the single filter (Proposition 5.3 with unit variance and no excess
    kurtosis), with $l = \sqrt{(1+\nu)/(1-\nu)}$:

    $$
    \mathbb{E}[f_t] = \frac{l\,\sigma_{\mathrm{target}}}{\sqrt{a}}\,(A_\nu + \mu^2),
    $$

    $$
    \operatorname{Var}[f_t] = \Big(\frac{l\,\sigma_{\mathrm{target}}}{\sqrt{a}}\Big)^2
    \big(B_\nu + A_\nu^2 + \mu^2 (1 + B_\nu + 2A_\nu)\big),
    $$

    $$
    A_\nu = \frac{1-\nu}{\nu}\,\Psi_\nu, \qquad
    B_\nu = \frac{1-\nu}{1+\nu}\,(1 + 2\Psi_\nu) .
    $$

    For the long-short filter the moments of the signal are those of
    Corollary 5.4, combined in the same way.

    Parameters
    ----------
    span_long : float
        Span of the (slow) filter.
    span_short : float, optional
        Span of the fast filter of a long-short system. `None` is the single
        filter.
    acf : array_like, optional
        Autocorrelations of the normalised returns at lags `1, 2, ...`
        (not from lag 0); `None` is white noise. Supply enough lags for the
        sum $\sum \nu^m \rho(m)$ to converge at the slow span: later lags
        count as zero.
    instrument_sharpe : float, default 0.0
        Annualised Sharpe ratio of the instrument, that is, the annualised mean
        of the normalised returns, $\sqrt{a}\,\mu$.
    sigma_target_annual : float, default 0.15
        Annualised volatility target.
    a : float, default 260
        Annualisation factor.

    Returns
    -------
    tuple of float
        `(mean, variance)` of the daily return of the system.
    """
    target = check_positive(sigma_target_annual, "sigma_target_annual")
    factor = check_positive(a, "a")
    drift = check_real(instrument_sharpe, "instrument_sharpe") / math.sqrt(factor)
    rho = _as_acf(acf)

    nu_long = nu_from_span(span_long, "span_long")
    if span_short is None:
        # l = sqrt(span), l (1 - nu) = 2 sqrt(span) / (span + 1)
        ratio = _psi_over_nu(rho, nu_long)
        loading = math.sqrt(float(span_long))
        signal_mean = loading * drift
        signal_variance = 1.0 + 2.0 * nu_long * ratio
        covariance = 2.0 * loading / (float(span_long) + 1.0) * ratio
    else:
        check_span_order(span_short, span_long)
        nu_short = nu_from_span(span_short, "span_short")
        slow, fast = float(span_long), float(span_short)
        scale_ls, difference = _long_short_terms(slow, fast)
        # The moments of Corollary 5.4, summed lag by lag. With the weights
        # q (nu1^k - nu2^k) of the filter and D_k = (nu1^k - nu2^k) / (nu1 - nu2):
        #   E[S]      = (l1 - l2) mu = q (nu1 - nu2) mu / ((1 - nu1)(1 - nu2))
        #   Cov[z, S] = q (nu1 - nu2) sum_m rho(m) D_{m-1}
        #   Var[S]    = 1 + 2 / (1 + nu1 nu2) sum_m rho(m) (D_{m+1} - (nu1 nu2)^2 D_{m-1})
        # Nothing here is a small difference of large terms, however close the spans.
        lags = np.arange(1, rho.size + 1, dtype=np.float64)
        below = _power_differences(nu_long, nu_short, difference, lags - 1.0)
        above = _power_differences(nu_long, nu_short, difference, lags + 1.0)
        product = nu_long * nu_short
        signal_mean = scale_ls * (slow + 1.0) * (fast + 1.0) / 4.0 * drift
        covariance = scale_ls * float(np.sum(rho * below))
        signal_variance = 1.0 + 2.0 / (1.0 + product) * float(
            np.sum(rho * (above - product * product * below))
        )

    # f = scale * z * S with z of mean `drift` and unit variance; the variance of
    # the product of two jointly Gaussian variables is given by Lemma 5.2.
    scale = target / math.sqrt(factor)
    mean = scale * (covariance + drift * signal_mean)
    variance = scale**2 * (
        signal_variance
        + covariance**2
        + drift**2 * signal_variance
        + signal_mean**2
        + 2.0 * drift * signal_mean * covariance
    )
    if not (signal_variance > 0.0 and variance > 0.0):
        raise ValueError(
            "acf is not a valid autocorrelation function: it gives the signal a "
            f"variance of {signal_variance:g}"
        )
    return mean, variance


def european_expected_return(
    *,
    span_long: float,
    span_short: float | None = None,
    acf: ArrayLike | None = None,
    instrument_sharpe: float = 0.0,
    sigma_target_annual: float = 0.15,
    a: float = 260,
) -> float:
    r"""Expected annual return of the European system (eqs. 4.22 and 4.23).

    For the single filter,

    $$
    \bar F_{1y} = l\,\sigma_{\mathrm{target}}\sqrt{a}\,\frac{1-\nu}{\nu}\,\Psi_\nu
    + \frac{l\,\sigma_{\mathrm{target}}}{\sqrt{a}}\,(\mu^{z}_{an})^2 ,
    $$

    where $\mu^{z}_{an}$ is the Sharpe ratio of the instrument. The first
    term is earned from autocorrelation, the second from drift. The result does
    not depend on the distribution of the returns beyond its first two moments.
    See `european_daily_moments` for the parameters.
    """
    mean, _ = european_daily_moments(
        span_long=span_long,
        span_short=span_short,
        acf=acf,
        instrument_sharpe=instrument_sharpe,
        sigma_target_annual=sigma_target_annual,
        a=a,
    )
    return float(a) * mean


def european_volatility(
    *,
    span_long: float,
    span_short: float | None = None,
    acf: ArrayLike | None = None,
    instrument_sharpe: float = 0.0,
    sigma_target_annual: float = 0.15,
    a: float = 260,
) -> float:
    """Annualised volatility of the European system for Gaussian returns.

    Equal to the target for white noise without drift. Drift and positive
    autocorrelation raise it; negative autocorrelation lowers it. A system run
    on data also estimates the volatility it divides by, which adds a few per
    cent. See `european_daily_moments` for the parameters.
    """
    _, variance = european_daily_moments(
        span_long=span_long,
        span_short=span_short,
        acf=acf,
        instrument_sharpe=instrument_sharpe,
        sigma_target_annual=sigma_target_annual,
        a=a,
    )
    return math.sqrt(float(a) * variance)


def european_sharpe_ratio(
    *,
    span_long: float,
    span_short: float | None = None,
    acf: ArrayLike | None = None,
    instrument_sharpe: float = 0.0,
    a: float = 260,
) -> float:
    r"""Annualised Sharpe ratio of the European system for Gaussian returns.

    For the single filter this is equation (5.12) without the kurtosis term:

    $$
    SR = \frac{\sqrt{a}\,A_\nu + (\mu^{z}_{an})^2/\sqrt{a}}
    {\sqrt{B_\nu + A_\nu^2 + \big((\mu^{z}_{an})^2/a\big)(1 + B_\nu + 2A_\nu)}} .
    $$

    The Sharpe ratio is defined on the moments of daily returns,
    $\sqrt{a}\,\mathbb{E}[f_t]/\sqrt{\operatorname{Var}[f_t]}$, and does
    not depend on the volatility target. See `european_daily_moments`
    for the parameters.
    """
    mean, variance = european_daily_moments(
        span_long=span_long,
        span_short=span_short,
        acf=acf,
        instrument_sharpe=instrument_sharpe,
        sigma_target_annual=1.0,
        a=a,
    )
    return math.sqrt(float(a)) * mean / math.sqrt(variance)


def expected_signal_turnover(
    *,
    span_long: float,
    span_short: float | None = None,
    sigma_target_annual: float = 0.15,
    a: float = 260,
) -> float:
    r"""Expected annual turnover of the European signal for Gaussian white noise.

    $$
    \mathbb{E}\big[a\,\sigma_{\mathrm{target}}\lvert S_t - S_{t-1}\rvert\big]
    = \frac{2a}{\sqrt{\pi}}\,\sigma_{\mathrm{target}}\sqrt{1-\nu}
    \qquad \text{(eq. 4.16)}
    $$

    for the single filter, and the same with $\sqrt{\zeta}$ in place of
    $\sqrt{1-\nu}$ for the long-short filter (eq. 4.17), where
    $\zeta = (1-\nu_1)(1-\nu_2)/(1+\nu_1\nu_2)$.

    This is a proxy for the volatility-normalised turnover of the system
    (`tfunify.volatility_weighted_turnover`): it leaves out the trades caused
    by changes of the volatility estimate, and it assumes independent Gaussian
    returns. The paper reports the turnover of the full system about 4 % above
    the proxy for a single filter, and 1.6 to 2.3 times the proxy for the
    long-short filter, whose signal changes so little from day to day that the
    volatility updates dominate.

    Parameters
    ----------
    span_long : float
        Span of the (slow) filter.
    span_short : float, optional
        Span of the fast filter of a long-short system.
    sigma_target_annual : float, default 0.15
        Annualised volatility target.
    a : float, default 260
        Annualisation factor.

    Returns
    -------
    float
        Expected turnover per year. The unit is that of the volatility target:
        a value of 0.88 at a target of 0.15 means that the trades of a year,
        each weighted by the volatility of the instrument, add up to 5.9 times
        the target.
    """
    target = check_positive(sigma_target_annual, "sigma_target_annual")
    factor = check_positive(a, "a")
    nu_from_span(span_long, "span_long")
    if span_short is None:
        loading = 2.0 / (float(span_long) + 1.0)  # 1 - nu
    else:
        check_span_order(span_short, span_long)
        # zeta of equation (4.17) simplifies to (1 - nu1)(1 - nu2) / (1 + nu1 nu2),
        # which in terms of the spans is 2 / (span_long * span_short + 1)
        loading = 2.0 / (float(span_long) * float(span_short) + 1.0)
    return 2.0 * factor / math.sqrt(math.pi) * target * math.sqrt(loading)


def _as_acf(acf: ArrayLike | None) -> FloatArray:
    """Autocorrelations at lags 1, 2, ... as an array (empty for white noise)."""
    if acf is None:
        return np.zeros(0)
    rho = as_real_array(acf, "acf")
    if rho.size == 0:
        return np.zeros(0)
    if rho.ndim != 1:
        raise ValueError(f"acf must be one-dimensional, got shape {rho.shape}")
    if not np.all(np.isfinite(rho)):
        raise ValueError("acf contains NaN or infinite values")
    if np.any(np.abs(rho) > 1.0):
        raise ValueError("acf must hold autocorrelations, which lie between -1 and 1")
    if rho[0] == 1.0:
        raise ValueError(
            "acf must start at lag 1, but its first value is 1: drop the lag-0 "
            "entry that estimators such as statsmodels' acf put in front"
        )
    return rho


def _psi_over_nu(rho: FloatArray, nu: float) -> float:
    r"""$\Psi_\nu / \nu = \sum_{m \ge 1} \nu^{m-1} \rho(m)$, finite at `nu = 0`."""
    if rho.size == 0:
        return 0.0
    return float(np.sum(nu ** np.arange(rho.size, dtype=np.float64) * rho))


def _power_differences(
    nu_long: float, nu_short: float, difference: float, exponents: FloatArray
) -> FloatArray:
    r"""$(\nu_1^k - \nu_2^k)/(\nu_1 - \nu_2)$ for the exponents $k \ge 0$ given.

    The numerator is evaluated as
    $\nu_1^k\,(1 - e^{-k \log(1 + (\nu_1-\nu_2)/\nu_2)})$, which keeps its
    digits when the two parameters are close; `difference` is their
    difference, computed from the spans.
    """
    powers = np.asarray(nu_long**exponents, dtype=np.float64)
    if nu_short > 0.0:
        powers = powers * -np.expm1(-exponents * math.log1p(difference / nu_short))
    else:  # nu_2**k is zero, except that 0**0 is one
        powers = np.where(exponents > 0.0, powers, 0.0)
    return np.asarray(powers / difference, dtype=np.float64)


def _long_short_terms(span_long: float, span_short: float) -> tuple[float, float]:
    r"""$q\,(\nu_1-\nu_2)$ and $\nu_1-\nu_2$ from the spans, without cancellation."""
    both = (span_long + 1.0) * (span_short + 1.0)
    difference = 2.0 * (span_long - span_short) / both
    scale = (
        4.0
        / both
        * math.sqrt(
            span_long * span_short * (span_long + span_short) / (span_long * span_short + 1.0)
        )
    )
    return scale, difference
