"""Summary statistics of a series of daily strategy returns.

The systems return profit and loss per unit of capital, without compounding.
The statistics here follow that convention: annual figures are arithmetic, and
the Sharpe ratio is the ratio of the moments of daily returns (equation 5.1 of
Sepp and Lucic, arXiv:2607.19497v1). No risk-free rate is subtracted: futures
returns are excess returns already. For positions that have to be financed,
such as shares, subtract the financing rate from the returns first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from ._validation import as_series, check_integer, check_positive

__all__ = [
    "PerformanceSummary",
    "annualised_return",
    "annualised_volatility",
    "max_drawdown",
    "performance_summary",
    "sharpe_ratio",
]


@dataclass(frozen=True)
class PerformanceSummary:
    """Statistics of a series of daily returns.

    Attributes
    ----------
    annual_return : float
        Mean daily return times the annualisation factor.
    annual_volatility : float
        Standard deviation of daily returns times the root of the factor.
    sharpe_ratio : float
        `annual_return / annual_volatility`; NaN when the volatility is zero.
    max_drawdown : float
        Largest fall of the cumulative return from an earlier peak, as a
        non-negative number.
    n_observations : int
        Number of daily returns.
    """

    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    n_observations: int

    def __str__(self) -> str:
        return (
            f"annual return {self.annual_return:.2%}, "
            f"annual volatility {self.annual_volatility:.2%}, "
            f"Sharpe ratio {self.sharpe_ratio:.2f}, "
            f"maximum drawdown {self.max_drawdown:.2%} "
            f"({self.n_observations} days)"
        )


def annualised_return(pnl: ArrayLike, a: float = 260) -> float:
    """Mean daily return times the annualisation factor `a`."""
    values = as_series(pnl, "pnl")
    return check_positive(a, "a") * float(np.mean(values))


def annualised_volatility(pnl: ArrayLike, a: float = 260, *, ddof: int = 1) -> float:
    """Standard deviation of daily returns times `sqrt(a)`.

    `ddof` is the delta degrees of freedom of the standard deviation; the
    series must have more than `ddof` observations. A series whose values are
    all equal has a volatility of exactly zero.
    """
    ddof = check_integer(ddof, "ddof", minimum=0)
    values = as_series(pnl, "pnl", min_length=ddof + 1)
    factor = check_positive(a, "a")
    if np.max(values) == np.min(values):
        return 0.0  # a constant series: its computed deviation is rounding noise
    return math.sqrt(factor) * float(np.std(values, ddof=ddof))


def sharpe_ratio(pnl: ArrayLike, a: float = 260, *, ddof: int = 1) -> float:
    r"""Annualised Sharpe ratio $\sqrt{a}\,\mathrm{mean}/\mathrm{std}$ of daily returns.

    Returns NaN when all returns are equal, so that there is no deviation to
    divide by.
    """
    volatility = annualised_volatility(pnl, a, ddof=ddof)
    if volatility == 0.0:
        return math.nan
    return annualised_return(pnl, a) / volatility


def max_drawdown(pnl: ArrayLike) -> float:
    """Largest fall of the cumulative return from an earlier peak.

    The cumulative return is the running sum of `pnl` and starts at zero, so
    a series that only loses has a drawdown equal to its total loss. The result
    is a non-negative number in the units of `pnl`.
    """
    values = as_series(pnl, "pnl")
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    return float(np.max(np.maximum.accumulate(cumulative) - cumulative))


def performance_summary(pnl: ArrayLike, a: float = 260) -> PerformanceSummary:
    """Annual return, volatility, Sharpe ratio and maximum drawdown of `pnl`.

    Parameters
    ----------
    pnl : array_like
        At least two daily returns per unit of capital, as returned by the
        systems.
    a : float, default 260
        Annualisation factor: trading days per year.
    """
    values = as_series(pnl, "pnl", min_length=2)
    return PerformanceSummary(
        annual_return=annualised_return(values, a),
        annual_volatility=annualised_volatility(values, a),
        sharpe_ratio=sharpe_ratio(values, a),
        max_drawdown=max_drawdown(values),
        n_observations=int(values.size),
    )
