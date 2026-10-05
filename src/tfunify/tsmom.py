"""Time-series momentum: signs of past returns, rebalanced on a grid.

Definition A.4 of Sepp and Lucic (arXiv:2607.19497v1), which generalises the
system of Moskowitz, Ooi and Pedersen (2012).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._results import Unpackable
from ._validation import (
    FloatArray,
    as_series,
    check_integer,
    check_optional_cap,
    check_positive,
    check_real,
    checked_span,
    store,
)
from .core import (
    ewma_volatility_from_returns,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
)

__all__ = ["TSMOM", "TSMOMConfig", "TSMOMResult"]


@dataclass(frozen=True)
class TSMOMConfig:
    """Parameters of the time-series momentum system.

    Parameters
    ----------
    sigma_target_annual : float, default 0.15
        Annualised volatility target of the position in one instrument.
    a : float, default 260
        Annualisation factor: trading days per year.
    span_sigma : float, default 33
        Span, in days, of the EWMA volatility estimate.
    L : int, default 10
        Length of a period in days. The position is rebalanced every `L` days.
    M : int, default 10
        Number of periods in the lookback, which therefore covers `M * L` days.
    signs : {"daily", "period"}, default "daily"
        `"daily"` is the definition of the paper: the signal is the sum of the
        signs of the `M * L` daily returns of the lookback, divided by
        `sqrt(M * L)`. `"period"` takes one sign per period, the sign of the
        sum of its volatility-normalised returns, and divides the sum of the
        `M` signs by `sqrt(M)`; this is the construction of tfunify 0.1. (It
        is not that of Moskowitz, Ooi and Pedersen, who take the sign of the
        return over the trailing twelve months.) Both have unit variance when
        the returns are serially independent and as likely to be positive as
        negative.
    warmup : int, optional
        Number of returns, counted from the first one that is not zero, that
        must be observed before the volatility estimate is used. Until then
        the weight is zero. Defaults to `span_sigma` rounded up.
    sigma_floor_annual : float, default 0.0
        Lower bound on the annualised volatility estimate. Zero is the system of
        the paper.
    weight_cap : float, optional
        Upper bound on the absolute weight. `None` is the system of the paper.
    """

    sigma_target_annual: float = 0.15
    a: float = 260
    span_sigma: float = 33
    L: int = 10
    M: int = 10
    signs: str = "daily"
    warmup: int | None = None
    sigma_floor_annual: float = 0.0
    weight_cap: float | None = None

    def __post_init__(self) -> None:
        store(
            self,
            sigma_target_annual=check_positive(self.sigma_target_annual, "sigma_target_annual"),
            a=check_positive(self.a, "a"),
            span_sigma=checked_span(self.span_sigma, "span_sigma"),
            L=check_integer(self.L, "L"),
            M=check_integer(self.M, "M"),
            warmup=None if self.warmup is None else check_integer(self.warmup, "warmup"),
            sigma_floor_annual=check_real(
                self.sigma_floor_annual, "sigma_floor_annual", minimum=0.0
            ),
            weight_cap=check_optional_cap(self.weight_cap, "weight_cap"),
        )
        if self.signs not in ("daily", "period"):
            raise ValueError(f"signs must be 'daily' or 'period', got {self.signs!r}")

    @property
    def warmup_periods(self) -> int:
        """Number of returns used to initialise the volatility estimate."""
        return math.ceil(self.span_sigma) if self.warmup is None else int(self.warmup)


@dataclass(frozen=True, eq=False)
class TSMOMResult(Unpackable):
    """Output of `TSMOM`. All arrays are aligned with the input.

    The object unpacks into four arrays, in the order
    `pnl, weights, signal, volatility`, and can be indexed like that tuple.

    Attributes
    ----------
    pnl : numpy.ndarray
        Daily return of the system per unit of capital,
        `pnl[t] = weights[t-1] * r[t]`.
    weights : numpy.ndarray
        Exposure per unit of capital: set on rebalancing days and held in
        between.
    signal : numpy.ndarray
        Signal in force on each day: computed on rebalancing days and held in
        between; zero before the first rebalancing.
    volatility : numpy.ndarray
        Daily volatility estimate; NaN during the warm-up, and before the
        first return that is not zero.
    rebalance : numpy.ndarray of bool
        True on the days on which the signal and the weight are recomputed.
    """

    pnl: FloatArray
    weights: FloatArray
    signal: FloatArray
    volatility: FloatArray
    rebalance: NDArray[np.bool_]

    _unpacked: ClassVar[tuple[str, ...]] = ("pnl", "weights", "signal", "volatility")


class TSMOM:
    r"""Time-series momentum with a lookback of `M` periods of `L` days.

    On the rebalancing days $t'$, every `L` days (eq. A.16):

    $$
    S_{t'} = \frac{1}{\sqrt{ML}} \sum_{k=0}^{ML-1} \operatorname{sgn}(r_{t'-k}),
    \qquad
    w_{t'} = \frac{\sigma_{\mathrm{target}}}{\sqrt{a}\,\sigma_{t'-1}}\, S_{t'}
    $$

    and the weight is held until the next rebalancing day. The weight set at
    $t'$ uses returns up to $t'$ and earns from day $t'+1$.
    The signal depends on `L` and `M` only through their product; `L` sets
    how often the position changes. With `signs="period"` the signal is

    $$
    S_{t'} = \frac{1}{\sqrt{M}} \sum_{m=0}^{M-1}
    \operatorname{sgn}\Big(\sum_{k=0}^{L-1} z_{t'-mL-k}\Big),
    \qquad z_t = \frac{r_t}{\sigma_{t-1}} .
    $$

    Periods are counted from the first return: rebalancing days are the last
    days of the consecutive blocks of `L` returns.

    Parameters
    ----------
    cfg : TSMOMConfig, optional
        Parameters; the defaults are `L = M = 10` with a 33-day volatility
        span and a 15 % target.

    Examples
    --------
    >>> import numpy as np
    >>> from tfunify import TSMOM, TSMOMConfig
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(1000)))
    >>> result = TSMOM(TSMOMConfig(L=5, M=12)).run_from_prices(prices)
    >>> int(result.rebalance.sum())
    188
    """

    def __init__(self, cfg: TSMOMConfig | None = None) -> None:
        self.cfg = TSMOMConfig() if cfg is None else cfg

    def run_from_prices(self, prices: ArrayLike) -> TSMOMResult:
        """Run the system on a price series.

        Parameters
        ----------
        prices : array_like
            At least `M * L + 1` finite, strictly positive prices. Simple
            returns are used.

        Returns
        -------
        TSMOMResult
            Arrays aligned with `prices`. Index 0 holds no return.
        """
        returns = pct_returns_from_prices(prices)[1:]
        result = self.run_from_returns(returns)
        return TSMOMResult(
            pnl=np.concatenate(([0.0], result.pnl)),
            weights=np.concatenate(([0.0], result.weights)),
            signal=np.concatenate(([0.0], result.signal)),
            volatility=np.concatenate(([np.nan], result.volatility)),
            rebalance=np.concatenate(([False], result.rebalance)),
        )

    def run_from_returns(self, r: ArrayLike) -> TSMOMResult:
        """Run the system on a series of returns.

        Parameters
        ----------
        r : array_like
            At least `M * L` daily returns. Every entry is treated as an
            observed return.

        Returns
        -------
        TSMOMResult
            Arrays aligned with `r`.
        """
        cfg = self.cfg
        returns = as_series(r, "r")
        period, n_periods = int(cfg.L), int(cfg.M)
        lookback = period * n_periods
        n = returns.size
        if n < lookback:
            raise ValueError(
                f"the lookback needs M * L = {lookback} returns, got {n}; "
                "use a shorter lookback or a longer series"
            )

        sigma = ewma_volatility_from_returns(
            returns, span_to_nu(cfg.span_sigma), min_periods=cfg.warmup_periods
        )
        if cfg.sigma_floor_annual > 0.0:
            sigma = np.maximum(sigma, cfg.sigma_floor_annual / math.sqrt(cfg.a))

        # Rebalancing days: the last day of every block of L returns, once the
        # lookback of M blocks is complete.
        ends = np.arange(lookback - 1, n, period)
        if cfg.signs == "daily":
            cumulative = np.concatenate(([0.0], np.cumsum(np.sign(returns))))
            raw = (cumulative[ends + 1] - cumulative[ends + 1 - lookback]) / math.sqrt(lookback)
        else:
            z = vol_normalised_returns(returns, sigma)
            n_blocks = n // period
            blocks = z[: n_blocks * period].reshape(n_blocks, period)
            sums = blocks.sum(axis=1)
            # A sum that is zero up to rounding counts as zero, not as the sign
            # of its rounding error.
            rounding = 4.0 * np.finfo(np.float64).eps * np.abs(blocks).sum(axis=1)
            block_signs = np.where(np.abs(sums) > rounding, np.sign(sums), 0.0)
            cumulative = np.concatenate(([0.0], np.cumsum(block_signs)))
            last_block = (ends + 1) // period
            raw = (cumulative[last_block] - cumulative[last_block - n_periods]) / math.sqrt(
                n_periods
            )
            # A block that reaches into the warm-up has returns that could not
            # be normalised: wait until the whole lookback can be.
            estimated = np.flatnonzero(np.isfinite(sigma))
            first_normalised = int(estimated[0]) + 1 if estimated.size else n
            raw[ends + 1 - lookback < first_normalised] = 0.0

        # w = S * target / (sqrt(a) * sigma of the day before), as in (A.16).
        lagged_sigma = np.full(ends.size, np.nan)
        has_lag = ends >= 1
        lagged_sigma[has_lag] = sigma[ends[has_lag] - 1]
        usable = np.isfinite(lagged_sigma) & (lagged_sigma > 0.0)
        scale = np.zeros(ends.size)
        np.divide(cfg.sigma_target_annual / math.sqrt(cfg.a), lagged_sigma, out=scale, where=usable)
        grid_weights = raw * scale + 0.0  # no negative zeros
        if cfg.weight_cap is not None:
            grid_weights = np.clip(grid_weights, -cfg.weight_cap, cfg.weight_cap)

        # Hold the signal and the weight until the next rebalancing day.
        rebalance = np.zeros(n, dtype=np.bool_)
        rebalance[ends] = True
        position = np.cumsum(rebalance) - 1  # index of the latest rebalancing day
        held = position >= 0
        signal = np.zeros(n)
        weights = np.zeros(n)
        signal[held] = raw[position[held]]
        weights[held] = grid_weights[position[held]]

        pnl = np.zeros(n)
        pnl[1:] = weights[:-1] * returns[1:] + 0.0
        return TSMOMResult(
            pnl=pnl, weights=weights, signal=signal, volatility=sigma, rebalance=rebalance
        )
