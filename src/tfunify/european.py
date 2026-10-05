"""European trend-following system: a continuous signal with volatility targeting.

Definition 4.1 of Sepp and Lucic (arXiv:2607.19497v1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from ._results import Unpackable
from ._validation import (
    FloatArray,
    as_series,
    check_integer,
    check_optional_cap,
    check_positive,
    check_real,
    check_span_order,
    checked_span,
    store,
)
from .core import (
    ewma_variance_preserving,
    ewma_volatility_from_returns,
    long_short_variance_preserving,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
)

__all__ = ["EuropeanTF", "EuropeanTFConfig", "EuropeanTFResult"]


@dataclass(frozen=True)
class EuropeanTFConfig:
    """Parameters of the European system.

    Parameters
    ----------
    sigma_target_annual : float, default 0.15
        Annualised volatility target of the position in one instrument.
    a : float, default 260
        Annualisation factor: trading days per year.
    span_sigma : float, default 33
        Span, in days, of the EWMA volatility estimate.
    mode : {"longshort", "single"}, default "longshort"
        `"single"` filters the normalised returns with one variance-preserving
        EWMA of span `span_long`. `"longshort"` uses the long-short filter
        with spans `span_long` and `span_short`.
    span_long : float, default 250
        Span of the slow filter.
    span_short : float, default 20
        Span of the fast filter. Used in `"longshort"` mode only, where it
        must be smaller than `span_long`.
    warmup : int, optional
        Number of returns, counted from the first one that is not zero, that
        must be observed before the volatility estimate is used. Until then
        returns are not normalised, so the signal and the weight are zero.
        Defaults to `span_sigma` rounded up.
    sigma_floor_annual : float, default 0.0
        Lower bound on the annualised volatility estimate, applied before it is
        used to normalise returns and to size the position. Zero, the default,
        is the system of the paper. A floor keeps the weights bounded when an
        instrument goes quiet.
    weight_cap : float, optional
        Upper bound on the absolute weight. `None`, the default, is the system
        of the paper.
    """

    sigma_target_annual: float = 0.15
    a: float = 260
    span_sigma: float = 33
    mode: str = "longshort"
    span_long: float = 250
    span_short: float = 20
    warmup: int | None = None
    sigma_floor_annual: float = 0.0
    weight_cap: float | None = None

    def __post_init__(self) -> None:
        store(
            self,
            sigma_target_annual=check_positive(self.sigma_target_annual, "sigma_target_annual"),
            a=check_positive(self.a, "a"),
            span_sigma=checked_span(self.span_sigma, "span_sigma"),
            span_long=checked_span(self.span_long, "span_long"),
            span_short=checked_span(self.span_short, "span_short"),
            warmup=None if self.warmup is None else check_integer(self.warmup, "warmup"),
            sigma_floor_annual=check_real(
                self.sigma_floor_annual, "sigma_floor_annual", minimum=0.0
            ),
            weight_cap=check_optional_cap(self.weight_cap, "weight_cap"),
        )
        if self.mode not in ("single", "longshort"):
            raise ValueError(f"mode must be 'single' or 'longshort', got {self.mode!r}")
        if self.mode == "longshort":
            check_span_order(self.span_short, self.span_long)

    @property
    def warmup_periods(self) -> int:
        """Number of returns used to initialise the volatility estimate."""
        return math.ceil(self.span_sigma) if self.warmup is None else int(self.warmup)


@dataclass(frozen=True, eq=False)
class EuropeanTFResult(Unpackable):
    """Output of `EuropeanTF`. All arrays are aligned with the input.

    The object unpacks into its four arrays, in the order
    `pnl, weights, signal, volatility`, and can be indexed like that tuple.

    Attributes
    ----------
    pnl : numpy.ndarray
        Daily return of the system per unit of capital,
        `pnl[t] = weights[t-1] * r[t]`.
    weights : numpy.ndarray
        Exposure per unit of capital decided at the close of each day.
    signal : numpy.ndarray
        Trend signal, of unit variance for serially independent returns.
    volatility : numpy.ndarray
        Daily volatility estimate used for normalisation and sizing; NaN during
        the warm-up, and before the first return that is not zero.
    """

    pnl: FloatArray
    weights: FloatArray
    signal: FloatArray
    volatility: FloatArray

    _unpacked: ClassVar[tuple[str, ...]] = ("pnl", "weights", "signal", "volatility")


class EuropeanTF:
    r"""European trend-following system.

    With daily returns $r_t$, volatility estimates $\sigma_t$ and
    normalised returns $z_t = r_t/\sigma_{t-1}$:

    $$
    S_t = \widetilde{\mathcal{L}}^{(\nu)}(z_t)
    \ \text{ or } \ \widetilde{\mathcal{LS}}^{(\nu_1,\nu_2)}(z_t),
    $$

    $$
    w_t = S_t\,\frac{\sigma_{\mathrm{target}}}{\sqrt{a}\,\sigma_t}, \qquad
    f_t = w_{t-1}\,r_t .
    $$

    The position is proportional to the signal, so it changes every day. For
    serially independent returns the signal has unit variance and the system
    runs close to its volatility target whatever the spans (a few per cent
    above it, because the volatility is estimated).

    Parameters
    ----------
    cfg : EuropeanTFConfig, optional
        Parameters; the defaults are the long-short filter LS(250, 20) with a
        33-day volatility span and a 15 % target.

    Examples
    --------
    >>> import numpy as np
    >>> from tfunify import EuropeanTF, EuropeanTFConfig
    >>> rng = np.random.default_rng(0)
    >>> prices = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(2000)))
    >>> result = EuropeanTF(EuropeanTFConfig(span_long=100, span_short=10)).run_from_prices(prices)
    >>> result.pnl.shape
    (2000,)
    """

    def __init__(self, cfg: EuropeanTFConfig | None = None) -> None:
        self.cfg = EuropeanTFConfig() if cfg is None else cfg

    def run_from_prices(self, prices: ArrayLike) -> EuropeanTFResult:
        """Run the system on a price series.

        Parameters
        ----------
        prices : array_like
            At least two finite, strictly positive prices. Simple returns are
            used.

        Returns
        -------
        EuropeanTFResult
            Arrays aligned with `prices`. Index 0 holds no return, so its
            profit, weight and signal are zero and its volatility is NaN.
        """
        returns = pct_returns_from_prices(prices)[1:]
        result = self.run_from_returns(returns)
        return EuropeanTFResult(
            pnl=np.concatenate(([0.0], result.pnl)),
            weights=np.concatenate(([0.0], result.weights)),
            signal=np.concatenate(([0.0], result.signal)),
            volatility=np.concatenate(([np.nan], result.volatility)),
        )

    def run_from_returns(self, r: ArrayLike) -> EuropeanTFResult:
        """Run the system on a series of returns.

        Parameters
        ----------
        r : array_like
            Daily returns. Every entry is treated as an observed return; the
            first one earns nothing, because no weight precedes it.

        Returns
        -------
        EuropeanTFResult
            Arrays aligned with `r`.
        """
        cfg = self.cfg
        returns = as_series(r, "r")

        sigma = ewma_volatility_from_returns(
            returns, span_to_nu(cfg.span_sigma), min_periods=cfg.warmup_periods
        )
        if cfg.sigma_floor_annual > 0.0:
            sigma = np.maximum(sigma, cfg.sigma_floor_annual / math.sqrt(cfg.a))
        z = vol_normalised_returns(returns, sigma)

        if cfg.mode == "single":
            signal = ewma_variance_preserving(z, span_to_nu(cfg.span_long))
        else:
            signal = long_short_variance_preserving(
                z, span_to_nu(cfg.span_long), span_to_nu(cfg.span_short)
            )

        weights = signal * volatility_target_weights(sigma, cfg.sigma_target_annual, cfg.a)
        if cfg.weight_cap is not None:
            weights = np.clip(weights, -cfg.weight_cap, cfg.weight_cap)

        weights = weights + 0.0  # no negative zeros where the signal is -0.0
        pnl = np.zeros_like(returns)
        pnl[1:] = weights[:-1] * returns[1:] + 0.0
        return EuropeanTFResult(pnl=pnl, weights=weights, signal=signal, volatility=sigma)
