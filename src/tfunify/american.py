"""American trend-following system: breakouts, fixed size, trailing stops.

Definitions A.1 and A.3 of Sepp and Lucic (arXiv:2607.19497v1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._results import Unpackable
from ._validation import (
    FloatArray,
    as_prices,
    check_integer,
    check_optional_cap,
    check_positive,
    check_span_order,
    checked_span,
    store,
)
from .core import ewma, pct_returns_from_prices, span_to_nu

__all__ = [
    "AmericanTF",
    "AmericanTFConfig",
    "AmericanTFResult",
    "average_true_range",
    "true_range",
]


def true_range(high: ArrayLike, low: ArrayLike, close: ArrayLike) -> FloatArray:
    r"""Daily true range.

    The largest of three distances (eq. A.1):

    $$
    tr_t = \max\{\lvert s^{high}_t - s^{low}_t\rvert,\
    \lvert s^{high}_t - s^{close}_{t-1}\rvert,\
    \lvert s^{low}_t - s^{close}_{t-1}\rvert\}
    $$

    Parameters
    ----------
    high, low, close : array_like
        Daily high, low and close prices of equal length, all on the same
        basis (all adjusted for dividends or rolls, or none). The close is not
        required to lie between the low and the high: a settlement price
        sometimes does not.

    Returns
    -------
    numpy.ndarray
        True range. The first day has no previous close, so `tr[0]` is the
        range of that day, `high[0] - low[0]`.
    """
    close_, high_, low_ = _as_ohlc(close, high, low)
    ranges = high_ - low_
    ranges[1:] = np.maximum(
        ranges[1:],
        np.maximum(np.abs(high_[1:] - close_[:-1]), np.abs(low_[1:] - close_[:-1])),
    )
    return np.asarray(ranges, dtype=np.float64)


def average_true_range(
    high: ArrayLike, low: ArrayLike, close: ArrayLike, period: int
) -> FloatArray:
    r"""Average true range over `period` days.

    $$
    ATR_t = \frac{1}{N}\sum_{n=0}^{N-1} tr_{t-n} \qquad \text{(eq. A.2)}
    $$

    The average starts at the first day whose true range is not zero. Days
    before it are not observations of the range: unchanged prices at the start
    of a padded series, or the first day of a series of closes only, which has
    no previous close to measure a range from.

    Parameters
    ----------
    high, low, close : array_like
        Daily high, low and close prices of equal length.
    period : int
        Number of days `N` in the average.

    Returns
    -------
    numpy.ndarray
        Average true range, in price units; NaN until `period` true ranges
        are available, counted from the first one that is not zero.
    """
    period = check_integer(period, "period")
    ranges = true_range(high, low, close)
    out = np.full(ranges.size, np.nan)
    moved = np.flatnonzero(ranges)
    if moved.size:
        first = int(moved[0])
        live = ranges[first:]
        if live.size >= period:
            windows = np.lib.stride_tricks.sliding_window_view(live, period)
            out[first + period - 1 :] = windows.mean(axis=1)
    return out


@dataclass(frozen=True)
class AmericanTFConfig:
    """Parameters of the American system.

    Parameters
    ----------
    span_long : float, default 250
        Span, in days, of the slow EWMA of the price.
    span_short : float, default 20
        Span of the fast EWMA; smaller than `span_long`.
    atr_period : int, default 33
        Number of days in the average true range.
    q : float, default 5.0
        Entry buffer in ATRs: a position opens when the fast filter is more
        than `q` ATRs away from the slow one. (The paper writes it as omega.)
    p : float, default 5.0
        Width of the trailing stop in ATRs.
    r_multiple : float, default 0.01
        Risk multiple `R`. The weight of a new position is
        `R * price / ATR`; its distance to the initial stop is then
        `R * p` of capital, whatever the volatility of the instrument.
    weight_cap : float, optional
        Upper bound on the absolute weight of a new position. `None`, the
        default, is the system of the paper.
    """

    span_long: float = 250
    span_short: float = 20
    atr_period: int = 33
    q: float = 5.0
    p: float = 5.0
    r_multiple: float = 0.01
    weight_cap: float | None = None

    def __post_init__(self) -> None:
        store(
            self,
            span_long=checked_span(self.span_long, "span_long"),
            span_short=checked_span(self.span_short, "span_short"),
            atr_period=check_integer(self.atr_period, "atr_period"),
            q=check_positive(self.q, "q"),
            p=check_positive(self.p, "p"),
            r_multiple=check_positive(self.r_multiple, "r_multiple"),
            weight_cap=check_optional_cap(self.weight_cap, "weight_cap"),
        )
        check_span_order(self.span_short, self.span_long)


@dataclass(frozen=True, eq=False)
class AmericanTFResult(Unpackable):
    """Output of `AmericanTF`. All arrays are aligned with the prices.

    The object unpacks into two arrays, `pnl, weights`, and can be indexed like
    that pair; the other arrays are reached by name.

    Attributes
    ----------
    pnl : numpy.ndarray
        Daily return of the system per unit of capital,
        `pnl[t] = weights[t-1] * (close[t] / close[t-1] - 1)`.
    weights : numpy.ndarray
        Exposure per unit of capital at the close of each day: fixed when a
        position is opened, zero when flat.
    position : numpy.ndarray of int
        `+1` long, `-1` short, `0` flat.
    stop : numpy.ndarray
        Trailing stop level in price units; NaN when flat.
    atr : numpy.ndarray
        Average true range; NaN until `atr_period` true ranges are available.
    fast, slow : numpy.ndarray
        The two EWMA filters of the close.
    """

    pnl: FloatArray
    weights: FloatArray
    position: NDArray[np.int8]
    stop: FloatArray
    atr: FloatArray
    fast: FloatArray
    slow: FloatArray

    _unpacked: ClassVar[tuple[str, ...]] = ("pnl", "weights")


class AmericanTF:
    r"""American trend-following system.

    Two EWMA filters of the close, slow and fast, give the direction; the
    average true range scales the entry buffer, the size and the stop.

    * **Entry**, from flat: long when `fast > slow + q * ATR`, short when
      `fast < slow - q * ATR`. The weight is `+/- R * close / ATR` and the
      stop starts at `close -/+ p * ATR` (eqs. A.6 to A.9).
    * **Exit**: a long position is closed when the close is below the stop of
      the previous day *and* the long entry condition no longer holds;
      otherwise the stop trails,
      `stop[t] = max(stop[t-1], close[t] - p * ATR[t])` (eqs. A.10 to A.13).
      Short positions mirror this.

    The size is fixed when the position is opened and the exit is a stop rather
    than a signal reversal. After an exit the system is flat for at least that
    day: a new position can open on the next day at the earliest.

    Everything is decided on closing prices; the weight decided at the close of
    day `t` earns the return of day `t + 1`. The weight is an exposure per
    unit of capital, held constant over the trade and applied to simple
    returns.

    Parameters
    ----------
    cfg : AmericanTFConfig, optional
        Parameters; the defaults are spans of 250 and 20 days, a 33-day ATR,
        `q = p = 5` and `R = 0.01`.

    Examples
    --------
    >>> import numpy as np
    >>> from tfunify import AmericanTF, AmericanTFConfig
    >>> close = np.linspace(100.0, 200.0, 400)
    >>> result = AmericanTF(AmericanTFConfig(span_long=50, span_short=10, q=1.0)).run(close)
    >>> int(result.position[-1])
    1
    """

    def __init__(self, cfg: AmericanTFConfig | None = None) -> None:
        self.cfg = AmericanTFConfig() if cfg is None else cfg

    def run(
        self,
        close: ArrayLike,
        high: ArrayLike | None = None,
        low: ArrayLike | None = None,
    ) -> AmericanTFResult:
        """Run the system.

        Parameters
        ----------
        close : array_like
            Closing prices, finite and strictly positive.
        high, low : array_like, optional
            Daily highs and lows. Give both or neither; without them the true
            range is the absolute change of the close.

        Returns
        -------
        AmericanTFResult
            Arrays aligned with `close`.
        """
        cfg = self.cfg
        if (high is None) != (low is None):
            raise ValueError("give both high and low, or neither")
        if high is None or low is None:
            high = low = close
        close_, high_, low_ = _as_ohlc(close, high, low)

        atr = average_true_range(high_, low_, close_, cfg.atr_period)
        slow = ewma(close_, span_to_nu(cfg.span_long))
        fast = ewma(close_, span_to_nu(cfg.span_short))
        buffer = cfg.q * atr
        long_on = fast > slow + buffer  # False where the ATR is not available yet
        short_on = fast < slow - buffer

        n = close_.size
        weights: list[float] = []
        position: list[int] = []
        stop: list[float] = []

        state = 0
        weight = 0.0
        level = float("nan")
        # The state machine is sequential by nature: each day depends on the
        # position and on the stop left by the day before. A position can only
        # exist once the ATR is available, and from then on it always is.
        for price, range_, is_long, is_short in zip(
            close_.tolist(), atr.tolist(), long_on.tolist(), short_on.tolist()
        ):
            if state > 0:
                if price < level and not is_long:
                    state, weight, level = 0, 0.0, float("nan")
                else:
                    level = max(level, price - cfg.p * range_)
            elif state < 0:
                if price > level and not is_short:
                    state, weight, level = 0, 0.0, float("nan")
                else:
                    level = min(level, price + cfg.p * range_)
            elif range_ > 0.0:  # flat since yesterday, and the ATR is available
                state = 1 if is_long else -1 if is_short else 0
                if state != 0:
                    weight = state * cfg.r_multiple * price / range_
                    if cfg.weight_cap is not None:
                        weight = max(-cfg.weight_cap, min(cfg.weight_cap, weight))
                    level = price - state * cfg.p * range_
            weights.append(weight)
            position.append(state)
            stop.append(level)

        weights_ = np.asarray(weights, dtype=np.float64)
        pnl = np.zeros(n)
        if n > 1:
            pnl[1:] = weights_[:-1] * pct_returns_from_prices(close_)[1:] + 0.0
        return AmericanTFResult(
            pnl=pnl,
            weights=weights_,
            position=np.asarray(position, dtype=np.int8),
            stop=np.asarray(stop, dtype=np.float64),
            atr=atr,
            fast=fast,
            slow=slow,
        )


def _as_ohlc(
    close: ArrayLike, high: ArrayLike, low: ArrayLike
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Validated close, high and low arrays of equal length."""
    close_ = as_prices(close, "close")
    high_ = as_prices(high, "high")
    low_ = as_prices(low, "low")
    if not close_.shape == high_.shape == low_.shape:
        raise ValueError(
            "close, high and low must have the same length, got "
            f"{close_.size}, {high_.size} and {low_.size}"
        )
    if np.any(high_ < low_):
        raise ValueError("high must not be below low")
    return close_, high_, low_
