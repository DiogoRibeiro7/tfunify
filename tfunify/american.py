from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from .core import span_to_nu, ewma_variance_preserving

FloatArray = NDArray[np.floating]


def _true_range(high: FloatArray, low: FloatArray, close: FloatArray) -> FloatArray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev_close = np.roll(close, 1); prev_close[0] = close[0]
    hl = np.abs(high - low)
    hc = np.abs(high - prev_close)
    lc = np.abs(low - prev_close)
    return np.maximum(hl, np.maximum(hc, lc))


def _atr(high: FloatArray, low: FloatArray, close: FloatArray, period: int) -> FloatArray:
    tr = _true_range(high, low, close)
    n = tr.size
    out = np.full(n, np.nan, dtype=float)
    csum = np.cumsum(tr)
    out[: period - 1] = np.nan
    out[period - 1 :] = (csum[period - 1 :] - np.r_[0.0, csum[:-period]]) / period
    return out


@dataclass
class AmericanTFConfig:
    span_long: int = 250
    span_short: int = 20
    atr_period: int = 33
    q: float = 5.0
    p: float = 5.0
    r_multiple: float = 0.01  # units = r_multiple * price / ATR


class AmericanTF:
    """American TF: fast/slow filters on price with ATR buffers + trailing stop."""

    def __init__(self, cfg: AmericanTFConfig):
        self.cfg = cfg

    def run(self, close: FloatArray, high: FloatArray | None = None, low: FloatArray | None = None):
        close = np.asarray(close, dtype=float)
        if high is None or low is None:
            high = low = close

        atr_vals = _atr(high, low, close, self.cfg.atr_period)
        nu_long = span_to_nu(self.cfg.span_long)
        nu_short = span_to_nu(self.cfg.span_short)
        s_long = ewma_variance_preserving(close, nu_long)
        s_fast = ewma_variance_preserving(close, nu_short)

        n = close.size
        units = np.zeros(n)
        pnl = np.zeros(n)
        stop = np.full(n, np.nan)
        in_pos = 0  # -1, 0, +1

        for t in range(1, n):
            price = close[t]
            atr_t = atr_vals[t]
            if not np.isfinite(atr_t) or atr_t <= 0.0:
                units[t] = units[t - 1]
                pnl[t] = units[t - 1] * (close[t] - close[t - 1])
                continue

            long_on = s_fast[t] > s_long[t] + self.cfg.q * atr_t
            short_on = s_fast[t] < s_long[t] - self.cfg.q * atr_t

            if in_pos > 0:
                stop[t] = max(stop[t - 1] if np.isfinite(stop[t - 1]) else -np.inf, price - self.cfg.p * atr_t)
            elif in_pos < 0:
                stop[t] = min(stop[t - 1] if np.isfinite(stop[t - 1]) else np.inf, price + self.cfg.p * atr_t)
            else:
                stop[t] = np.nan

            if in_pos > 0:
                breach = price < stop[t]; signal_off = not long_on
                if breach and signal_off:
                    in_pos = 0; units[t] = 0.0
                else:
                    units[t] = units[t - 1]
            elif in_pos < 0:
                breach = price > stop[t]; signal_off = not short_on
                if breach and signal_off:
                    in_pos = 0; units[t] = 0.0
                else:
                    units[t] = units[t - 1]
            else:
                units[t] = 0.0

            if in_pos == 0:
                if long_on:
                    in_pos = +1
                    units[t] = self.cfg.r_multiple * price / atr_t
                    stop[t] = price - self.cfg.p * atr_t
                elif short_on:
                    in_pos = -1
                    units[t] = -self.cfg.r_multiple * price / atr_t
                    stop[t] = price + self.cfg.p * atr_t

            pnl[t] = units[t - 1] * (close[t] - close[t - 1])

        return pnl, units
