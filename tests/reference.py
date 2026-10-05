"""Slow, literal implementations of the definitions, used as test oracles.

Each function evaluates the formulas of the paper as they are written (explicit
sums over lags, one day at a time) and shares no code with the package.
"""

from __future__ import annotations

import math

import numpy as np


def ewma_sum(x, nu, initial=None):
    """EWMA as an explicit weighted sum over lags."""
    x = np.asarray(x, dtype=float)
    out = np.empty(x.size)
    for t in range(x.size):
        lags = np.arange(t + 1)
        if initial is None:
            # started at x[0]: x[0] carries nu**t, the others (1 - nu) * nu**lag
            weights = (1.0 - nu) * nu**lags
            weights[t] = nu**t
            out[t] = np.sum(weights * x[t - lags])
        else:
            out[t] = (1.0 - nu) * np.sum(nu**lags * x[t - lags]) + nu ** (t + 1) * initial
    return out


def first_move(values):
    """Index of the first entry that is not zero; None if there is none."""
    for t, value in enumerate(values):
        if value != 0.0:
            return t
    return None


def volatility_sum(r, nu, min_periods=1):
    """Root of the weighted mean of squared returns, weights nu**lag rescaled to sum to one.

    The mean runs over the returns from the first one that is not zero.
    """
    r = np.asarray(r, dtype=float)
    out = np.full(r.size, np.nan)
    start = first_move(r)
    if start is None:
        return out
    for t in range(start, r.size):
        lags = np.arange(t - start + 1)
        weights = nu**lags
        if lags.size >= min_periods:
            out[t] = math.sqrt(np.sum(weights * r[t - lags] ** 2) / np.sum(weights))
    return out


def paper_q(nu_1, nu_2):
    """Equation (2.10), as printed."""
    return (1.0 / (1.0 - nu_1**2) + 1.0 / (1.0 - nu_2**2) - 2.0 / (1.0 - nu_1 * nu_2)) ** -0.5


def european(r, *, target, a, span_sigma, span_long, span_short=None, warmup=None):
    """European system by Definition 4.1, with explicit sums. Returns f, w, S, sigma."""
    r = np.asarray(r, dtype=float)
    n = r.size
    warmup = math.ceil(span_sigma) if warmup is None else warmup
    sigma = volatility_sum(r, 1.0 - 2.0 / (span_sigma + 1.0), warmup)

    z = np.zeros(n)
    for t in range(1, n):
        if np.isfinite(sigma[t - 1]) and sigma[t - 1] > 0.0:
            z[t] = r[t] / sigma[t - 1]

    nu_1 = 1.0 - 2.0 / (span_long + 1.0)
    lags = np.arange(n)
    if span_short is None:
        kernel = math.sqrt((1.0 + nu_1) / (1.0 - nu_1)) * (1.0 - nu_1) * nu_1**lags
    else:
        nu_2 = 1.0 - 2.0 / (span_short + 1.0)
        q = paper_q(nu_1, nu_2)
        kernel = q / (1.0 - nu_1) * (1.0 - nu_1) * nu_1**lags
        kernel -= q / (1.0 - nu_2) * (1.0 - nu_2) * nu_2**lags

    signal = np.array([np.sum(kernel[: t + 1] * z[t::-1]) for t in range(n)])
    weights = np.zeros(n)
    for t in range(n):
        if np.isfinite(sigma[t]) and sigma[t] > 0.0:
            weights[t] = signal[t] * target / (math.sqrt(a) * sigma[t])
    pnl = np.zeros(n)
    pnl[1:] = weights[:-1] * r[1:]
    return pnl, weights, signal, sigma


def tsmom(r, *, target, a, span_sigma, L, M, signs="daily", warmup=None):
    """TSMOM by equation (A.16), one rebalancing day at a time. Returns f, w, S, rebalance."""
    r = np.asarray(r, dtype=float)
    n = r.size
    warmup = math.ceil(span_sigma) if warmup is None else warmup
    sigma = volatility_sum(r, 1.0 - 2.0 / (span_sigma + 1.0), warmup)
    z = np.zeros(n)
    for t in range(1, n):
        if np.isfinite(sigma[t - 1]) and sigma[t - 1] > 0.0:
            z[t] = r[t] / sigma[t - 1]

    signal = np.zeros(n)
    weights = np.zeros(n)
    rebalance = np.zeros(n, dtype=bool)
    current_signal = current_weight = 0.0
    for t in range(n):
        day = t + 1  # days are counted from one: day L closes the first period
        if day % L == 0 and day >= M * L:
            rebalance[t] = True
            if signs == "daily":
                total = sum(np.sign(r[t - k]) for k in range(M * L))
                current_signal = total / math.sqrt(M * L)
            else:
                total = 0.0
                for m in range(M):
                    end = t - m * L
                    total += np.sign(sum(z[end - k] for k in range(L)))
                current_signal = total / math.sqrt(M)
                # the lookback must not reach into the warm-up: its first return
                # needs the volatility of the day before
                first_return = t - M * L + 1
                if first_return < 1 or not np.isfinite(sigma[first_return - 1]):
                    current_signal = 0.0
            current_weight = 0.0
            if t >= 1 and np.isfinite(sigma[t - 1]) and sigma[t - 1] > 0.0:
                current_weight = target / (math.sqrt(a) * sigma[t - 1]) * current_signal
        signal[t] = current_signal
        weights[t] = current_weight
    pnl = np.zeros(n)
    pnl[1:] = weights[:-1] * r[1:]
    return pnl, weights, signal, rebalance


def true_range_loop(high, low, close):
    out = np.empty(len(close))
    for t in range(len(close)):
        out[t] = abs(high[t] - low[t])
        if t > 0:  # the first day has no previous close
            out[t] = max(out[t], abs(high[t] - close[t - 1]), abs(low[t] - close[t - 1]))
    return out


def atr_loop(high, low, close, period):
    """Mean of the last `period` true ranges, from the first range that is not zero."""
    ranges = true_range_loop(high, low, close)
    out = np.full(len(close), np.nan)
    start = first_move(ranges)
    if start is None:
        return out
    for t in range(start + period - 1, len(close)):
        out[t] = sum(ranges[t - k] for k in range(period)) / period
    return out
