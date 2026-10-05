"""One test for each defect of tfunify 0.1.3.

Each of these was reproduced on the 0.1.3 code before it was fixed; the figure
that 0.1.3 produced is quoted in the test.
"""

from __future__ import annotations

import math
from importlib.metadata import version

import numpy as np
import pytest

import tfunify
from tfunify import (
    TSMOM,
    AmericanTF,
    EuropeanTF,
    EuropeanTFConfig,
    TSMOMConfig,
    ewma_volatility_from_returns,
    long_short_variance_preserving,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
)


def test_pct_returns_are_simple_returns():
    # 0.1.3 returned log returns: [0, 0.0953, -0.1054]
    np.testing.assert_allclose(
        pct_returns_from_prices([100.0, 110.0, 99.0]), [0.0, 0.10, -0.10], atol=1e-15
    )


@pytest.mark.parametrize("daily_vol", [0.0001, 0.30])
def test_volatility_estimate_is_not_clipped(daily_vol):
    # 0.1.3 clipped the estimate to [0.0005, 0.15]: a true 0.0001 was reported
    # as 0.0005 and a true 0.30 as 0.15
    rng = np.random.default_rng(0)
    sigma = ewma_volatility_from_returns(daily_vol * rng.standard_normal(60_000), span_to_nu(33))
    assert sigma[500:].mean() == pytest.approx(daily_vol, rel=0.03)


def test_volatility_target_has_no_hidden_floor():
    # 0.1.3 computed the weight with max(sigma, 0.005): an asset with 0.3 % daily
    # volatility (4.8 % a year) got 60 % of the exposure it needed
    daily_vol = 0.003
    weight = volatility_target_weights([daily_vol], 0.15, 260)[0]
    assert weight * daily_vol * math.sqrt(260) == pytest.approx(0.15)


def test_long_short_filter_preserves_variance():
    # 0.1.3 used the reciprocal of the normalising constant q: standard deviation
    # 48.9 instead of 1 for spans (250, 20)
    rng = np.random.default_rng(0)
    out = long_short_variance_preserving(
        rng.standard_normal(400_000), span_to_nu(250), span_to_nu(20)
    )
    assert out[5000:].std() == pytest.approx(1.0, abs=0.07)  # standard error 0.013


@pytest.mark.parametrize("mode", ["single", "longshort"])
@pytest.mark.parametrize("daily_vol", [0.003, 0.01])
def test_european_system_runs_at_its_target(mode, daily_vol):
    # 0.1.3: 4.8 % (single) for a 15 % target at 1 % daily volatility, and
    # 2.8 % (single) / 8.8 % (long-short) at 0.3 %
    rng = np.random.default_rng(0)
    cfg = EuropeanTFConfig(mode=mode)
    result = EuropeanTF(cfg).run_from_returns(daily_vol * rng.standard_normal(200_000))
    # about 6 % above the target, because the volatility is estimated
    assert 0.95 < result.pnl[2000:].std() * math.sqrt(cfg.a) / 0.15 < 1.18


def test_european_signal_is_continuous_not_squashed():
    # 0.1.3 passed the signal through tanh(s / 3): with the inflated long-short
    # filter 88 % of the days had |signal| > 0.99, a binary position
    rng = np.random.default_rng(0)
    signal = EuropeanTF().run_from_returns(0.01 * rng.standard_normal(200_000)).signal[2000:]
    assert np.abs(signal).max() > 2.0  # tanh never exceeds one
    inside = np.mean(np.abs(signal) < 1.0)
    assert inside == pytest.approx(0.68, abs=0.09)  # a unit normal, not +/-1


def test_american_system_goes_long_in_a_rising_market():
    # 0.1.3 scaled the slow filter by sqrt(250) and the fast one by sqrt(20), so
    # "fast < slow" on every day: 0 % long and 99 % short on rising prices
    rng = np.random.default_rng(0)
    shares = {1: [], -1: []}  # share of days with the trend and against it
    for direction in (1, -1):
        for _ in range(16):  # one path is too few: the share varies by 0.12 between paths
            steps = direction * 0.0006 + 0.01 * rng.standard_normal(3000)
            position = AmericanTF().run(100 * np.exp(np.cumsum(steps))).position
            shares[1].append(np.mean(position == direction))
            shares[-1].append(np.mean(position == -direction))
    with_trend, against = np.mean(shares[1]), np.mean(shares[-1])
    assert with_trend > 0.4  # about 0.6; the rest of the time it is mostly flat
    assert with_trend > 3 * against  # about 0.08


def test_american_profit_is_a_return_on_capital():
    # 0.1.3 multiplied a dimensionless weight by a price change, so the result
    # scaled with the price level of the instrument
    rng = np.random.default_rng(1)
    close = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(2000)))
    cheap = AmericanTF().run(close)
    expensive = AmericanTF().run(1000 * close)
    assert np.any(cheap.pnl != 0.0)
    np.testing.assert_allclose(expensive.pnl, cheap.pnl, rtol=1e-9, atol=1e-15)


def test_tsmom_goes_flat_on_a_zero_signal():
    # 0.1.3 filled forward "if w[t] == 0", so a zero signal on a rebalancing day
    # kept the old position: 37 of 37 such days in the reproduction
    rng = np.random.default_rng(0)
    result = TSMOM(TSMOMConfig(L=5, M=2, span_sigma=10)).run_from_returns(
        0.01 * rng.standard_normal(2000)
    )
    zero = result.rebalance & (result.signal == 0.0)
    assert zero.sum() >= 50  # a quarter of 399 rebalancing days
    np.testing.assert_array_equal(result.weights[zero], 0.0)


def test_first_normalised_returns_are_not_inflated():
    # 0.1.3 started the variance filter on a placeholder return of zero; the
    # first normalised returns of this series were -2.6, 12.8, 0.7, -3.4
    rng = np.random.default_rng(0)
    r = 0.01 * rng.standard_normal(1000)
    result = EuropeanTF(EuropeanTFConfig(mode="single", span_long=20)).run_from_returns(r)
    z = vol_normalised_returns(r, result.volatility)
    assert np.all(z[:33] == 0.0)  # warm-up: not used
    assert np.abs(z[33:60]).max() < 6.0
    assert z[33:].std() == pytest.approx(1.0, abs=0.15)  # standard error 0.02


def test_version_is_the_installed_version():
    # 0.1.3 shipped __version__ = "0.1.0"
    assert tfunify.__version__ == version("tfunify")
