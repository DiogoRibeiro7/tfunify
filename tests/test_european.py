"""European system against the literal definition, and its invariances."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tests import reference
from tfunify import (
    EuropeanTF,
    EuropeanTFConfig,
    EuropeanTFResult,
    ewma_volatility_from_returns,
    pct_returns_from_prices,
    span_to_nu,
)


@pytest.fixture
def rng():
    return np.random.default_rng(4107)


def returns(rng, n=400, vol=0.012, drift=0.0003):
    return drift + vol * rng.standard_normal(n)


CONFIGS = [
    EuropeanTFConfig(mode="single", span_long=30, span_sigma=12),
    EuropeanTFConfig(mode="longshort", span_long=60, span_short=8, span_sigma=20),
    EuropeanTFConfig(mode="longshort", span_long=15, span_short=3, span_sigma=5, warmup=2),
    EuropeanTFConfig(mode="single", span_long=10, span_sigma=7.5, sigma_target_annual=0.3, a=252),
]


class TestDefinition:
    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_matches_the_definition_evaluated_with_explicit_sums(self, rng, cfg):
        r = returns(rng)
        result = EuropeanTF(cfg).run_from_returns(r)
        pnl, weights, signal, sigma = reference.european(
            r,
            target=cfg.sigma_target_annual,
            a=cfg.a,
            span_sigma=cfg.span_sigma,
            span_long=cfg.span_long,
            span_short=cfg.span_short if cfg.mode == "longshort" else None,
            warmup=cfg.warmup,
        )
        np.testing.assert_allclose(result.volatility, sigma, rtol=1e-10)
        np.testing.assert_allclose(result.signal, signal, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(result.weights, weights, rtol=1e-9, atol=1e-11)
        np.testing.assert_allclose(result.pnl, pnl, rtol=1e-9, atol=1e-13)

    def test_daily_return_is_the_signal_times_the_normalised_return(self, rng):
        # eq. (4.3): f_t = (target / sqrt(a)) S_{t-1} z_t
        cfg = EuropeanTFConfig(span_long=40, span_short=6, span_sigma=15)
        r = returns(rng)
        result = EuropeanTF(cfg).run_from_returns(r)
        sigma = result.volatility
        z = np.zeros_like(r)
        ok = np.isfinite(sigma[:-1])
        z[1:][ok] = r[1:][ok] / sigma[:-1][ok]
        expected = cfg.sigma_target_annual / math.sqrt(cfg.a) * result.signal[:-1] * z[1:]
        np.testing.assert_allclose(result.pnl[1:], expected, rtol=1e-10, atol=1e-15)

    def test_weight_is_signal_times_volatility_target(self, rng):
        cfg = EuropeanTFConfig(span_long=40, span_short=6, span_sigma=15)
        result = EuropeanTF(cfg).run_from_returns(returns(rng))
        valid = np.isfinite(result.volatility)
        np.testing.assert_allclose(
            result.weights[valid],
            result.signal[valid] * 0.15 / (math.sqrt(260) * result.volatility[valid]),
            rtol=1e-12,
        )

    def test_default_configuration_is_the_one_of_the_paper(self):
        cfg = EuropeanTF().cfg
        assert (cfg.mode, cfg.span_long, cfg.span_short, cfg.span_sigma) == (
            "longshort",
            250,
            20,
            33,
        )
        assert (cfg.sigma_target_annual, cfg.a) == (0.15, 260)
        assert cfg.sigma_floor_annual == 0.0
        assert cfg.weight_cap is None


class TestTiming:
    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_results_do_not_depend_on_later_data(self, rng, cfg):
        r = returns(rng, 300)
        full = EuropeanTF(cfg).run_from_returns(r)
        for cut in (40, 150, 299):
            part = EuropeanTF(cfg).run_from_returns(r[:cut])
            for name in ("pnl", "weights", "signal", "volatility"):
                np.testing.assert_array_equal(getattr(part, name), getattr(full, name)[:cut])

    def test_the_weight_of_a_day_earns_the_return_of_the_next(self, rng):
        r = returns(rng)
        result = EuropeanTF(CONFIGS[1]).run_from_returns(r)
        assert result.pnl[0] == 0.0
        np.testing.assert_allclose(result.pnl[1:], result.weights[:-1] * r[1:], rtol=1e-15)

    def test_warmup(self, rng):
        cfg = EuropeanTFConfig(span_long=30, span_short=5, span_sigma=10, warmup=25)
        r = returns(rng, 200)
        result = EuropeanTF(cfg).run_from_returns(r)
        assert np.all(np.isnan(result.volatility[:24]))
        assert np.all(np.isfinite(result.volatility[24:]))
        # the first return that can be normalised is number 25 (index 25 uses sigma[24])
        assert np.all(result.signal[:25] == 0.0)
        assert np.all(result.weights[:25] == 0.0)
        assert np.all(result.pnl[:26] == 0.0)
        assert result.signal[26] != 0.0

    def test_default_warmup_is_the_volatility_span(self):
        assert EuropeanTFConfig(span_sigma=33).warmup_periods == 33
        assert EuropeanTFConfig(span_sigma=7.2).warmup_periods == 8
        assert EuropeanTFConfig(span_sigma=33, warmup=5).warmup_periods == 5


class TestInvariances:
    @pytest.mark.parametrize("cfg", CONFIGS[:2])
    @pytest.mark.parametrize("scale", [1e-160, 1e-3, 0.2, 40.0, 1e160])
    def test_scale_of_the_returns_does_not_matter(self, rng, cfg, scale):
        # the signal is built from normalised returns and the weight is inversely
        # proportional to the volatility: a quieter asset is simply levered more.
        # (The two extreme scales are there for the arithmetic, not for a market.)
        r = returns(rng, drift=0.0)
        base = EuropeanTF(cfg).run_from_returns(r)
        scaled = EuropeanTF(cfg).run_from_returns(scale * r)
        np.testing.assert_allclose(scaled.signal, base.signal, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(scaled.pnl, base.pnl, rtol=1e-10, atol=1e-14)
        np.testing.assert_allclose(scaled.weights * scale, base.weights, rtol=1e-10, atol=1e-12)

    @pytest.mark.parametrize("cfg", CONFIGS[:2])
    def test_mirrored_returns_mirror_the_position(self, rng, cfg):
        r = returns(rng)
        base = EuropeanTF(cfg).run_from_returns(r)
        mirrored = EuropeanTF(cfg).run_from_returns(-r)
        np.testing.assert_allclose(mirrored.signal, -base.signal, rtol=1e-12)
        np.testing.assert_allclose(mirrored.weights, -base.weights, rtol=1e-12)
        np.testing.assert_allclose(mirrored.pnl, base.pnl, rtol=1e-12)

    def test_target_scales_the_position(self, rng):
        r = returns(rng)
        low = EuropeanTF(EuropeanTFConfig(span_long=30, span_short=5)).run_from_returns(r)
        high = EuropeanTF(
            EuropeanTFConfig(span_long=30, span_short=5, sigma_target_annual=0.30)
        ).run_from_returns(r)
        np.testing.assert_allclose(high.pnl, 2 * low.pnl, rtol=1e-12)
        np.testing.assert_array_equal(high.signal, low.signal)


class TestStatistics:
    @pytest.mark.parametrize("mode", ["single", "longshort"])
    @pytest.mark.parametrize("daily_vol", [0.0005, 0.01, 0.2])
    def test_runs_at_its_target_on_white_noise(self, mode, daily_vol):
        # A unit-variance signal times a unit-variance normalised return, were
        # the volatility known. It is estimated from about 33 returns, and a
        # return divided by an estimate has a standard deviation some 3 % above
        # one: the signal comes out near 1.03 and the realised volatility near
        # 1.06 times the target. The bounds are five standard errors wide.
        rng = np.random.default_rng(99)
        cfg = EuropeanTFConfig(mode=mode)
        result = EuropeanTF(cfg).run_from_returns(daily_vol * rng.standard_normal(300_000))
        realised = result.pnl[3000:].std() * math.sqrt(cfg.a)
        assert 0.97 < realised / 0.15 < 1.16
        assert 0.94 < result.signal[3000:].std() < 1.13
        assert abs(result.signal[3000:].mean()) < 0.17

    def test_makes_money_on_a_trend_and_loses_on_a_reversal(self):
        cfg = EuropeanTFConfig(mode="single", span_long=20, span_sigma=10)
        rng = np.random.default_rng(5)
        noise = 0.002 * rng.standard_normal(600)
        trend = EuropeanTF(cfg).run_from_returns(0.01 + noise)
        assert trend.signal[100:].min() > 3.0  # about sqrt(20) times a normalised return of one
        assert trend.pnl[100:].sum() > 0.0
        # alternating days: yesterday's direction is always wrong
        zigzag = 0.01 * np.where(np.arange(600) % 2 == 0, 1.0, -1.0)
        assert (
            EuropeanTF(EuropeanTFConfig(mode="single", span_long=1.5))
            .run_from_returns(zigzag)
            .pnl.sum()
            < 0.0
        )


class TestPrices:
    def test_prices_are_turned_into_simple_returns(self, rng):
        prices = 80 * np.cumprod(1 + returns(rng, 300))
        cfg = CONFIGS[1]
        from_prices = EuropeanTF(cfg).run_from_prices(prices)
        from_returns = EuropeanTF(cfg).run_from_returns(pct_returns_from_prices(prices)[1:])
        assert from_prices.pnl.shape == prices.shape
        assert from_prices.pnl[0] == 0.0
        assert from_prices.weights[0] == 0.0
        assert from_prices.signal[0] == 0.0
        assert math.isnan(from_prices.volatility[0])
        for name in ("pnl", "weights", "signal", "volatility"):
            np.testing.assert_array_equal(
                getattr(from_prices, name)[1:], getattr(from_returns, name)
            )

    def test_constant_prices_give_no_position(self):
        result = EuropeanTF(
            EuropeanTFConfig(span_long=10, span_short=3, span_sigma=5)
        ).run_from_prices(np.full(100, 42.0))
        for name in ("pnl", "weights", "signal"):
            np.testing.assert_array_equal(getattr(result, name), 0.0)
        # prices that never move are no evidence of a volatility of zero
        assert np.all(np.isnan(result.volatility))

    def test_first_move_after_a_flat_stretch(self):
        prices = np.r_[np.full(40, 100.0), 101.0, 102.0, 101.5, 103.0]
        result = EuropeanTF(
            EuropeanTFConfig(mode="single", span_long=5, span_sigma=5, warmup=1)
        ).run_from_prices(prices)
        for name in ("pnl", "weights", "signal"):
            assert np.all(np.isfinite(getattr(result, name)))
        assert np.all(np.isnan(result.volatility[:40]))
        assert result.volatility[40] == pytest.approx(0.01)  # the move itself: 1 %
        assert result.signal[40] == 0.0  # which cannot be normalised by anything earlier
        # the second move, 102 / 101 - 1, over the 1 % of the day before; the factor
        # is the loading of the latest observation, sqrt((1 + nu) / (1 - nu)) (1 - nu)
        nu = span_to_nu(5)
        assert result.signal[41] == pytest.approx(
            math.sqrt((1 + nu) * (1 - nu)) * (1 / 101) / 0.01, rel=1e-12
        )

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_unchanged_prices_in_front_of_a_series_change_nothing(self, rng, cfg):
        # A series padded with its first price. Counted as observations, sixty
        # returns of zero would be a volatility of zero, and the first real
        # return, divided by it, a signal and a weight of any size.
        prices = 80 * np.cumprod(1 + returns(rng, 300))
        padded = np.r_[np.full(60, prices[0]), prices]
        plain = EuropeanTF(cfg).run_from_prices(prices)
        late = EuropeanTF(cfg).run_from_prices(padded)
        for name in ("pnl", "weights", "signal", "volatility"):
            np.testing.assert_array_equal(getattr(late, name)[60:], getattr(plain, name))
            np.testing.assert_array_equal(getattr(late, name)[:60], getattr(plain, name)[0])
        assert np.any(plain.weights != 0.0)

    @pytest.mark.parametrize(
        ("prices", "message"),
        [([100.0], "at least 2"), ([100.0, -1.0, 3.0], "positive"), ([1.0, math.nan], "NaN")],
    )
    def test_rejects_invalid_prices(self, prices, message):
        with pytest.raises(ValueError, match=message):
            EuropeanTF().run_from_prices(prices)

    @pytest.mark.parametrize(
        ("r", "message"), [([], "at least 1"), ([0.1, math.inf], "infinite"), ([[0.1]], "one-dim")]
    )
    def test_rejects_invalid_returns(self, r, message):
        with pytest.raises(ValueError, match=message):
            EuropeanTF().run_from_returns(r)


class TestRiskControls:
    def test_floor_bounds_the_weight_after_a_quiet_stretch(self):
        # A market that all but stops moving for a year and then wakes up. While
        # it is quiet the volatility estimate decays, to 1.5e-6 a day here, and
        # the weight grows as its reciprocal: 5600 times the capital on the last
        # quiet day, which then earns the first normal return of 1 %.
        days = np.tile([1.0, 1.0, -1.0], 200)  # two days up and one down: a trend
        quiet = np.r_[0.01 * days[:201], 1e-6 * days[:300], 0.01 * days[:99]]
        plain = EuropeanTF(EuropeanTFConfig(span_long=20, span_short=4)).run_from_returns(quiet)
        floored = EuropeanTF(
            EuropeanTFConfig(span_long=20, span_short=4, sigma_floor_annual=0.05)
        ).run_from_returns(quiet)
        floor_daily = 0.05 / math.sqrt(260)
        assert np.nanmin(floored.volatility) == pytest.approx(floor_daily)
        np.testing.assert_allclose(
            floored.volatility[33:],
            np.maximum(ewma_volatility_from_returns(quiet, span_to_nu(33))[33:], floor_daily),
        )
        assert np.abs(plain.weights).max() == pytest.approx(5644, rel=0.01)
        assert np.abs(plain.pnl).max() == pytest.approx(56.4, rel=0.01)  # times the capital
        assert np.abs(floored.weights).max() < 5
        assert np.abs(floored.pnl).max() < 0.05
        # before the floor binds the two systems are the same
        np.testing.assert_array_equal(plain.weights[:200], floored.weights[:200])

    def test_weight_cap(self, rng):
        r = returns(rng)
        cfg = EuropeanTFConfig(span_long=20, span_short=4)
        plain = EuropeanTF(cfg).run_from_returns(r)
        capped = EuropeanTF(dataclasses.replace(cfg, weight_cap=0.5)).run_from_returns(r)
        assert np.abs(plain.weights).max() > 0.5
        np.testing.assert_array_equal(capped.weights, np.clip(plain.weights, -0.5, 0.5))
        np.testing.assert_allclose(capped.pnl[1:], capped.weights[:-1] * r[1:])
        np.testing.assert_array_equal(capped.signal, plain.signal)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sigma_target_annual": 0.0}, "sigma_target_annual"),
            ({"sigma_target_annual": -0.1}, "sigma_target_annual"),
            ({"sigma_target_annual": math.nan}, "sigma_target_annual"),
            ({"a": 0}, "a must"),
            ({"span_sigma": 0.5}, "span_sigma"),
            ({"span_long": 0}, "span_long"),
            ({"span_short": 0}, "span_short"),
            ({"span_long": "250"}, "span_long"),
            ({"mode": "both"}, "mode"),
            ({"mode": "SINGLE"}, "mode"),
            ({"span_long": 20, "span_short": 20}, "smaller than span_long"),
            ({"span_long": 20, "span_short": 50}, "smaller than span_long"),
            ({"warmup": 0}, "warmup"),
            ({"warmup": 2.5}, "warmup"),
            ({"sigma_floor_annual": -0.01}, "sigma_floor_annual"),
            ({"weight_cap": 0.0}, "weight_cap"),
            ({"weight_cap": -1.0}, "weight_cap"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            EuropeanTFConfig(**kwargs)

    def test_single_mode_ignores_the_short_span(self, rng):
        r = returns(rng)
        a = EuropeanTF(
            EuropeanTFConfig(mode="single", span_long=20, span_short=5)
        ).run_from_returns(r)
        b = EuropeanTF(
            EuropeanTFConfig(mode="single", span_long=20, span_short=90)
        ).run_from_returns(r)
        np.testing.assert_array_equal(a.pnl, b.pnl)

    def test_is_immutable(self):
        cfg = EuropeanTFConfig()
        with pytest.raises(dataclasses.FrozenInstanceError):
            cfg.span_long = 10  # type: ignore[misc]

    def test_stores_plain_python_numbers(self, rng):
        typed = EuropeanTFConfig(
            sigma_target_annual=np.float64(0.15),
            a=np.int64(260),
            span_sigma=np.float32(33),
            span_long=np.int32(250),
            span_short=20,
            warmup=np.int64(40),
            sigma_floor_annual=np.float32(0.0),
            weight_cap=np.float64(8.0),
        )
        assert typed == EuropeanTFConfig(warmup=40, weight_cap=8.0)
        for field in dataclasses.fields(typed):
            assert type(getattr(typed, field.name)) in (float, int, str), field.name
        assert type(typed.warmup) is int
        assert typed.warmup_periods == 40
        r = returns(rng)
        np.testing.assert_array_equal(
            EuropeanTF(typed).run_from_returns(r).pnl,
            EuropeanTF(EuropeanTFConfig(warmup=40, weight_cap=8.0)).run_from_returns(r).pnl,
        )
        with pytest.raises(ValueError, match="a must be finite"):
            EuropeanTFConfig(a=10**400)


class TestResult:
    def test_unpacks_in_the_documented_order(self, rng):
        result = EuropeanTF(CONFIGS[0]).run_from_returns(returns(rng))
        assert isinstance(result, EuropeanTFResult)
        pnl, weights, signal, volatility = result
        assert pnl is result.pnl
        assert weights is result.weights
        assert signal is result.signal
        assert volatility is result.volatility

    def test_indexes_and_measures_like_the_tuple_it_replaces(self, rng):
        result = EuropeanTF(CONFIGS[0]).run_from_returns(returns(rng))
        assert len(result) == 4
        assert result[0] is result.pnl
        assert result[-1] is result.volatility
        middle = result[1:3]
        assert isinstance(middle, tuple)
        assert middle[0] is result.weights
        assert middle[1] is result.signal
        with pytest.raises(IndexError):
            result[4]

    def test_does_not_modify_its_input(self, rng):
        r = returns(rng)
        copy = r.copy()
        EuropeanTF(CONFIGS[1]).run_from_returns(r)
        np.testing.assert_array_equal(r, copy)
