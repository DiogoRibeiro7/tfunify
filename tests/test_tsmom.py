"""Time-series momentum against equation (A.16) evaluated day by day."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tests import reference
from tfunify import TSMOM, TSMOMConfig, TSMOMResult, pct_returns_from_prices


@pytest.fixture
def rng():
    return np.random.default_rng(271828)


def returns(rng, n=500, vol=0.011, drift=0.0002):
    return drift + vol * rng.standard_normal(n)


CONFIGS = [
    TSMOMConfig(L=5, M=4, span_sigma=10),
    TSMOMConfig(L=10, M=10, span_sigma=33),
    TSMOMConfig(L=1, M=7, span_sigma=5),
    TSMOMConfig(L=7, M=1, span_sigma=5, warmup=3),
    TSMOMConfig(L=3, M=5, span_sigma=40, sigma_target_annual=0.2, a=252),
    TSMOMConfig(L=5, M=4, span_sigma=10, signs="period"),
    TSMOMConfig(L=10, M=10, span_sigma=33, signs="period"),
    TSMOMConfig(L=4, M=3, span_sigma=30, signs="period"),
    TSMOMConfig(L=1, M=6, span_sigma=5, signs="period", warmup=2),
]


class TestDefinition:
    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_matches_the_definition_evaluated_day_by_day(self, rng, cfg):
        r = returns(rng)
        result = TSMOM(cfg).run_from_returns(r)
        pnl, weights, signal, rebalance = reference.tsmom(
            r,
            target=cfg.sigma_target_annual,
            a=cfg.a,
            span_sigma=cfg.span_sigma,
            L=cfg.L,
            M=cfg.M,
            signs=cfg.signs,
            warmup=cfg.warmup,
        )
        np.testing.assert_array_equal(result.rebalance, rebalance)
        np.testing.assert_allclose(result.signal, signal, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(result.weights, weights, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(result.pnl, pnl, rtol=1e-10, atol=1e-17)
        assert np.any(result.weights != 0.0)

    def test_worked_example_daily_signs(self):
        # L = 2, M = 2: lookback of four days, rebalanced every second day
        r = np.array([0.01, 0.02, -0.01, 0.03, 0.01, -0.02, -0.01, -0.03])
        result = TSMOM(TSMOMConfig(L=2, M=2, span_sigma=3, warmup=1)).run_from_returns(r)
        np.testing.assert_array_equal(
            result.rebalance, [False, False, False, True, False, True, False, True]
        )
        # day 3: signs + + - +  -> 2 / sqrt(4);  day 5: - + + -  -> 0;  day 7: + - - -  -> -1
        np.testing.assert_allclose(result.signal, [0, 0, 0, 1, 1, 0, 0, -1])
        sigma = result.volatility
        scale = 0.15 / math.sqrt(260)
        np.testing.assert_allclose(
            result.weights,
            [0, 0, 0, scale / sigma[2], scale / sigma[2], 0, 0, -scale / sigma[6]],
            rtol=1e-14,
        )

    def test_worked_example_period_signs(self):
        # one sign per two-day period, of the sum of its normalised returns
        r = np.array([0.01, 0.02, 0.01, 0.03, 0.01, 0.02, -0.01, -0.03])
        cfg = TSMOMConfig(L=2, M=2, span_sigma=3, warmup=1, signs="period")
        result = TSMOM(cfg).run_from_returns(r)
        root = math.sqrt(2)
        # periods: + + + -.  Day 3: the lookback starts at day 0, which cannot be
        # normalised (one day of warm-up), so no signal.  Day 5: (+ +) / sqrt(2).
        # Day 7: (+ -) / sqrt(2) = 0.
        np.testing.assert_allclose(result.signal, [0, 0, 0, 0, 0, root, root, 0], atol=1e-15)
        scale = 0.15 / math.sqrt(260)
        np.testing.assert_allclose(
            result.weights,
            [
                0,
                0,
                0,
                0,
                0,
                root * scale / result.volatility[4],
                root * scale / result.volatility[4],
                0,
            ],
            rtol=1e-14,
        )

    def test_signal_depends_on_the_lookback_only_through_its_length(self, rng):
        # (A.16): M periods of L days hold M * L daily signs; L sets the rebalancing
        r = returns(rng, 600)
        weekly = TSMOM(TSMOMConfig(L=5, M=12)).run_from_returns(r)
        monthly = TSMOM(TSMOMConfig(L=20, M=3)).run_from_returns(r)
        both = weekly.rebalance & monthly.rebalance
        assert both.sum() == monthly.rebalance.sum() > 5
        np.testing.assert_array_equal(weekly.signal[both], monthly.signal[both])
        np.testing.assert_array_equal(weekly.weights[both], monthly.weights[both])

    def test_daily_signal_ignores_the_volatility_estimate(self, rng):
        r = returns(rng)
        a = TSMOM(TSMOMConfig(L=5, M=6, span_sigma=10)).run_from_returns(r)
        b = TSMOM(TSMOMConfig(L=5, M=6, span_sigma=80)).run_from_returns(r)
        np.testing.assert_array_equal(a.signal, b.signal)
        assert not np.array_equal(a.weights, b.weights)

    def test_default_configuration_is_the_one_of_the_paper(self):
        cfg = TSMOM().cfg
        assert (cfg.L, cfg.M, cfg.span_sigma, cfg.signs) == (10, 10, 33, "daily")
        assert (cfg.sigma_target_annual, cfg.a) == (0.15, 260)


class TestGrid:
    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_rebalancing_days_close_the_periods(self, rng, cfg):
        result = TSMOM(cfg).run_from_returns(returns(rng, 300))
        days = np.flatnonzero(result.rebalance) + 1  # counted from one
        np.testing.assert_array_equal(days, np.arange(cfg.L * cfg.M, 301, cfg.L))

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_position_is_held_between_rebalancing_days(self, rng, cfg):
        result = TSMOM(cfg).run_from_returns(returns(rng, 300))
        quiet = ~result.rebalance
        quiet[0] = False
        np.testing.assert_array_equal(np.diff(result.weights)[quiet[1:]], 0.0)
        np.testing.assert_array_equal(np.diff(result.signal)[quiet[1:]], 0.0)
        first = np.flatnonzero(result.rebalance)[0]
        np.testing.assert_array_equal(result.weights[:first], 0.0)
        np.testing.assert_array_equal(result.signal[:first], 0.0)

    def test_a_zero_signal_closes_the_position(self):
        # 0.1.3 kept the previous weight whenever the new one was zero
        up, flat = [0.01] * 4, [0.01, -0.01, 0.01, -0.01]
        r = np.array(up + flat + up + flat + flat)
        result = TSMOM(TSMOMConfig(L=4, M=1, span_sigma=3, warmup=1)).run_from_returns(r)
        np.testing.assert_array_equal(result.signal[3::4], [2, 0, 2, 0, 0])
        assert np.all(result.weights[3:7] > 0)
        np.testing.assert_array_equal(result.weights[7:11], 0.0)
        assert np.all(result.weights[11:15] > 0)
        np.testing.assert_array_equal(result.weights[15:], 0.0)

    def test_weight_uses_the_volatility_of_the_day_before(self, rng):
        cfg = TSMOMConfig(L=5, M=4, span_sigma=10)
        r = returns(rng)
        result = TSMOM(cfg).run_from_returns(r)
        days = np.flatnonzero(result.rebalance)
        np.testing.assert_allclose(
            result.weights[days],
            result.signal[days] * 0.15 / (math.sqrt(260) * result.volatility[days - 1]),
            rtol=1e-12,
        )

    def test_no_weight_until_the_volatility_is_available(self, rng):
        cfg = TSMOMConfig(L=2, M=2, span_sigma=30)
        # every day up, so that the signal is 4 / sqrt(4) at every rebalancing
        result = TSMOM(cfg).run_from_returns(np.abs(returns(rng, 200)) + 1e-4)
        np.testing.assert_array_equal(result.signal[3:], 2.0)
        # sigma exists from index 29; a rebalancing day t needs sigma[t-1], and
        # the rebalancing days are the odd ones
        np.testing.assert_array_equal(result.weights[:31], 0.0)
        assert np.all(result.weights[31:] > 0.0)

    def test_period_signs_wait_for_a_fully_normalised_lookback(self, rng):
        # one period of two days, so that the signal is +/-1 and never zero by chance
        cfg = TSMOMConfig(L=2, M=1, span_sigma=30, signs="period")
        result = TSMOM(cfg).run_from_returns(returns(rng, 200))
        # returns can be normalised from index 30; the lookback [t-1, t] must start there
        assert np.all(result.signal[:31] == 0.0)
        assert abs(result.signal[31]) == 1.0

    def test_period_whose_returns_cancel_has_no_sign(self):
        # With nu = 1/3 the volatility is 0.005 on days 3 and 4 alike, so the
        # normalised returns of the second period are 0, -1 and +1: a sum of zero,
        # which in floating point comes out as a few 1e-17 of either sign.
        r = 0.005 * np.array([-2.0, 0.0, -2.0, 0.0, -1.0, 1.0, 1.0, 1.0, 1.0])
        cfg = TSMOMConfig(L=3, M=1, span_sigma=2, signs="period")
        result = TSMOM(cfg).run_from_returns(r)
        np.testing.assert_allclose(result.volatility[3:5], 0.005, rtol=1e-14)
        np.testing.assert_array_equal(result.rebalance, [0, 0, 1, 0, 0, 1, 0, 0, 1])
        # first period: reaches into the warm-up; second: cancels; third: up
        np.testing.assert_array_equal(result.signal[[2, 5, 8]], [0.0, 0.0, 1.0])
        np.testing.assert_array_equal(result.weights[:8], 0.0)
        assert result.weights[8] > 0.0

    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_zero_returns_in_front_of_a_series(self, rng, cfg):
        # A series padded with its first price, by a whole number of periods so
        # that the grid stays where it was. Once the lookback has left the
        # padding, nothing differs; and inside it no weight is of another order.
        r = returns(rng, 400)
        pad = 12 * cfg.L
        plain = TSMOM(cfg).run_from_returns(r)
        late = TSMOM(cfg).run_from_returns(np.r_[np.zeros(pad), r])
        np.testing.assert_array_equal(late.volatility[pad:], plain.volatility)
        assert np.all(np.isnan(late.volatility[:pad]))
        start = cfg.L * cfg.M - 1  # the first rebalancing day of the unpadded series
        for name in ("weights", "signal", "rebalance"):
            np.testing.assert_array_equal(
                getattr(late, name)[pad + start :], getattr(plain, name)[start:], err_msg=name
            )
        np.testing.assert_array_equal(late.pnl[pad + start + 1 :], plain.pnl[start + 1 :])
        np.testing.assert_array_equal(late.weights[:pad], 0.0)
        # in between, the lookback still holds padding, and the weights are sized
        # with the volatility of the series itself
        scale = cfg.sigma_target_annual / math.sqrt(cfg.a)
        for t in np.flatnonzero(late.rebalance[: pad + start]):
            lagged = plain.volatility[t - pad - 1] if t > pad else math.nan
            expected = late.signal[t] * scale / lagged if np.isfinite(lagged) else 0.0
            assert late.weights[t] == pytest.approx(expected, rel=1e-12)


class TestTiming:
    @pytest.mark.parametrize("cfg", CONFIGS)
    def test_results_do_not_depend_on_later_data(self, rng, cfg):
        r = returns(rng, 320)
        full = TSMOM(cfg).run_from_returns(r)
        for cut in (cfg.L * cfg.M, 173, 319):
            part = TSMOM(cfg).run_from_returns(r[:cut])
            for field in dataclasses.fields(TSMOMResult):
                np.testing.assert_array_equal(
                    getattr(part, field.name), getattr(full, field.name)[:cut]
                )

    def test_the_weight_of_a_day_earns_the_return_of_the_next(self, rng):
        r = returns(rng)
        result = TSMOM(CONFIGS[0]).run_from_returns(r)
        assert result.pnl[0] == 0.0
        np.testing.assert_allclose(result.pnl[1:], result.weights[:-1] * r[1:], rtol=1e-15)


class TestStatistics:
    @pytest.mark.parametrize("signs", ["daily", "period"])
    @pytest.mark.parametrize(("L", "M"), [(10, 10), (5, 4), (21, 12)])
    def test_signal_has_unit_variance_and_the_system_runs_at_its_target(self, signs, L, M):
        rng = np.random.default_rng(3)
        cfg = TSMOMConfig(L=L, M=M, signs=signs)
        result = TSMOM(cfg).run_from_returns(0.01 * rng.standard_normal(400_000))
        grid = result.signal[result.rebalance][50:]
        # five standard errors, for the longest lookback (252 days: 1600
        # independent values of the signal in the sample)
        assert grid.std() == pytest.approx(1.0, abs=0.09)
        assert abs(grid.mean()) < 0.15
        # about 3 % above the target, because the volatility is estimated
        assert 0.97 < result.pnl[2000:].std() * math.sqrt(cfg.a) / 0.15 < 1.10

    def test_signal_is_bounded(self, rng):
        r = np.abs(returns(rng))  # every day up
        daily = TSMOM(TSMOMConfig(L=5, M=4)).run_from_returns(r)
        period = TSMOM(TSMOMConfig(L=5, M=4, signs="period")).run_from_returns(r)
        assert daily.signal.max() == pytest.approx(math.sqrt(20))
        assert period.signal.max() == pytest.approx(math.sqrt(4))

    def test_mirrored_returns_mirror_the_position(self, rng):
        r = returns(rng)
        for cfg in (CONFIGS[0], CONFIGS[5]):
            base = TSMOM(cfg).run_from_returns(r)
            mirrored = TSMOM(cfg).run_from_returns(-r)
            np.testing.assert_allclose(mirrored.weights, -base.weights, rtol=1e-12)
            np.testing.assert_allclose(mirrored.pnl, base.pnl, rtol=1e-12)

    @pytest.mark.parametrize("scale", [1e-3, 50.0])
    def test_scale_of_the_returns_does_not_matter(self, rng, scale):
        r = returns(rng, drift=0.0)
        for cfg in (CONFIGS[0], CONFIGS[5]):
            base = TSMOM(cfg).run_from_returns(r)
            scaled = TSMOM(cfg).run_from_returns(scale * r)
            np.testing.assert_allclose(scaled.signal, base.signal, rtol=1e-12)
            np.testing.assert_allclose(scaled.pnl, base.pnl, rtol=1e-10, atol=1e-15)


class TestPrices:
    def test_prices_are_turned_into_simple_returns(self, rng):
        prices = 60 * np.cumprod(1 + returns(rng, 300))
        cfg = CONFIGS[0]
        from_prices = TSMOM(cfg).run_from_prices(prices)
        from_returns = TSMOM(cfg).run_from_returns(pct_returns_from_prices(prices)[1:])
        assert from_prices.pnl.shape == prices.shape
        assert from_prices.pnl[0] == 0.0
        assert from_prices.weights[0] == 0.0
        assert not from_prices.rebalance[0]
        assert math.isnan(from_prices.volatility[0])
        for field in dataclasses.fields(TSMOMResult):
            np.testing.assert_array_equal(
                getattr(from_prices, field.name)[1:], getattr(from_returns, field.name)
            )
        # in price indices the rebalancing days are the multiples of L
        np.testing.assert_array_equal(np.flatnonzero(from_prices.rebalance) % cfg.L, 0)

    def test_constant_prices_give_no_position_and_no_nan(self):
        for signs in ("daily", "period"):
            result = TSMOM(TSMOMConfig(L=3, M=3, span_sigma=5, signs=signs)).run_from_prices(
                np.full(80, 10.0)
            )
            np.testing.assert_array_equal(result.pnl, 0.0)
            np.testing.assert_array_equal(result.weights, 0.0)
            np.testing.assert_array_equal(result.signal, 0.0)

    def test_series_shorter_than_the_lookback_is_an_error(self):
        cfg = TSMOMConfig(L=10, M=10)
        with pytest.raises(ValueError, match=r"M \* L = 100 returns, got 99"):
            TSMOM(cfg).run_from_returns(np.full(99, 0.01))
        with pytest.raises(ValueError, match=r"M \* L = 100 returns, got 99"):
            TSMOM(cfg).run_from_prices(np.full(100, 50.0))
        assert TSMOM(cfg).run_from_returns(np.full(100, 0.01)).rebalance.sum() == 1

    @pytest.mark.parametrize(
        ("prices", "message"),
        [([100.0], "at least 2"), ([100.0, 0.0], "positive"), ([100.0, math.nan], "NaN")],
    )
    def test_rejects_invalid_prices(self, prices, message):
        with pytest.raises(ValueError, match=message):
            TSMOM().run_from_prices(prices)


class TestRiskControls:
    def test_weight_cap(self, rng):
        r = returns(rng)
        cfg = TSMOMConfig(L=5, M=4, span_sigma=10)
        plain = TSMOM(cfg).run_from_returns(r)
        capped = TSMOM(dataclasses.replace(cfg, weight_cap=0.7)).run_from_returns(r)
        assert np.abs(plain.weights).max() > 0.7
        np.testing.assert_array_equal(capped.weights, np.clip(plain.weights, -0.7, 0.7))

    def test_floor(self, rng):
        r = 1e-5 * rng.standard_normal(300)
        cfg = TSMOMConfig(L=5, M=4, span_sigma=10, sigma_floor_annual=0.04)
        result = TSMOM(cfg).run_from_returns(r)
        np.testing.assert_allclose(result.volatility[10:], 0.04 / math.sqrt(260))
        days = np.flatnonzero(result.rebalance)
        np.testing.assert_allclose(result.weights[days], result.signal[days] * 0.15 / 0.04)


class TestConfig:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"sigma_target_annual": 0.0}, "sigma_target_annual"),
            ({"a": -260}, "a must"),
            ({"span_sigma": 0}, "span_sigma"),
            ({"L": 0}, "L must"),
            ({"L": 2.5}, "L must"),
            ({"M": 0}, "M must"),
            ({"M": True}, "M must"),
            ({"signs": "weekly"}, "signs"),
            ({"warmup": 0}, "warmup"),
            ({"sigma_floor_annual": -1.0}, "sigma_floor_annual"),
            ({"weight_cap": -2.0}, "weight_cap"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            TSMOMConfig(**kwargs)

    def test_stores_plain_python_numbers(self):
        typed = TSMOMConfig(
            L=np.int64(5), M=np.int32(4), span_sigma=np.float32(10), a=np.int64(252)
        )
        assert typed == TSMOMConfig(L=5, M=4, span_sigma=10, a=252)
        assert (type(typed.L), type(typed.M)) == (int, int)
        assert (type(typed.span_sigma), type(typed.a)) == (float, float)

    def test_unpacks_in_the_documented_order(self, rng):
        result = TSMOM(CONFIGS[0]).run_from_returns(returns(rng))
        pnl, weights, signal, volatility = result
        assert pnl is result.pnl
        assert weights is result.weights
        assert signal is result.signal
        assert volatility is result.volatility
        assert result.rebalance.dtype == np.bool_
        # the grid is reached by name: 0.1 returned four arrays
        assert len(result) == 4
        assert result[1] is result.weights
        assert result[3] is result.volatility
