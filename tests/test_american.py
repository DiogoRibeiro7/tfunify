"""American system: a hand-worked path, the rules as invariants, and the range."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from tests import reference
from tfunify import (
    AmericanTF,
    AmericanTFConfig,
    AmericanTFResult,
    average_true_range,
    span_to_nu,
    true_range,
)

NAN = math.nan


def ohlc(seed, n=1500, drift=0.0002, vol=0.012):
    """A price path with highs and lows around the close."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(drift + vol * rng.standard_normal(n)))
    high = close * (1 + 0.006 * rng.random(n))
    low = close * (1 - 0.006 * rng.random(n))
    return close, high, low


class TestTrueRange:
    def test_definition(self):
        high = np.array([11.0, 12.0, 10.5, 14.0])
        low = np.array([9.0, 11.0, 9.0, 13.0])
        close = np.array([10.0, 11.5, 9.5, 13.5])
        # day 0: range; day 1: 12 - 10 (gap up); day 2: 11.5 - 9 (gap down); day 3: 14 - 9.5
        np.testing.assert_allclose(true_range(high, low, close), [2.0, 2.0, 2.5, 4.5])

    def test_against_a_loop(self):
        close, high, low = ohlc(1, 300)
        np.testing.assert_allclose(
            true_range(high, low, close), reference.true_range_loop(high, low, close), rtol=1e-14
        )

    def test_first_day_is_the_range_of_that_day(self):
        # there is no previous close; the close of the day itself is not a
        # substitute for it, also when it lies outside the range (a settlement)
        assert true_range([11.0], [9.0], [12.0]).tolist() == [2.0]
        assert true_range([11.0, 11.0], [9.0, 9.0], [12.0, 10.0]).tolist() == [2.0, 3.0]

    def test_close_only_is_the_absolute_change(self):
        close = np.array([10.0, 12.0, 11.0, 11.0])
        np.testing.assert_allclose(true_range(close, close, close), [0.0, 2.0, 1.0, 0.0])

    @pytest.mark.parametrize("period", [1, 2, 7, 33])
    def test_average_against_a_loop(self, period):
        close, high, low = ohlc(2, 200)
        np.testing.assert_allclose(
            average_true_range(high, low, close, period),
            reference.atr_loop(high, low, close, period),
            rtol=1e-12,
        )

    def test_average_is_missing_until_the_window_is_full(self):
        close, high, low = ohlc(3, 50)
        atr = average_true_range(high, low, close, 20)
        assert np.all(np.isnan(atr[:19]))
        assert np.all(np.isfinite(atr[19:]))
        assert np.all(np.isnan(average_true_range(high, low, close, 51)))

    def test_average_starts_at_the_first_range_that_is_not_zero(self):
        # five unchanged prices say nothing about the range of the instrument
        close = np.array([50.0, 50.0, 50.0, 50.0, 50.0, 51.0, 53.0, 52.0, 52.0])
        np.testing.assert_array_equal(true_range(close, close, close), [0, 0, 0, 0, 0, 1, 2, 1, 0])
        np.testing.assert_allclose(
            average_true_range(close, close, close, 2), [NAN] * 6 + [1.5, 1.5, 0.5]
        )
        np.testing.assert_allclose(
            average_true_range(close, close, close, 3), [NAN] * 7 + [4 / 3, 1.0]
        )

    @pytest.mark.parametrize("period", [1, 5, 33])
    def test_average_of_closes_only_needs_one_more_day(self, period):
        # without highs and lows the first day has no range: `period` changes of
        # the close are available on day `period`, counted from zero
        close, _, _ = ohlc(4, 60)
        atr = average_true_range(close, close, close, period)
        assert np.all(np.isnan(atr[:period]))
        assert np.all(np.isfinite(atr[period:]))
        assert atr[period] == pytest.approx(np.abs(np.diff(close))[:period].mean(), rel=1e-12)

    def test_average_of_unchanged_prices_is_missing(self):
        flat = np.full(30, 7.0)
        assert np.all(np.isnan(average_true_range(flat, flat, flat, 3)))

    @pytest.mark.parametrize("period", [0, -1, 2.0, True])
    def test_rejects_invalid_period(self, period):
        with pytest.raises(ValueError, match="period"):
            average_true_range([2.0, 2.0], [1.0, 1.0], [1.5, 1.5], period)

    @pytest.mark.parametrize(
        ("high", "low", "close", "message"),
        [
            ([2.0, 2.0], [1.0, 1.0], [1.5], "same length"),
            ([1.0, 2.0], [1.5, 1.0], [1.2, 1.5], "high must not be below low"),
            ([2.0, math.nan], [1.0, 1.0], [1.5, 1.5], "high contains NaN"),
            ([2.0, 2.0], [0.0, 1.0], [1.5, 1.5], "low must be strictly positive"),
            ([2.0, 2.0], [1.0, 1.0], [1.5, -1.5], "close must be strictly positive"),
        ],
    )
    def test_rejects_invalid_prices(self, high, low, close, message):
        with pytest.raises(ValueError, match=message):
            true_range(high, low, close)


class TestWorkedExample:
    """Nine closes, worked by hand.

    Fast span 1 (the close itself), slow span 3 (nu = 1/2), two-day ATR of
    absolute changes, buffer q = 0.5, stop p = 1, R = 0.01.

    ========  =====  =====  =====  =====  =====  ======  =======  ========  =========
    day           0      1      2      3      4       5        6         7          8
    close       100    100    104    108    110     107      100        96         97
    change        -      0      4      4      2       3        7         4          1
    ATR           -      -      -      4      3     2.5        5       5.5        2.5
    slow        100    100    102    105  107.5  107.25  103.625   99.8125   98.40625
    ========  =====  =====  =====  =====  =====  ======  =======  ========  =========

    The first change that is not zero is that of day 2, so the first two-day ATR
    is that of day 3.
    Day 3: 108 > 105 + 2, long, weight 0.01 * 108 / 4, stop 104. Days 4 and 5 the
    stop trails to 107 and stays (a close equal to the stop is not a breach).
    Day 6: 100 < 107 and the long signal is off: exit. The short signal is on
    that day already, but the system stays flat on the day of the exit.
    Day 7: 96 < 99.8125 - 2.75, short, weight -0.01 * 96 / 5.5, stop 101.5.
    Day 8: the stop trails down to 99.5.
    """

    close = np.array([100.0, 100.0, 104.0, 108.0, 110.0, 107.0, 100.0, 96.0, 97.0])
    cfg = AmericanTFConfig(span_long=3, span_short=1, atr_period=2, q=0.5, p=1.0, r_multiple=0.01)

    @pytest.fixture
    def result(self):
        return AmericanTF(self.cfg).run(self.close)

    def test_indicators(self, result):
        np.testing.assert_allclose(result.atr, [NAN, NAN, NAN, 4, 3, 2.5, 5, 5.5, 2.5])
        np.testing.assert_allclose(
            result.slow, [100, 100, 102, 105, 107.5, 107.25, 103.625, 99.8125, 98.40625]
        )
        np.testing.assert_array_equal(result.fast, self.close)

    def test_positions(self, result):
        np.testing.assert_array_equal(result.position, [0, 0, 0, 1, 1, 1, 0, -1, -1])
        short = -0.01 * 96 / 5.5
        np.testing.assert_allclose(
            result.weights, [0, 0, 0, 0.27, 0.27, 0.27, 0, short, short], rtol=1e-15
        )

    def test_stops(self, result):
        np.testing.assert_allclose(result.stop, [NAN, NAN, NAN, 104, 107, 107, NAN, 101.5, 99.5])

    def test_profit(self, result):
        short = -0.01 * 96 / 5.5
        expected = [
            0,
            0,
            0,
            0,
            0.27 * (110 / 108 - 1),
            0.27 * (107 / 110 - 1),
            0.27 * (100 / 107 - 1),
            0,
            short * (97 / 96 - 1),
        ]
        np.testing.assert_allclose(result.pnl, expected, rtol=1e-13, atol=1e-17)

    def test_weight_cap(self):
        capped = AmericanTF(dataclasses.replace(self.cfg, weight_cap=0.2)).run(self.close)
        short = -0.01 * 96 / 5.5  # 0.17: inside the cap
        np.testing.assert_allclose(
            capped.weights, [0, 0, 0, 0.2, 0.2, 0.2, 0, short, short], rtol=1e-15
        )
        np.testing.assert_array_equal(capped.position, [0, 0, 0, 1, 1, 1, 0, -1, -1])
        # a tighter cap binds on the short side as well
        tight = AmericanTF(dataclasses.replace(self.cfg, weight_cap=0.1)).run(self.close)
        np.testing.assert_allclose(tight.weights, [0, 0, 0, 0.1, 0.1, 0.1, 0, -0.1, -0.1])
        np.testing.assert_array_equal(tight.stop, capped.stop)  # stops do not move


RULE_CASES = [
    (11, AmericanTFConfig(span_long=50, span_short=10, atr_period=20, q=2.0, p=3.0)),
    (12, AmericanTFConfig(span_long=20, span_short=5, atr_period=10, q=0.5, p=1.0)),
    (13, AmericanTFConfig(span_long=250, span_short=20, atr_period=33, q=5.0, p=5.0)),
    (14, AmericanTFConfig(span_long=8, span_short=2, atr_period=3, q=0.2, p=0.5, r_multiple=0.02)),
]


class TestRules:
    """Every day of a random path obeys Definition A.3, and nothing else happens."""

    @pytest.mark.parametrize("close_only", [False, True])
    @pytest.mark.parametrize(("seed", "cfg"), RULE_CASES)
    def test_every_day_follows_the_definition(self, seed, cfg, close_only):
        close, high, low = ohlc(seed)
        if close_only:
            high = low = close
            result = AmericanTF(cfg).run(close)
        else:
            result = AmericanTF(cfg).run(close, high, low)

        atr = reference.atr_loop(high, low, close, cfg.atr_period)
        slow = reference.ewma_sum(close, span_to_nu(cfg.span_long))
        fast = reference.ewma_sum(close, span_to_nu(cfg.span_short))
        np.testing.assert_allclose(result.atr, atr, rtol=1e-12)
        np.testing.assert_allclose(result.slow, slow, rtol=1e-12)
        np.testing.assert_allclose(result.fast, fast, rtol=1e-12)

        position, weights, stop = result.position, result.weights, result.stop
        assert position[0] == 0
        counts = dict.fromkeys(("entry", "exit", "held through a breach", "trail"), 0)
        for t in range(1, close.size):
            available = np.isfinite(atr[t]) and atr[t] > 0
            long_on = available and fast[t] > slow[t] + cfg.q * atr[t]
            short_on = available and fast[t] < slow[t] - cfg.q * atr[t]
            before = position[t - 1]
            if before == 0:
                expected = 1 if long_on else -1 if short_on else 0
                assert position[t] == expected
                if expected != 0:
                    counts["entry"] += 1
                    assert weights[t] == pytest.approx(
                        expected * cfg.r_multiple * close[t] / atr[t], rel=1e-12
                    )
                    assert stop[t] == pytest.approx(close[t] - expected * cfg.p * atr[t])
                    # the distance to the stop, as a return, times the weight is R * p
                    assert abs(weights[t]) * cfg.p * atr[t] / close[t] == pytest.approx(
                        cfg.r_multiple * cfg.p
                    )
                else:
                    assert weights[t] == 0.0
                    assert np.isnan(stop[t])
            else:
                signal_on = long_on if before > 0 else short_on
                breached = close[t] < stop[t - 1] if before > 0 else close[t] > stop[t - 1]
                if breached and not signal_on:
                    counts["exit"] += 1
                    assert position[t] == 0  # flat, never straight into the opposite side
                    assert weights[t] == 0.0
                    assert np.isnan(stop[t])
                else:
                    counts["held through a breach"] += int(breached)
                    assert position[t] == before
                    assert weights[t] == weights[t - 1]  # the size is fixed at inception
                    trailed = close[t] - before * cfg.p * atr[t]
                    expected_stop = (
                        max(stop[t - 1], trailed) if before > 0 else min(stop[t - 1], trailed)
                    )
                    assert stop[t] == pytest.approx(expected_stop, rel=1e-12)
                    counts["trail"] += int(stop[t] != stop[t - 1])
        # the path must have exercised the rules it is meant to check
        assert counts["entry"] >= 3
        assert counts["exit"] >= 2
        assert counts["trail"] >= 10

    def test_a_breach_does_not_close_the_position_while_the_signal_is_on(self):
        # tight stops and a small buffer: breaches with the signal still on are common
        close, high, low = ohlc(12)
        cfg = AmericanTFConfig(span_long=20, span_short=5, atr_period=10, q=0.05, p=0.3)
        result = AmericanTF(cfg).run(close, high, low)
        held = 0
        for t in range(1, close.size):
            if result.position[t - 1] == 1 and close[t] < result.stop[t - 1]:
                long_on = result.fast[t] > result.slow[t] + cfg.q * result.atr[t]
                if long_on:
                    held += 1
                    assert result.position[t] == 1
        assert held >= 5

    @pytest.mark.parametrize(("seed", "cfg"), RULE_CASES)
    def test_profit_is_the_lagged_weight_times_the_simple_return(self, seed, cfg):
        close, high, low = ohlc(seed)
        result = AmericanTF(cfg).run(close, high, low)
        assert result.pnl[0] == 0.0
        np.testing.assert_allclose(
            result.pnl[1:], result.weights[:-1] * (close[1:] - close[:-1]) / close[:-1], rtol=1e-12
        )

    @pytest.mark.parametrize(("seed", "cfg"), RULE_CASES)
    def test_results_do_not_depend_on_later_data(self, seed, cfg):
        close, high, low = ohlc(seed, 600)
        full = AmericanTF(cfg).run(close, high, low)
        for cut in (60, 300, 599):
            part = AmericanTF(cfg).run(close[:cut], high[:cut], low[:cut])
            for field in dataclasses.fields(AmericanTFResult):
                np.testing.assert_array_equal(
                    getattr(part, field.name), getattr(full, field.name)[:cut]
                )


class TestBehaviour:
    def test_long_in_a_rising_market_and_short_in_a_falling_one(self):
        cfg = AmericanTFConfig(span_long=50, span_short=10, atr_period=20, q=1.0, p=3.0)
        rising = AmericanTF(cfg).run(np.linspace(100.0, 300.0, 500))
        falling = AmericanTF(cfg).run(np.linspace(300.0, 100.0, 500))
        assert np.all(rising.position >= 0)
        assert np.all(falling.position <= 0)
        assert np.all(rising.position[100:] == 1)
        assert np.all(falling.position[100:] == -1)
        assert rising.pnl.sum() > 0
        assert falling.pnl.sum() > 0

    def test_default_parameters_take_both_sides(self):
        # 0.1.3 was short on every path; with the paper's parameters a drifting
        # random walk is long in rising stretches and short in falling ones
        rng = np.random.default_rng(7)
        long_share, short_share = [], []
        for _ in range(16):  # the shares of one path vary by 0.12
            up = 100 * np.exp(np.cumsum(0.0006 + 0.01 * rng.standard_normal(3000)))
            position = AmericanTF().run(up).position
            long_share.append(np.mean(position == 1))
            short_share.append(np.mean(position == -1))
        assert np.mean(long_share) > 0.4  # about 0.6
        assert np.mean(long_share) > 3 * np.mean(short_share)  # about 0.08
        assert np.mean(short_share) > 0.01  # and it does go short at times

    def test_a_line_and_its_reflection_take_opposite_sides(self):
        # filters and ranges of a reflected price path are the reflections of the
        # originals, so the positions mirror exactly (the sizes do not: they are
        # proportional to the price)
        cfg = AmericanTFConfig(span_long=30, span_short=5, atr_period=10, q=1.0, p=2.0)
        line = np.linspace(100.0, 200.0, 300)
        up = AmericanTF(cfg).run(line)
        down = AmericanTF(cfg).run(300.0 - line)
        np.testing.assert_array_equal(up.position, -down.position)

    def test_constant_prices_never_trade(self):
        result = AmericanTF(AmericanTFConfig(span_long=5, span_short=2, atr_period=3)).run(
            np.full(60, 50.0)
        )
        np.testing.assert_array_equal(result.position, 0)
        np.testing.assert_array_equal(result.weights, 0.0)
        np.testing.assert_array_equal(result.pnl, 0.0)
        assert np.all(np.isnan(result.atr))

    @pytest.mark.parametrize("close_only", [False, True])
    def test_unchanged_prices_in_front_of_a_series_change_nothing(self, close_only):
        # A series padded with its first price, as data vendors deliver an
        # instrument that started trading later than the others. The padding
        # must not count as sixty days of zero range: the first real move would
        # then meet an ATR of almost nothing, and R * price / ATR would be a
        # position of many times the capital.
        close, high, low = ohlc(21, 900)
        cfg = AmericanTFConfig(span_long=40, span_short=8, atr_period=10, q=1.0, p=2.0)
        pad = np.full(60, close[0])
        if close_only:
            plain = AmericanTF(cfg).run(close)
            padded = AmericanTF(cfg).run(np.r_[pad, close])
        else:
            plain = AmericanTF(cfg).run(close, high, low)
            padded = AmericanTF(cfg).run(np.r_[pad, close], np.r_[pad, high], np.r_[pad, low])
        assert np.any(plain.position == 1)
        assert np.any(plain.position == -1)
        for name in ("position", "weights", "stop", "atr", "pnl"):
            np.testing.assert_array_equal(
                getattr(padded, name)[60:], getattr(plain, name), err_msg=name
            )
        # the filters of a constant are that constant up to rounding
        np.testing.assert_allclose(padded.fast[60:], plain.fast, rtol=1e-13)
        np.testing.assert_allclose(padded.slow[60:], plain.slow, rtol=1e-13)
        np.testing.assert_array_equal(padded.position[:60], 0)
        assert np.all(np.isnan(padded.atr[:60]))

    def test_no_position_is_opened_while_the_range_is_zero(self):
        # A jump followed by unchanged prices. While the jump is inside the ATR
        # window the buffer of three ATRs keeps the system out; afterwards the ATR
        # is zero, the entry condition "fast > slow + 0" holds, and the size
        # R * price / ATR would be infinite.
        close = np.r_[np.full(10, 100.0), np.full(30, 120.0)]
        cfg = AmericanTFConfig(span_long=40, span_short=2, atr_period=3, q=3.0, p=1.0)
        result = AmericanTF(cfg).run(close)
        after_jump = np.arange(close.size) >= 13
        np.testing.assert_array_equal(result.atr[after_jump], 0.0)
        assert np.all(result.fast[after_jump] > result.slow[after_jump])
        np.testing.assert_array_equal(result.position, 0)
        np.testing.assert_array_equal(result.weights, 0.0)
        np.testing.assert_array_equal(result.pnl, 0.0)

    def test_short_series(self):
        result = AmericanTF().run([100.0])
        assert result.pnl.tolist() == [0.0]
        assert result.position.tolist() == [0]
        result = AmericanTF().run([100.0, 101.0, 102.0])
        np.testing.assert_array_equal(result.pnl, 0.0)  # the ATR is never available

    def test_close_only_equals_high_and_low_at_the_close(self):
        close, _, _ = ohlc(5, 400)
        cfg = AmericanTFConfig(span_long=30, span_short=5, atr_period=10, q=1.0, p=2.0)
        np.testing.assert_array_equal(
            AmericanTF(cfg).run(close).weights, AmericanTF(cfg).run(close, close, close).weights
        )

    def test_wider_ranges_mean_smaller_positions(self):
        close, high, low = ohlc(6, 800)
        cfg = AmericanTFConfig(span_long=30, span_short=5, atr_period=10, q=0.5, p=2.0)
        narrow = AmericanTF(cfg).run(close)
        wide = AmericanTF(cfg).run(close, high * 1.02, low * 0.98)
        assert np.abs(wide.weights).max() < np.abs(narrow.weights).max()


class TestInput:
    def test_high_without_low_is_an_error(self):
        close, high, low = ohlc(8, 50)
        with pytest.raises(ValueError, match="both high and low"):
            AmericanTF().run(close, high=high)
        with pytest.raises(ValueError, match="both high and low"):
            AmericanTF().run(close, low=low)

    @pytest.mark.parametrize(
        ("close", "high", "low", "message"),
        [
            ([], None, None, "at least 1"),
            ([1.0, 2.0], [2.0, 3.0, 4.0], [1.0, 1.0, 1.0], "same length"),
            ([1.0, 2.0], [2.0, 1.0], [1.0, 1.5], "high must not be below low"),
            ([1.0, 0.0], None, None, "positive"),
            ([1.0, math.nan], None, None, "NaN"),
            ([[1.0, 2.0]], None, None, "one-dimensional"),
        ],
    )
    def test_rejects_invalid_prices(self, close, high, low, message):
        with pytest.raises(ValueError, match=message):
            AmericanTF().run(close, high, low)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"span_long": 0}, "span_long"),
            ({"span_short": 0.5}, "span_short"),
            ({"span_long": 20, "span_short": 20}, "smaller than span_long"),
            ({"span_long": 20, "span_short": 30}, "smaller than span_long"),
            ({"atr_period": 0}, "atr_period"),
            ({"atr_period": 2.5}, "atr_period"),
            ({"q": 0.0}, "q must"),
            ({"q": -1.0}, "q must"),
            ({"p": 0.0}, "p must"),
            ({"r_multiple": 0.0}, "r_multiple"),
            ({"r_multiple": math.inf}, "r_multiple"),
            ({"weight_cap": 0.0}, "weight_cap"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            AmericanTFConfig(**kwargs)

    def test_default_configuration_is_the_one_of_the_paper(self):
        cfg = AmericanTF().cfg
        assert (cfg.span_long, cfg.span_short, cfg.atr_period) == (250, 20, 33)
        assert (cfg.q, cfg.p, cfg.r_multiple, cfg.weight_cap) == (5.0, 5.0, 0.01, None)

    def test_numpy_scalars_in_the_configuration_do_not_change_the_arithmetic(self):
        # np.float32(3.0) is exactly 3, but left in place it would turn the stop
        # and the weight into single-precision numbers: wrong in the eighth digit
        close, high, low = ohlc(15, 600)
        plain = AmericanTFConfig(span_long=40, span_short=8, atr_period=10, q=1.0, p=3.0)
        typed = AmericanTFConfig(
            span_long=np.float32(40),
            span_short=np.int64(8),
            atr_period=np.int32(10),
            q=np.float32(1.0),
            p=np.float32(3.0),
            r_multiple=np.float64(0.01),
            weight_cap=None,
        )
        assert typed == plain
        for name in ("span_long", "span_short", "q", "p", "r_multiple"):
            assert type(getattr(typed, name)) is float, name
        assert type(typed.atr_period) is int
        expected = AmericanTF(plain).run(close, high, low)
        result = AmericanTF(typed).run(close, high, low)
        assert np.any(expected.position != 0)
        for field in dataclasses.fields(AmericanTFResult):
            np.testing.assert_array_equal(
                getattr(result, field.name), getattr(expected, field.name)
            )

    def test_unpacks_into_profit_and_weights(self):
        close, high, low = ohlc(9, 300)
        result = AmericanTF(AmericanTFConfig(span_long=30, span_short=5, atr_period=10)).run(
            close, high, low
        )
        pnl, weights = result
        assert pnl is result.pnl
        assert weights is result.weights
        assert result.position.dtype == np.int8
        assert len(result) == 2
        assert result[0] is result.pnl
        assert result[1] is result.weights
        with pytest.raises(IndexError):
            result[2]
