"""Filters, returns, volatility and turnover against explicit sums and closed forms."""

from __future__ import annotations

import math
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest

from tests import reference
from tfunify import (
    ewma,
    ewma_variance_preserving,
    ewma_volatility_from_returns,
    log_returns_from_prices,
    long_short_loadings,
    long_short_variance_preserving,
    nu_to_span,
    pct_returns_from_prices,
    span_to_nu,
    vol_normalised_returns,
    volatility_target_weights,
    volatility_weighted_turnover,
)

NUS = [0.0, 0.3, 0.9, 0.992]


@pytest.fixture
def rng():
    return np.random.default_rng(20261005)


class TestSpan:
    @pytest.mark.parametrize(
        ("span", "nu"), [(1, 0.0), (3, 0.5), (19, 0.9), (33, 1 - 2 / 34), (2.5, 1 - 2 / 3.5)]
    )
    def test_definition(self, span, nu):
        assert span_to_nu(span) == pytest.approx(nu, abs=1e-15)

    @pytest.mark.parametrize("span", [1, 2, 20, 33.5, 250, 1e6])
    def test_round_trip(self, span):
        assert nu_to_span(span_to_nu(span)) == pytest.approx(span, rel=1e-9)

    def test_span_is_the_window_of_the_equivalent_moving_average(self):
        # eq. (2.6): the filter divides the variance of white noise by its span
        nu = span_to_nu(40)
        assert (1 - nu) / (1 + nu) == pytest.approx(1 / 40)

    @pytest.mark.parametrize("span", [0, 0.99, -3, math.nan, math.inf, "20", None, True])
    def test_rejects_invalid_spans(self, span):
        with pytest.raises(ValueError, match="span"):
            span_to_nu(span)

    def test_rejects_a_span_whose_parameter_rounds_to_one(self):
        # nu = 1 would be a filter that never moves, and 1 / (1 - nu) a division by zero
        assert span_to_nu(1e15) < 1.0
        with pytest.raises(ValueError, match="span is too large"):
            span_to_nu(1e17)
        with pytest.raises(ValueError, match="span must be finite"):
            span_to_nu(10**400)  # an integer no float can hold

    @pytest.mark.parametrize("nu", [-0.1, 1.0, 1.5, math.nan])
    def test_rejects_invalid_nu(self, nu):
        with pytest.raises(ValueError, match="nu"):
            nu_to_span(nu)


class TestEwma:
    @pytest.mark.parametrize("nu", NUS)
    def test_started_at_the_first_observation(self, rng, nu):
        x = rng.standard_normal(80)
        np.testing.assert_allclose(ewma(x, nu), reference.ewma_sum(x, nu), rtol=0, atol=1e-13)

    @pytest.mark.parametrize("nu", NUS)
    @pytest.mark.parametrize("initial", [0.0, -2.5])
    def test_started_from_a_given_state(self, rng, nu, initial):
        x = rng.standard_normal(80)
        np.testing.assert_allclose(
            ewma(x, nu, initial=initial), reference.ewma_sum(x, nu, initial), rtol=0, atol=1e-13
        )

    def test_recursion(self, rng):
        x = rng.standard_normal(50)
        y = ewma(x, 0.8)
        assert y[0] == x[0]
        # absolute tolerance: the two terms can cancel
        np.testing.assert_allclose(y[1:], 0.2 * x[1:] + 0.8 * y[:-1], rtol=0, atol=1e-15)

    def test_nu_zero_returns_the_input(self, rng):
        x = rng.standard_normal(20)
        np.testing.assert_array_equal(ewma(x, 0.0), x)
        np.testing.assert_array_equal(ewma(x, 0.0, initial=7.0), x)

    def test_constant_input_is_a_fixed_point(self):
        np.testing.assert_allclose(ewma(np.full(200, 3.2), 0.97), 3.2, rtol=1e-15)

    def test_matches_pandas(self, rng):
        pd = pytest.importorskip("pandas")
        x = rng.standard_normal(300)
        expected = pd.Series(x).ewm(span=25, adjust=False).mean().to_numpy()
        # absolute tolerance: the average of a series around zero passes close to zero
        np.testing.assert_allclose(ewma(x, span_to_nu(25)), expected, rtol=1e-12, atol=1e-14)

    def test_accepts_lists_and_does_not_modify_the_input(self):
        x = [1.0, 2.0, 4.0]
        np.testing.assert_allclose(ewma(x, 0.5), [1.0, 1.5, 2.75])
        assert x == [1.0, 2.0, 4.0]

    def test_long_series(self, rng):
        # the loop must stay exact over a long run: compare the tail with the sum
        x = rng.standard_normal(100_000)
        nu = 0.95
        tail = np.sum((1 - nu) * nu ** np.arange(2000) * x[:-2001:-1])
        assert ewma(x, nu)[-1] == pytest.approx(tail, abs=1e-12)

    @pytest.mark.parametrize(
        ("x", "message"),
        [
            ([], "at least 1"),
            ([[1.0, 2.0]], "one-dimensional"),
            ([1.0, math.nan], "NaN"),
            ([1.0, math.inf], "NaN or infinite"),
            (["a"], "numeric"),
            (["1.5", "2.5"], "numeric"),
            ([True, False], "numeric"),
            ([1 + 2j], "complex"),
            ([1.0, None], "NaN|numeric"),
            ([[1.0, 2.0], [3.0]], "numeric"),  # rows of unequal length
            ([1.0, object()], "numeric"),
            (np.array(["1.5", "2"], dtype=object), "numeric, got a value of type str"),
            (np.array([True, False], dtype=object), "numeric, got a value of type bool"),
            (np.array([1.0, np.True_], dtype=object), "numeric, got a value of type"),
            ([10**400, 1], "numeric"),
            (np.ma.masked_array([1.0, 2.0, 3.0], mask=[False, True, False]), "masked"),
        ],
    )
    def test_rejects_invalid_series(self, x, message):
        with pytest.raises(ValueError, match=message):
            ewma(x, 0.5)

    def test_accepts_integers_and_unmasked_masked_arrays(self):
        np.testing.assert_allclose(ewma(np.array([1, 2, 4]), 0.5), [1.0, 1.5, 2.75])
        np.testing.assert_allclose(ewma(np.ma.masked_array([1.0, 2.0, 4.0]), 0.5), [1.0, 1.5, 2.75])
        # numbers held as Python objects, as in a pandas column of dtype object
        mixed = np.array([1, 2.0, Fraction(4)], dtype=object)
        np.testing.assert_allclose(ewma(mixed, 0.5), [1.0, 1.5, 2.75])

    @pytest.mark.parametrize("nu", [-0.01, 1.0, 2.0, math.nan, "0.5"])
    def test_rejects_invalid_nu(self, nu):
        with pytest.raises(ValueError, match="nu"):
            ewma([1.0, 2.0], nu)

    def test_rejects_invalid_initial(self):
        with pytest.raises(ValueError, match="initial"):
            ewma([1.0, 2.0], 0.5, initial=math.inf)


def impulse(n):
    x = np.zeros(n)
    x[0] = 1.0
    return x


class TestVariancePreservingFilter:
    @pytest.mark.parametrize("nu", [0.3, 0.9, 0.97])
    def test_impulse_response_has_unit_energy(self, nu):
        # The variance of the output for white noise is the sum of the squared
        # weights; Proposition 2.4 says it is one.
        response = ewma_variance_preserving(impulse(3000), nu)
        assert np.sum(response**2) == pytest.approx(1.0, abs=1e-12)

    def test_is_the_scaled_filter_started_from_zero(self, rng):
        x = rng.standard_normal(100)
        nu = 0.9
        np.testing.assert_allclose(
            ewma_variance_preserving(x, nu),
            math.sqrt((1 + nu) / (1 - nu)) * reference.ewma_sum(x, nu, 0.0),
            rtol=1e-12,
        )

    def test_started_at_the_first_observation_on_request(self, rng):
        x = rng.standard_normal(30)
        nu = 0.8
        np.testing.assert_allclose(
            ewma_variance_preserving(x, nu, initial=None), 3.0 * ewma(x, nu), rtol=1e-14
        )

    def test_variance_rises_to_the_input_variance_from_below(self):
        # weights of the first t+1 observations: energy 1 - nu**(2(t+1))
        nu = 0.9
        for t in (0, 5, 40):
            weights = ewma_variance_preserving(impulse(t + 1)[::-1].copy(), nu)
            response = ewma_variance_preserving(impulse(t + 1), nu)
            assert np.sum(response**2) == pytest.approx(1 - nu ** (2 * (t + 1)), rel=1e-12)
            assert weights[-1] == pytest.approx(math.sqrt((1 + nu) / (1 - nu)) * (1 - nu))

    def test_white_noise_keeps_its_variance(self, rng):
        x = 2.0 * rng.standard_normal(400_000)
        out = ewma_variance_preserving(x, span_to_nu(50))[2000:]
        assert out.var() == pytest.approx(4.0, rel=0.09)  # standard error 1.6 %

    def test_accepts_a_list_at_nu_zero(self):
        np.testing.assert_array_equal(ewma_variance_preserving([1.0, -2.0], 0.0), [1.0, -2.0])


class TestLongShortFilter:
    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(250, 20), (50, 5), (30, 20), (250, 249), (10, 1), (3, 2)]
    )
    def test_impulse_response_has_unit_energy(self, span_long, span_short):
        # Proposition 2.6. (250, 249) checks the loadings where the three terms
        # of equation (2.10) cancel to twelve digits.
        response = long_short_variance_preserving(
            impulse(40_000), span_to_nu(span_long), span_to_nu(span_short)
        )
        assert np.sum(response**2) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize(("span_long", "span_short"), [(250, 20), (50, 5), (125, 20)])
    def test_loadings_are_those_of_the_paper(self, span_long, span_short):
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        q = reference.paper_q(nu_1, nu_2)
        l_1, l_2 = long_short_loadings(nu_1, nu_2)
        assert l_1 == pytest.approx(q / (1 - nu_1), rel=1e-12)
        assert l_2 == pytest.approx(q / (1 - nu_2), rel=1e-12)
        # the same through the variance-preserving filters (first line of eq. 2.9)
        assert l_1 == pytest.approx(
            q / math.sqrt(1 - nu_1**2) * math.sqrt((1 + nu_1) / (1 - nu_1)), rel=1e-12
        )

    def test_weights_are_q_times_the_difference_of_the_decays(self):
        nu_1, nu_2 = span_to_nu(60), span_to_nu(8)
        response = long_short_variance_preserving(impulse(500), nu_1, nu_2)
        lags = np.arange(500)
        np.testing.assert_allclose(
            response, reference.paper_q(nu_1, nu_2) * (nu_1**lags - nu_2**lags), atol=1e-14
        )
        assert response[0] == pytest.approx(0.0, abs=1e-15)  # no weight on the latest value
        assert np.all(response[1:] > 0.0)

    def test_latest_observation_does_not_move_the_output(self, rng):
        x = rng.standard_normal(300)
        bumped = x.copy()
        bumped[-1] += 10.0
        nu_1, nu_2 = span_to_nu(60), span_to_nu(8)
        assert long_short_variance_preserving(bumped, nu_1, nu_2)[-1] == pytest.approx(
            long_short_variance_preserving(x, nu_1, nu_2)[-1], abs=1e-12
        )

    def test_white_noise_keeps_its_variance(self, rng):
        x = rng.standard_normal(600_000)
        out = long_short_variance_preserving(x, span_to_nu(250), span_to_nu(20))[5000:]
        assert out.std() == pytest.approx(1.0, abs=0.06)  # standard error 0.010

    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(250, 20), (15, 3), (40, 39.5), (250, 249.999)]
    )
    @pytest.mark.parametrize("initial", [0.0, 0.7, None])
    def test_is_the_difference_of_the_two_loaded_filters(self, rng, span_long, span_short, initial):
        # eq. (2.9) literally; the package computes the same sequence as a cascade
        x = rng.standard_normal(400)
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        l_1, l_2 = long_short_loadings(nu_1, nu_2)
        literal = l_1 * ewma(x, nu_1, initial=initial) - l_2 * ewma(x, nu_2, initial=initial)
        # the literal form loses digits as the loadings grow: l_1 / |output| of them
        tolerance = 1e-13 * max(1.0, l_1)
        np.testing.assert_allclose(
            long_short_variance_preserving(x, nu_1, nu_2, initial=initial),
            literal,
            rtol=0,
            atol=tolerance * np.abs(x).max(),
        )

    def test_spans_that_almost_coincide(self, rng):
        # The limit of equal spans exists: the weights tend to q' k nu**k. The
        # loadings themselves grow without bound, and their difference of two
        # filters would be rounding noise; the cascade is not.
        nu_2 = span_to_nu(250)
        nu_1 = math.nextafter(nu_2, 1.0)
        response = long_short_variance_preserving(impulse(60_000), nu_1, nu_2)
        assert np.sum(response**2) == pytest.approx(1.0, abs=1e-9)
        lags = np.arange(60_000)
        shape = lags * nu_2 ** np.maximum(lags - 1, 0)
        np.testing.assert_allclose(response, shape / math.sqrt(np.sum(shape**2)), atol=1e-12)
        assert long_short_loadings(nu_1, nu_2)[0] > 1e12
        x = rng.standard_normal(5000)
        assert np.abs(long_short_variance_preserving(x, nu_1, nu_2)).max() < 10.0

    def test_starts_at_the_first_observation_on_request(self):
        nu_1, nu_2 = span_to_nu(30), span_to_nu(4)
        l_1, l_2 = long_short_loadings(nu_1, nu_2)
        # both filters at the constant: the output is (l_1 - l_2) times it, and stays
        out = long_short_variance_preserving(np.full(50, 2.0), nu_1, nu_2, initial=None)
        np.testing.assert_allclose(out, (l_1 - l_2) * 2.0, rtol=1e-12)
        assert long_short_variance_preserving([3.0], nu_1, nu_2, initial=None)[0] == pytest.approx(
            (l_1 - l_2) * 3.0
        )

    def test_rejects_invalid_initial(self):
        with pytest.raises(ValueError, match="initial"):
            long_short_variance_preserving([1.0, 2.0], 0.9, 0.5, initial=math.nan)

    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(250, 20), (1e4, 250), (1e6, 250), (1e8, 250), (1e8, 1)]
    )
    def test_loadings_keep_their_precision_for_long_spans(self, span_long, span_short):
        # q**2 in exact rational arithmetic on the two parameters as floats see them;
        # 1 - nu**2 evaluated as it stands would be wrong in the tenth digit at 1e8
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        a, b = Fraction(nu_1), Fraction(nu_2)
        q_squared = (1 - a * a) * (1 - b * b) * (1 - a * b) / ((a - b) ** 2 * (1 + a * b))
        l_1, l_2 = long_short_loadings(nu_1, nu_2)
        assert l_1 == pytest.approx(math.sqrt(q_squared / (1 - a) ** 2), rel=1e-14)
        assert l_2 == pytest.approx(math.sqrt(q_squared / (1 - b) ** 2), rel=1e-14)

    @pytest.mark.parametrize(
        ("nu_long", "nu_short"), [(0.5, 0.5), (0.3, 0.7), (1.0, 0.5), (0.9, -0.1)]
    )
    def test_rejects_invalid_parameters(self, nu_long, nu_short):
        with pytest.raises(ValueError, match="nu"):
            long_short_variance_preserving([1.0, 2.0], nu_long, nu_short)


class TestReturns:
    def test_simple_returns(self):
        np.testing.assert_allclose(
            pct_returns_from_prices([100.0, 110.0, 99.0]), [0.0, 0.1, -0.1], atol=1e-15
        )

    def test_log_returns(self):
        np.testing.assert_allclose(
            log_returns_from_prices([100.0, 110.0, 99.0]),
            [0.0, math.log(1.1), math.log(0.9)],
            atol=1e-15,
        )

    def test_small_returns_keep_their_precision(self):
        # s_t / s_{t-1} - 1 would round this return to a multiple of 2.2e-16,
        # and log(s_t) - log(s_{t-1}) is no better
        prices = [1.0, 1.0 + 3 * 2.0**-50]
        assert pct_returns_from_prices(prices)[1] == 3 * 2.0**-50
        assert log_returns_from_prices(prices)[1] == pytest.approx(3 * 2.0**-50, rel=1e-15)

    @pytest.mark.parametrize(("level", "size"), [(5000.0, 1e-10), (80.0, 0.02), (0.3, 1e-5)])
    def test_log_returns_keep_their_precision(self, rng, level, size):
        # Around a price of 5000 a logarithm carries a rounding error of 1e-15,
        # which is a hundred-thousandth of a return of 1e-10: the difference of
        # two logarithms would be wrong in the fifth digit. The reference is the
        # return of the two floats in exact rational arithmetic.
        prices = level * np.cumprod(1 + size * rng.standard_normal(60))
        exact = [
            math.log1p(float(Fraction(after) / Fraction(before) - 1))
            for before, after in pairwise(prices.tolist())
        ]
        np.testing.assert_allclose(log_returns_from_prices(prices)[1:], exact, rtol=1e-12)

    def test_simple_returns_compound_back_to_the_prices(self, rng):
        prices = 50 * np.exp(np.cumsum(0.02 * rng.standard_normal(200)))
        np.testing.assert_allclose(
            prices[0] * np.cumprod(1 + pct_returns_from_prices(prices)), prices, rtol=1e-12
        )

    @pytest.mark.parametrize("function", [pct_returns_from_prices, log_returns_from_prices])
    @pytest.mark.parametrize(
        ("prices", "message"),
        [
            ([100.0], "at least 2"),
            ([100.0, 0.0], "positive"),
            ([100.0, -1.0], "positive"),
            ([100.0, math.nan], "NaN"),
            ([[100.0, 101.0]], "one-dimensional"),
        ],
    )
    def test_rejects_invalid_prices(self, function, prices, message):
        with pytest.raises(ValueError, match=message):
            function(prices)


class TestVolatility:
    @pytest.mark.parametrize("nu", NUS)
    def test_is_the_weighted_mean_of_squared_returns(self, rng, nu):
        r = 0.01 * rng.standard_normal(120)
        np.testing.assert_allclose(
            ewma_volatility_from_returns(r, nu), reference.volatility_sum(r, nu), rtol=1e-11
        )

    def test_weights_sum_to_one_from_the_first_day(self, rng):
        # returns of constant size: any weighted mean of their squares is that size
        r = 0.013 * rng.choice([-1.0, 1.0], size=300)
        for correction in (True, False):
            sigma = ewma_volatility_from_returns(r, span_to_nu(33), bias_correction=correction)
            np.testing.assert_allclose(sigma, 0.013, rtol=1e-12)

    def test_is_unbiased_at_every_horizon(self, rng):
        # mean of sigma**2 over many paths, at days where a start-up bias would show
        paths = 0.02 * rng.standard_normal((40_000, 12))
        variance = np.array(
            [ewma_volatility_from_returns(path, span_to_nu(33)) ** 2 for path in paths]
        )
        np.testing.assert_allclose(variance.mean(axis=0), 0.02**2, rtol=0.03)

    def test_without_correction_is_the_recursion_started_at_the_first_return(self, rng):
        r = 0.01 * rng.standard_normal(100)
        nu = span_to_nu(20)
        sigma = ewma_volatility_from_returns(r, nu, bias_correction=False)
        np.testing.assert_allclose(sigma**2, reference.ewma_sum(r**2, nu), rtol=1e-11)

    def test_both_starts_converge_to_the_same_estimate(self, rng):
        r = 0.01 * rng.standard_normal(2000)
        nu = span_to_nu(33)
        corrected = ewma_volatility_from_returns(r, nu)
        plain = ewma_volatility_from_returns(r, nu, bias_correction=False)
        np.testing.assert_allclose(corrected[1000:], plain[1000:], rtol=1e-12)

    def test_matches_pandas(self, rng):
        pd = pytest.importorskip("pandas")
        r = 0.01 * rng.standard_normal(400)
        expected = np.sqrt(pd.Series(r**2).ewm(span=33, adjust=True).mean().to_numpy())
        np.testing.assert_allclose(
            ewma_volatility_from_returns(r, span_to_nu(33)), expected, rtol=1e-11
        )

    @pytest.mark.parametrize("scale", [1e-160, 1e-6, 1e-4, 0.3, 5.0, 1e160])
    def test_scales_with_the_returns_at_any_level(self, rng, scale):
        # no floor and no cap: 0.01 % a day and 500 % a day are both reported as they
        # are; the two extremes would underflow and overflow if squared as they come
        r = rng.standard_normal(5000)
        base = ewma_volatility_from_returns(r, span_to_nu(33))
        np.testing.assert_allclose(
            ewma_volatility_from_returns(scale * r, span_to_nu(33)), scale * base, rtol=1e-12
        )

    def test_estimates_the_level(self, rng):
        sigma = ewma_volatility_from_returns(rng.standard_normal(60_000), span_to_nu(33))
        # the root of an unbiased variance estimate is slightly low: about 0.4 % here
        assert sigma[500:].mean() == pytest.approx(1.0, rel=0.03)
        assert np.mean(sigma[500:] ** 2) == pytest.approx(1.0, rel=0.03)

    def test_min_periods(self, rng):
        r = rng.standard_normal(50)
        full = ewma_volatility_from_returns(r, 0.9)
        late = ewma_volatility_from_returns(r, 0.9, min_periods=10)
        assert np.all(np.isnan(late[:9]))
        np.testing.assert_array_equal(late[9:], full[9:])
        assert np.all(np.isnan(ewma_volatility_from_returns(r, 0.9, min_periods=80)))

    def test_no_estimate_without_a_return_that_is_not_zero(self):
        # unchanged prices are no evidence of zero volatility
        assert np.all(np.isnan(ewma_volatility_from_returns(np.zeros(10), 0.9)))

    @pytest.mark.parametrize("correction", [True, False])
    def test_starts_at_the_first_return_that_is_not_zero(self, rng, correction):
        r = 0.01 * rng.standard_normal(60)
        padded = np.r_[np.zeros(7), r]
        for min_periods in (1, 5):
            alone = ewma_volatility_from_returns(
                r, 0.9, min_periods=min_periods, bias_correction=correction
            )
            sigma = ewma_volatility_from_returns(
                padded, 0.9, min_periods=min_periods, bias_correction=correction
            )
            assert np.all(np.isnan(sigma[:7]))
            np.testing.assert_array_equal(sigma[7:], alone)
            assert np.flatnonzero(np.isfinite(sigma))[0] == 7 + min_periods - 1

    def test_zero_returns_after_the_start_are_observations(self):
        # a day without a move, once the series has started, is a return of zero
        r = np.array([0.02, 0.0, 0.0, 0.01, 0.0])
        sigma = ewma_volatility_from_returns(r, 0.5)
        np.testing.assert_allclose(sigma, reference.volatility_sum(r, 0.5), rtol=1e-14)
        assert sigma[0] == pytest.approx(0.02)
        assert sigma[1] == pytest.approx(0.02 * math.sqrt(0.5 / 1.5))
        assert np.all(np.diff(sigma[:3]) < 0)

    def test_extreme_magnitudes_are_not_lost_in_the_squares(self):
        # squared as they are, the first pair would be zero and the others infinite
        for size in (1e-170, 1e170, 1e308):
            for correction in (True, False):
                sigma = ewma_volatility_from_returns(
                    [size, -size, size], 0.9, bias_correction=correction
                )
                np.testing.assert_allclose(sigma, size, rtol=1e-14)

    def test_first_estimate_is_the_size_of_the_first_return(self, rng):
        for value in rng.standard_normal(200):
            for correction in (True, False):
                sigma = ewma_volatility_from_returns(
                    [value, 0.5], span_to_nu(33), bias_correction=correction
                )
                assert sigma[0] == abs(value)  # exactly

    @pytest.mark.parametrize("nu", [0.0, 0.5, 0.94])
    def test_a_later_return_of_any_size_changes_nothing_before_it(self, rng, nu):
        # Point in time to the last bit, also across magnitudes that no market
        # has: a scale taken from the whole series would reach back from the
        # large return and round the small ones away.
        early = 1e-170 * rng.standard_normal(40)
        for later in (0.01, 1e150, 1e300):
            series = np.r_[early, later, early]
            sigma = ewma_volatility_from_returns(series, nu)
            np.testing.assert_array_equal(sigma[:40], ewma_volatility_from_returns(early, nu))
            assert np.all(sigma[:40] > 0.0)
            assert sigma[40] == pytest.approx(later * math.sqrt((1 - nu) / (1 - nu**41)), rel=1e-12)
        # and with nu = 0 the estimate is the size of the return of the day, whatever came before
        np.testing.assert_array_equal(
            ewma_volatility_from_returns([1e100, -1e-200, 1e-200], 0.0), [1e100, 1e-200, 1e-200]
        )

    def test_uses_only_the_past(self, rng):
        r = rng.standard_normal(200)
        changed = r.copy()
        changed[150:] = 5.0
        np.testing.assert_array_equal(
            ewma_volatility_from_returns(r, 0.9)[:150],
            ewma_volatility_from_returns(changed, 0.9)[:150],
        )

    @pytest.mark.parametrize("min_periods", [0, -1, 1.5, True])
    def test_rejects_invalid_min_periods(self, min_periods):
        with pytest.raises(ValueError, match="min_periods"):
            ewma_volatility_from_returns([0.01, 0.02], 0.9, min_periods=min_periods)

    def test_rejects_invalid_nu(self):
        with pytest.raises(ValueError, match="nu_sigma"):
            ewma_volatility_from_returns([0.01, 0.02], 1.0)


class TestNormalisedReturns:
    def test_divides_by_the_volatility_of_the_day_before(self, rng):
        r = rng.standard_normal(40)
        sigma = 0.5 + rng.random(40)
        z = vol_normalised_returns(r, sigma)
        assert z[0] == 0.0
        np.testing.assert_allclose(z[1:], r[1:] / sigma[:-1], rtol=1e-15)

    def test_zero_where_the_volatility_is_missing_or_zero(self):
        z = vol_normalised_returns([0.1, 0.2, 0.3, 0.4, 0.5], [math.nan, 0.0, 2.0, 4.0, 1.0])
        np.testing.assert_allclose(z, [0.0, 0.0, 0.0, 0.2, 0.125])

    def test_the_return_of_the_day_does_not_scale_itself(self, rng):
        r = 0.01 * rng.standard_normal(300)
        shocked = r.copy()
        shocked[200] = 0.5
        nu = span_to_nu(33)
        z = vol_normalised_returns(r, ewma_volatility_from_returns(r, nu))
        z_shocked = vol_normalised_returns(shocked, ewma_volatility_from_returns(shocked, nu))
        np.testing.assert_array_equal(z[:200], z_shocked[:200])
        # the shock is divided by the volatility of the day before, which has not seen it
        assert z_shocked[200] == pytest.approx(0.5 / ewma_volatility_from_returns(r, nu)[199])

    @pytest.mark.parametrize(
        ("r", "sigma", "message"),
        [
            ([0.1, 0.2], [1.0], "same length"),
            ([0.1, 0.2], [1.0, -1.0], "negative"),
            ([0.1, 0.2], [1.0, math.inf], "infinite"),
            ([0.1, math.nan], [1.0, 1.0], "NaN"),
            ([0.1, 0.2], [[1.0, 1.0]], "one-dimensional"),
            ([0.1, 0.2], [1.0, 1j], "complex"),
            ([0.1, 0.2], ["a", "b"], "numeric"),
        ],
    )
    def test_rejects_invalid_input(self, r, sigma, message):
        with pytest.raises(ValueError, match=message):
            vol_normalised_returns(r, sigma)


class TestVolatilityTarget:
    def test_definition(self):
        weights = volatility_target_weights([0.01, 0.02], 0.15, 260)
        np.testing.assert_allclose(
            weights, [0.15 / (math.sqrt(260) * 0.01), 0.15 / (math.sqrt(260) * 0.02)], rtol=1e-15
        )

    def test_brings_the_position_to_the_target(self, rng):
        daily = 0.004
        r = daily * rng.standard_normal(200_000)
        weight = volatility_target_weights([daily], 0.15, 260)[0]
        assert (weight * r).std() * math.sqrt(260) == pytest.approx(0.15, rel=0.01)

    def test_no_floor_on_the_volatility(self):
        # 0.1 % a day: far below the 0.5 % at which 0.1.3 stopped scaling up
        assert volatility_target_weights([0.001], 0.15, 260)[0] == pytest.approx(
            0.15 / (math.sqrt(260) * 0.001)
        )

    def test_zero_where_the_volatility_is_missing_or_zero(self):
        np.testing.assert_array_equal(
            volatility_target_weights([math.nan, 0.0, 0.01], 0.15, 260)[:2], [0.0, 0.0]
        )

    @pytest.mark.parametrize(
        ("target", "a", "message"),
        [(0.0, 260, "sigma_target_annual"), (-0.1, 260, "sigma_target_annual"), (0.15, 0, "a")],
    )
    def test_rejects_invalid_parameters(self, target, a, message):
        with pytest.raises(ValueError, match=message):
            volatility_target_weights([0.01], target, a)


class TestTurnover:
    def test_definition(self):
        w = np.array([0.0, 1.0, 1.0, -0.5])
        sigma = np.array([0.01, 0.02, 0.02, 0.04])
        expected = math.sqrt(260) * sigma * np.array([0.0, 1.0, 0.0, 1.5])
        np.testing.assert_allclose(volatility_weighted_turnover(w, sigma, 260), expected)

    def test_missing_volatility_is_allowed_where_nothing_trades(self):
        turnover = volatility_weighted_turnover([0.0, 0.0, 1.0], [math.nan, math.nan, 0.01], 260)
        np.testing.assert_allclose(turnover, [0.0, 0.0, math.sqrt(260) * 0.01])

    def test_missing_volatility_on_a_trade_is_an_error(self):
        with pytest.raises(ValueError, match="sigma is missing"):
            volatility_weighted_turnover([0.0, 1.0], [0.01, math.nan], 260)

    @pytest.mark.parametrize(
        ("w", "sigma", "a", "message"),
        [
            ([0.0, 1.0], [0.01], 260, "same length"),
            ([0.0, 1.0], [0.01, 0.01], 0, "a must"),
            ([0.0, math.nan], [0.01, 0.01], 260, "NaN"),
        ],
    )
    def test_rejects_invalid_input(self, w, sigma, a, message):
        with pytest.raises(ValueError, match=message):
            volatility_weighted_turnover(w, sigma, a)
