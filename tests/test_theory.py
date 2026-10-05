"""Closed forms against the paper's special cases, direct sums and simulation."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tests import reference
from tfunify import (
    EuropeanTF,
    EuropeanTFConfig,
    ewma_variance_preserving,
    long_short_loadings,
    long_short_variance_preserving,
    span_to_nu,
    volatility_weighted_turnover,
)
from tfunify.theory import (
    acf_generating_function,
    ar1_acf,
    european_daily_moments,
    european_expected_return,
    european_sharpe_ratio,
    european_volatility,
    expected_signal_turnover,
)

A = 260
TARGET = 0.15
SPANS = [5, 10, 21, 63, 125, 250, 500]  # the paper's 1w ... 2y


def arfima_acf(d, n_lags):
    """Autocorrelations of fractional noise ARFIMA(0, d, 0): a long-memory process."""
    rho = np.empty(n_lags)
    value = 1.0
    for m in range(1, n_lags + 1):
        value *= (m - 1 + d) / (m - d)
        rho[m - 1] = value
    return rho


ACFS = {
    "white noise": None,
    "AR(1), phi = 0.1": ar1_acf(0.1, 4000),
    "AR(1), phi = -0.3": ar1_acf(-0.3, 4000),
    "MA(2)": np.array([0.35, 0.1]),
    "fractional noise, d = 0.1": arfima_acf(0.1, 4000),
}


def direct_moments(kernel, acf, drift):
    """Mean and variance of z_t * S_{t-1} for Gaussian z, by summing over the kernel.

    ``S_{t-1} = sum_k kernel[k] z_{t-1-k}``; z has mean ``drift``, unit variance
    and autocorrelations ``acf``.
    """
    n = kernel.size
    rho = np.zeros(n + 1)
    rho[0] = 1.0
    if acf is not None:
        m = min(n, len(acf))
        rho[1 : m + 1] = acf[:m]
    # Var[S] = sum_ij k_i k_j rho(|i - j|): lagged products of the kernel
    products = np.correlate(kernel, kernel, mode="full")[n - 1 :]
    var_s = products[0] + 2.0 * np.sum(products[1:] * rho[1:n])
    cov = np.sum(kernel * rho[1 : n + 1])
    mean_s = drift * np.sum(kernel)
    mean = cov + drift * mean_s
    # Isserlis: Var[XY] for jointly Gaussian X (mean drift, variance 1) and Y = S
    variance = var_s + cov**2 + drift**2 * var_s + mean_s**2 + 2 * drift * mean_s * cov
    return mean, variance


def single_kernel(span, n=6000):
    nu = span_to_nu(span)
    return math.sqrt((1 + nu) / (1 - nu)) * (1 - nu) * nu ** np.arange(n)


def long_short_kernel(span_long, span_short, n=6000):
    nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
    l_1, l_2 = long_short_loadings(nu_1, nu_2)
    lags = np.arange(n)
    return l_1 * (1 - nu_1) * nu_1**lags - l_2 * (1 - nu_2) * nu_2**lags


class TestGeneratingFunction:
    @pytest.mark.parametrize("phi", [-0.4, 0.05, 0.6])
    @pytest.mark.parametrize("span", [1, 5, 60])
    def test_ar1_closed_form(self, phi, span):
        nu = span_to_nu(span)
        assert acf_generating_function(ar1_acf(phi, 3000), nu) == pytest.approx(
            nu * phi / (1 - nu * phi), abs=1e-14
        )

    def test_white_noise(self):
        assert acf_generating_function(None, 0.9) == 0.0
        assert acf_generating_function([], 0.9) == 0.0

    def test_definition(self):
        assert acf_generating_function([0.5, 0.2, -0.1], 0.5) == pytest.approx(
            0.5 * 0.5 + 0.25 * 0.2 - 0.125 * 0.1
        )

    def test_ar1_acf(self):
        np.testing.assert_allclose(ar1_acf(0.5, 3), [0.5, 0.25, 0.125])

    @pytest.mark.parametrize(
        ("phi", "n_lags", "message"),
        [(1.0, 5, "phi"), (-1.2, 5, "phi"), (0.5, 0, "n_lags"), (0.5, 2.5, "n_lags")],
    )
    def test_ar1_acf_rejects_invalid_parameters(self, phi, n_lags, message):
        with pytest.raises(ValueError, match=message):
            ar1_acf(phi, n_lags)

    def test_rejects_values_that_are_not_correlations(self):
        with pytest.raises(ValueError, match="between -1 and 1"):
            acf_generating_function([0.5, 1.5], 0.9)

    def test_rejects_a_sequence_that_starts_at_lag_zero(self):
        # what statsmodels' acf and most estimators return; taken as lags 1, 2, ...
        # it would be a process with rho(1) = 1 and every result would be wrong
        with pytest.raises(ValueError, match="must start at lag 1"):
            acf_generating_function([1.0, 0.5, 0.25], 0.9)
        with pytest.raises(ValueError, match="must start at lag 1"):
            european_sharpe_ratio(span_long=20, acf=np.r_[1.0, ar1_acf(0.1, 50)])


class TestMomentsAgainstDirectSums:
    @pytest.mark.parametrize("name", ACFS)
    @pytest.mark.parametrize("span", [1, 5, 21, 60])
    @pytest.mark.parametrize("sharpe", [0.0, 0.8, -0.5])
    def test_single_filter(self, name, span, sharpe):
        acf = ACFS[name]
        mean, variance = european_daily_moments(
            span_long=span, acf=acf, instrument_sharpe=sharpe, sigma_target_annual=TARGET, a=A
        )
        scale = TARGET / math.sqrt(A)
        expected_mean, expected_variance = direct_moments(
            single_kernel(span), acf, sharpe / math.sqrt(A)
        )
        assert mean == pytest.approx(scale * expected_mean, rel=1e-9, abs=1e-15)
        assert variance == pytest.approx(scale**2 * expected_variance, rel=1e-9)

    @pytest.mark.parametrize("name", ACFS)
    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(60, 5), (21, 10), (40, 1), (40, 39), (250, 249)]
    )
    @pytest.mark.parametrize("sharpe", [0.0, 0.8])
    def test_long_short_filter(self, name, span_long, span_short, sharpe):
        acf = ACFS[name]
        mean, variance = european_daily_moments(
            span_long=span_long,
            span_short=span_short,
            acf=acf,
            instrument_sharpe=sharpe,
            sigma_target_annual=TARGET,
            a=A,
        )
        scale = TARGET / math.sqrt(A)
        expected_mean, expected_variance = direct_moments(
            long_short_kernel(span_long, span_short), acf, sharpe / math.sqrt(A)
        )
        assert mean == pytest.approx(scale * expected_mean, rel=1e-8, abs=1e-15)
        assert variance == pytest.approx(scale**2 * expected_variance, rel=1e-8)


class TestPaperSpecialCases:
    @pytest.mark.parametrize("span", SPANS)
    def test_white_noise_without_drift_runs_at_the_target_and_earns_nothing(self, span):
        assert european_volatility(span_long=span) == pytest.approx(TARGET, rel=1e-13)
        assert european_volatility(span_long=span, span_short=3) == pytest.approx(TARGET, rel=1e-12)
        assert european_expected_return(span_long=span) == 0.0
        assert european_sharpe_ratio(span_long=span) == 0.0

    @pytest.mark.parametrize("span", SPANS)
    @pytest.mark.parametrize("sharpe", [-0.5, 0.5, 1.0])
    def test_white_noise_with_drift(self, span, sharpe):
        # expected return: (l target / sqrt(a)) mu_an**2, with l = sqrt(span)
        assert european_expected_return(span_long=span, instrument_sharpe=sharpe) == pytest.approx(
            math.sqrt(span) * TARGET / math.sqrt(A) * sharpe**2, rel=1e-12
        )
        # Sharpe ratio: mu_an**2 sqrt(span / a) / sqrt(1 + (mu_an**2 / a)(span + 1))
        expected = sharpe**2 * math.sqrt(span / A) / math.sqrt(1 + sharpe**2 / A * (span + 1))
        assert european_sharpe_ratio(span_long=span, instrument_sharpe=sharpe) == pytest.approx(
            expected, rel=1e-12
        )

    @pytest.mark.parametrize("span", SPANS)
    @pytest.mark.parametrize("phi", [-0.05, 0.05, 0.2])
    def test_ar1_without_drift(self, span, phi):
        nu = span_to_nu(span)
        loading = math.sqrt((1 + nu) / (1 - nu))
        acf = ar1_acf(phi, 400)
        # expected return: h [phi nu / (1 - phi nu)], h = l target sqrt(a) (1 - nu) / nu
        h = loading * TARGET * math.sqrt(A) * (1 - nu) / nu
        assert european_expected_return(span_long=span, acf=acf) == pytest.approx(
            h * phi * nu / (1 - phi * nu), rel=1e-12
        )
        a_nu = (1 - nu) * phi / (1 - nu * phi)
        b_nu = (1 - nu) * (1 + nu * phi) / ((1 + nu) * (1 - nu * phi))
        assert european_sharpe_ratio(span_long=span, acf=acf) == pytest.approx(
            math.sqrt(A) * a_nu / math.sqrt(b_nu + a_nu**2), rel=1e-12
        )

    def test_ar1_sharpe_ratio_is_close_to_its_small_phi_approximation(self):
        for span in SPANS:
            nu = span_to_nu(span)
            exact = european_sharpe_ratio(span_long=span, acf=ar1_acf(0.05, 400))
            assert exact == pytest.approx(0.05 * math.sqrt(A * (1 - nu**2)), rel=0.06)

    def test_trend_following_earns_from_autocorrelation_and_from_squared_drift(self):
        acf = ar1_acf(0.1, 400)
        both = european_expected_return(span_long=21, acf=acf, instrument_sharpe=0.7)
        from_acf = european_expected_return(span_long=21, acf=acf)
        from_drift = european_expected_return(span_long=21, instrument_sharpe=0.7)
        assert both == pytest.approx(from_acf + from_drift, rel=1e-12)
        assert european_expected_return(span_long=21, instrument_sharpe=-0.7) == pytest.approx(
            from_drift, rel=1e-12
        )
        assert european_expected_return(span_long=21, acf=ar1_acf(-0.1, 400)) < 0

    def test_sharpe_ratio_does_not_depend_on_the_target(self):
        acf = ar1_acf(0.1, 400)
        moments = [
            european_daily_moments(span_long=21, acf=acf, sigma_target_annual=target)
            for target in (0.05, 0.4)
        ]
        ratios = [mean / math.sqrt(variance) for mean, variance in moments]
        assert ratios[0] == pytest.approx(ratios[1], rel=1e-12)
        assert moments[1][0] == pytest.approx(8 * moments[0][0], rel=1e-12)


class TestLongShortClosedForms:
    """The long-short moments for an AR(1) process, and for spans that almost coincide."""

    @staticmethod
    def scale(nu_1, nu_2):
        """q (nu_1 - nu_2) in a form without the difference of the two parameters."""
        product = nu_1 * nu_2
        return math.sqrt((1 - nu_1**2) * (1 - nu_2**2) * (1 - product) / (1 + product))

    @pytest.mark.parametrize(("span_long", "span_short"), [(60, 5), (250, 20), (125, 1)])
    @pytest.mark.parametrize("phi", [-0.2, 0.05, 0.3])
    def test_expected_return_for_an_ar1_process(self, span_long, span_short, phi):
        # Cov[z_t, S_{t-1}] = q (Psi_1 / nu_1 - Psi_2 / nu_2) with Psi / nu = phi / (1 - nu phi):
        #   q (nu_1 - nu_2) phi**2 / ((1 - nu_1 phi)(1 - nu_2 phi))
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        q = reference.paper_q(nu_1, nu_2)
        expected = (
            math.sqrt(A) * TARGET * q * (nu_1 - nu_2) * phi**2
            / ((1 - nu_1 * phi) * (1 - nu_2 * phi))
        )  # fmt: skip
        assert european_expected_return(
            span_long=span_long, span_short=span_short, acf=ar1_acf(phi, 6000)
        ) == pytest.approx(expected, rel=1e-11)

    @pytest.mark.parametrize(
        ("span_long", "span_short"),
        [
            (60, 5),
            (250, 20),
            (125, 1),
            (250, 249),
            (250, 250 - 1e-6),
            (250, 250 - 1e-9),
            (250, 250 - 5e-12),
            (5, math.nextafter(5.0, 0.0)),
            (2.0, 1.9999999999999993),
            (1e6, 250),
        ],
    )
    @pytest.mark.parametrize("phi", [-0.9, -0.2, 0.05, 0.4])
    def test_variance_of_the_signal_for_an_ar1_process(self, span_long, span_short, phi):
        # Summing phi**m against the lagged products of the weights q (nu_1^k - nu_2^k):
        #   Var[S] = 1 + 2 phi (nu_1 + nu_2 - phi P (1 + P)) / ((1 + P)(1 - nu_1 phi)(1 - nu_2 phi))
        # with P = nu_1 nu_2. No term of it is a difference of nearby numbers, so it
        # is a reference also where the two spans are a few floats apart. With
        # white-noise variance one and no drift, Var[f] = scale**2 (Var[S] + Cov**2).
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        product = nu_1 * nu_2
        both = (1 - nu_1 * phi) * (1 - nu_2 * phi)
        signal_variance = 1 + 2 * phi * (nu_1 + nu_2 - phi * product * (1 + product)) / (
            (1 + product) * both
        )
        covariance = self.scale(nu_1, nu_2) * phi**2 / both
        _, variance = european_daily_moments(
            span_long=span_long, span_short=span_short, acf=ar1_acf(phi, 3000)
        )
        scale = TARGET / math.sqrt(A)
        assert variance == pytest.approx(scale**2 * (signal_variance + covariance**2), rel=1e-11)

    def test_a_single_lag_of_autocorrelation_earns_nothing(self):
        # the long-short filter has no weight on the latest return
        assert european_expected_return(span_long=60, span_short=5, acf=[0.3]) == 0.0
        assert european_expected_return(span_long=60, acf=[0.3]) > 0.0

    @pytest.mark.parametrize("gap", [1e-3, 1e-6, 1e-9])
    def test_spans_that_almost_coincide(self, gap):
        # q grows without bound as the spans approach, and every moment is q times
        # a difference that vanishes; the results must stay those of the limit
        span_long, span_short = 250.0, 250.0 - gap
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        assert european_volatility(span_long=span_long, span_short=span_short) == pytest.approx(
            TARGET, rel=1e-13
        )
        assert expected_signal_turnover(
            span_long=span_long, span_short=span_short
        ) == pytest.approx(0.248938, abs=1e-6)
        phi = 0.1
        acf = ar1_acf(phi, 6000)
        expected = (
            math.sqrt(A) * TARGET * self.scale(nu_1, nu_2) * phi**2
            / ((1 - nu_1 * phi) * (1 - nu_2 * phi))
        )  # fmt: skip
        assert european_expected_return(
            span_long=span_long, span_short=span_short, acf=acf
        ) == pytest.approx(expected, rel=1e-11)
        # and nothing jumps on the way to the limit
        near = european_daily_moments(
            span_long=250, span_short=249.99, acf=acf, instrument_sharpe=1
        )
        here = european_daily_moments(
            span_long=span_long, span_short=span_short, acf=acf, instrument_sharpe=1
        )
        assert here == pytest.approx(near, rel=1e-4)

    def test_spans_that_floats_cannot_tell_apart_are_rejected(self):
        with pytest.raises(ValueError, match="smaller than span_long"):
            european_daily_moments(span_long=250 + 1e-14, span_short=250)
        with pytest.raises(ValueError, match="smaller than span_long"):
            EuropeanTFConfig(span_long=250 + 1e-14, span_short=250)


class TestTurnover:
    def test_values_published_in_the_paper(self):
        # Section 4.4: 393 % a year for the single 250-day filter, 88 % for LS(250, 20)
        assert expected_signal_turnover(span_long=250) == pytest.approx(3.93, abs=0.005)
        assert expected_signal_turnover(span_long=250, span_short=20) == pytest.approx(
            0.88, abs=0.005
        )

    def test_single_filter_formula(self):
        for span in SPANS:
            assert expected_signal_turnover(span_long=span) == pytest.approx(
                2 * A / math.sqrt(math.pi) * TARGET * math.sqrt(2 / (span + 1)), rel=1e-12
            )

    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(250, 20), (125, 20), (60, 5), (21, 10), (10, 1)]
    )
    def test_long_short_loading_is_that_of_the_paper(self, span_long, span_short):
        # equation (4.17) as printed; the package uses 2 / (span_long * span_short + 1)
        nu_1, nu_2 = span_to_nu(span_long), span_to_nu(span_short)
        zeta = (
            reference.paper_q(nu_1, nu_2) ** 2
            / 2
            * (
                (1 - nu_1) / (1 + nu_1)
                + (1 - nu_2) / (1 + nu_2)
                - 2 * (1 - nu_1) * (1 - nu_2) / (1 - nu_1 * nu_2)
            )
        )
        assert expected_signal_turnover(
            span_long=span_long, span_short=span_short
        ) == pytest.approx(2 * A / math.sqrt(math.pi) * TARGET * math.sqrt(zeta), rel=1e-10)
        assert zeta == pytest.approx((1 - nu_1) * (1 - nu_2) / (1 + nu_1 * nu_2), rel=1e-10)

    def test_break_even_cost_of_an_ar1_alpha(self):
        # The cost per unit of turnover at which the expected return of an AR(1)
        # alpha equals the cost of its signal turnover (Proposition C.1):
        #   c*(eta) = sqrt(pi / 2a) phi sqrt(1 - eta / 2) / (1 - phi + eta phi),
        # eta = 2 / (span + 1).  The paper quotes 37 to 41 basis points for
        # phi = 0.05 between the one-week and the two-year span.
        phi = 0.05
        acf = ar1_acf(phi, 400)
        costs = []
        for span in SPANS:
            eta = 2 / (span + 1)
            cost = european_expected_return(span_long=span, acf=acf) / expected_signal_turnover(
                span_long=span
            )
            assert cost == pytest.approx(
                math.sqrt(math.pi / (2 * A)) * phi * math.sqrt(1 - eta / 2) / (1 - phi + eta * phi),
                rel=1e-12,
            )
            costs.append(cost)
        assert costs == sorted(costs)  # increasing in the span
        assert round(1e4 * costs[0]) == 37
        assert round(1e4 * costs[-1]) == 41

    @pytest.mark.parametrize(
        ("span_long", "span_short"), [(21, None), (250, None), (60, 5), (250, 20)]
    )
    def test_against_simulated_signals(self, span_long, span_short):
        rng = np.random.default_rng(17)
        z = rng.standard_normal(1_500_000)
        if span_short is None:
            signal = ewma_variance_preserving(z, span_to_nu(span_long))
        else:
            signal = long_short_variance_preserving(
                z, span_to_nu(span_long), span_to_nu(span_short)
            )
        simulated = A * TARGET * np.mean(np.abs(np.diff(signal[5000:])))
        # the increments of the long-short signal are correlated over the fast
        # span, which leaves a standard error of 0.3 %
        assert expected_signal_turnover(
            span_long=span_long, span_short=span_short
        ) == pytest.approx(simulated, rel=0.02)

    @pytest.mark.parametrize(
        ("mode", "low", "high"), [("single", 1.02, 1.06), ("longshort", 1.45, 1.75)]
    )
    def test_is_the_signal_part_of_the_turnover_of_the_system(self, mode, low, high):
        # The system also trades the changes of its volatility estimate. With a
        # 33-day volatility span the paper puts the full turnover about 4 % above
        # the proxy for a single filter and at 1.6 to 2.3 times the proxy for the
        # long-short filter, whose signal moves so little that those trades
        # dominate. Simulated here: 1.037 and 1.59 for the spans (250, 20).
        rng = np.random.default_rng(23)
        r = 0.01 * rng.standard_normal(600_000)
        cfg = EuropeanTFConfig(mode=mode, span_long=250, span_short=20)
        result = EuropeanTF(cfg).run_from_returns(r)
        valid = np.isfinite(result.volatility)
        turnover = volatility_weighted_turnover(
            result.weights[valid], result.volatility[valid], cfg.a
        )
        realised = cfg.a * turnover[3000:].mean()
        proxy = expected_signal_turnover(
            span_long=250, span_short=20 if mode == "longshort" else None
        )
        assert low < realised / proxy < high

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"span_long": 20, "span_short": 20}, "smaller than span_long"),
            ({"span_long": 0.5}, "span"),
            ({"span_long": 20, "sigma_target_annual": 0.0}, "sigma_target_annual"),
            ({"span_long": 20, "a": 0}, "a must"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            expected_signal_turnover(**kwargs)


def simulate_ar1(rng, n, phi):
    """Gaussian AR(1) with unit variance."""
    shocks = (math.sqrt(1 - phi**2) * rng.standard_normal(n)).tolist()
    out = [rng.standard_normal()]
    for shock in shocks[1:]:
        out.append(phi * out[-1] + shock)
    return np.asarray(out)


class TestAgainstSimulation:
    @pytest.mark.parametrize(
        ("phi", "sharpe", "span_long", "span_short"),
        [
            (0.1, 0.0, 21, None),
            (-0.1, 0.0, 10, None),
            (0.05, 1.0, 63, None),
            (0.4, 0.0, 20, 3),
            (0.0, 1.5, 60, 10),
        ],
    )
    def test_signal_times_normalised_return(self, phi, sharpe, span_long, span_short):
        # The identity f_t = (target / sqrt(a)) S_{t-1} z_t on simulated z, with no
        # volatility estimate in the way: the closed forms should hold exactly.
        rng = np.random.default_rng(1234)
        n = 1_200_000
        z = sharpe / math.sqrt(A) + simulate_ar1(rng, n, phi)
        if span_short is None:
            signal = ewma_variance_preserving(z, span_to_nu(span_long))
        else:
            signal = long_short_variance_preserving(
                z, span_to_nu(span_long), span_to_nu(span_short)
            )
        f = (TARGET / math.sqrt(A) * signal[:-1] * z[1:])[5000:]

        mean, variance = european_daily_moments(
            span_long=span_long,
            span_short=span_short,
            acf=ar1_acf(phi, 500) if phi else None,
            instrument_sharpe=sharpe,
        )

        def standard_error(values):
            """Of the mean of a serially dependent series, from 200 batch means."""
            batches = values[: values.size // 200 * 200].reshape(200, -1).mean(axis=1)
            return batches.std(ddof=1) / math.sqrt(200)

        error = standard_error(f)
        assert abs(f.mean() - mean) < 5 * error
        assert error < 0.2 * abs(mean)  # the comparison has power
        error = standard_error((f - mean) ** 2)
        assert abs(np.mean((f - mean) ** 2) - variance) < 5 * error
        assert error < 0.02 * variance

    def test_the_system_with_estimated_volatility_comes_close(self):
        # Returns with constant volatility. The system has to estimate it, which
        # dampens the autocorrelation of the normalised returns a little; the
        # paper reports a realised Sharpe ratio 2 % to 8 % below the closed form.
        # Measured against the signal on exactly normalised returns of the same
        # path, so that the sampling error of the two Sharpe ratios cancels: the
        # ratio is 0.974 with a standard error of 0.004.
        rng = np.random.default_rng(77)
        z = simulate_ar1(rng, 1_000_000, 0.1)
        cfg = EuropeanTFConfig(mode="single", span_long=21)
        pnl = EuropeanTF(cfg).run_from_returns(0.01 * z).pnl[5000:]
        signal = ewma_variance_preserving(z, span_to_nu(21))
        ideal = (TARGET / math.sqrt(A) * signal[:-1] * z[1:])[4999:]

        def sharpe(f):
            return math.sqrt(A) * f.mean() / f.std()

        assert 0.95 < sharpe(pnl) / sharpe(ideal) < 0.995
        # against the closed form itself the sampling error is 3 %
        acf = ar1_acf(0.1, 500)
        assert 0.80 < sharpe(pnl) / european_sharpe_ratio(span_long=21, acf=acf) < 1.15
        # and the volatility is 2 % above the closed form, again from the estimate
        realised = pnl.std() * math.sqrt(A)
        assert 1.0 < realised / european_volatility(span_long=21, acf=acf) < 1.05


class TestInput:
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"span_long": 0}, "span"),
            ({"span_long": 20, "span_short": 30}, "smaller than span_long"),
            ({"span_long": 20, "instrument_sharpe": math.nan}, "instrument_sharpe"),
            ({"span_long": 20, "sigma_target_annual": -0.1}, "sigma_target_annual"),
            ({"span_long": 20, "a": 0}, "a must"),
            ({"span_long": 20, "acf": [0.5, math.nan]}, "acf contains NaN"),
            ({"span_long": 20, "acf": [[0.5]]}, "acf must be one-dimensional"),
            ({"span_long": 20, "acf": [0.5j]}, "acf must be real"),
            ({"span_long": 20, "acf": ["0.5"]}, "acf must be numeric"),
            ({"span_long": 20, "acf": [1.0, 0.5]}, "acf must start at lag 1"),
            ({"span_long": 1e17}, "span_long is too large"),
        ],
    )
    def test_rejects_invalid_parameters(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            european_daily_moments(**kwargs)

    def test_rejects_a_sequence_that_is_not_an_autocorrelation_function(self):
        # rho(1) = -0.99, rho(2) = -0.99 is impossible: the variance of the signal
        # it implies is negative
        with pytest.raises(ValueError, match="not a valid autocorrelation"):
            european_daily_moments(span_long=3, acf=[-0.99, -0.99])
