"""Summary statistics on series small enough to work out by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tfunify import PerformanceSummary, performance_summary
from tfunify.metrics import annualised_return, annualised_volatility, max_drawdown, sharpe_ratio

PNL = np.array([0.01, -0.02, 0.03, 0.00])  # mean 0.005, sample variance 0.0013 / 3


def test_annualised_return():
    assert annualised_return(PNL, 260) == pytest.approx(1.3)
    assert annualised_return(PNL, 100) == pytest.approx(0.5)


def test_annualised_volatility():
    sample = math.sqrt(0.0013 / 3)
    assert annualised_volatility(PNL, 260) == pytest.approx(math.sqrt(260) * sample)
    assert annualised_volatility(PNL, 260, ddof=0) == pytest.approx(
        math.sqrt(260) * math.sqrt(0.0013 / 4)
    )


def test_sharpe_ratio_is_the_ratio_of_daily_moments():
    assert sharpe_ratio(PNL, 260) == pytest.approx(math.sqrt(260) * 0.005 / math.sqrt(0.0013 / 3))
    # scaling the returns leaves it unchanged; flipping them flips it
    assert sharpe_ratio(7 * PNL) == pytest.approx(sharpe_ratio(PNL))
    assert sharpe_ratio(-PNL) == pytest.approx(-sharpe_ratio(PNL))


def test_sharpe_ratio_of_constant_returns_is_undefined():
    assert math.isnan(sharpe_ratio([0.0, 0.0, 0.0]))
    assert math.isnan(sharpe_ratio([0.01, 0.01, 0.01]))


@pytest.mark.parametrize("value", [0.1, 0.07, 1 / 3, -0.0123, 1e-9])
@pytest.mark.parametrize("n", [3, 7, 10, 1000])
def test_constant_returns_have_no_volatility(value, n):
    # the deviation NumPy computes for such a series is a few 1e-17, not zero,
    # and a mean divided by it would be a Sharpe ratio of 1e16
    pnl = np.full(n, value)
    assert annualised_volatility(pnl) == 0.0
    assert math.isnan(sharpe_ratio(pnl))
    summary = performance_summary(pnl)
    assert summary.annual_volatility == 0.0
    assert math.isnan(summary.sharpe_ratio)
    assert "Sharpe ratio nan" in str(summary)


@pytest.mark.parametrize(
    ("pnl", "expected"),
    [
        ([0.01, 0.02, 0.03], 0.0),  # never falls
        ([-0.01, -0.02], 0.03),  # falls from the start: measured from zero
        ([0.05, -0.02, -0.04, 0.03, -0.01], 0.06),  # peak 0.05, trough -0.01
        ([0.05, -0.02, 0.10, -0.03, -0.05, 0.01], 0.08),  # the later fall is the larger
        ([0.0], 0.0),
    ],
)
def test_max_drawdown(pnl, expected):
    assert max_drawdown(pnl) == pytest.approx(expected, abs=1e-15)


def test_performance_summary():
    summary = performance_summary(PNL, 260)
    assert isinstance(summary, PerformanceSummary)
    assert summary.annual_return == pytest.approx(1.3)
    assert summary.annual_volatility == pytest.approx(annualised_volatility(PNL))
    assert summary.sharpe_ratio == pytest.approx(sharpe_ratio(PNL))
    assert summary.max_drawdown == pytest.approx(0.02)
    assert summary.n_observations == 4


def test_summary_prints_in_plain_ascii():
    text = str(performance_summary(PNL))
    assert text == (
        "annual return 130.00%, annual volatility 33.57%, Sharpe ratio 3.87, "
        "maximum drawdown 2.00% (4 days)"
    )
    text.encode("ascii")  # a Windows console in cp1252 can print it


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (annualised_return, ([],), "at least 1"),
        (annualised_return, ([0.1, math.nan],), "NaN"),
        (annualised_return, ([0.1], 0), "a must"),
        (annualised_volatility, ([0.1],), "at least 2"),
        (annualised_volatility, ([[0.1, 0.2]],), "one-dimensional"),
        (sharpe_ratio, ([0.1],), "at least 2"),
        (max_drawdown, ([],), "at least 1"),
        (performance_summary, ([0.1],), "at least 2"),
        (performance_summary, ([0.1, 0.2], -1), "a must"),
    ],
)
def test_rejects_invalid_input(function, args, message):
    with pytest.raises(ValueError, match=message):
        function(*args)


def test_rejects_invalid_ddof():
    with pytest.raises(ValueError, match="ddof"):
        annualised_volatility([0.1, 0.2], ddof=-1)
    assert annualised_volatility([0.1], ddof=0) == 0.0
