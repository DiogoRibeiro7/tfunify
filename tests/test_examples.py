"""Every example script runs, from any directory, and returns sensible figures."""

from __future__ import annotations

import importlib.util
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def load(name, folder=EXAMPLES):
    spec = importlib.util.spec_from_file_location(f"example_{name}", folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def elsewhere(tmp_path, monkeypatch):
    """The examples must not depend on the working directory."""
    monkeypatch.chdir(tmp_path)


def test_every_example_is_covered_here():
    scripts = sorted(path.stem for path in EXAMPLES.glob("*.py"))
    assert scripts == [
        "basic_usage",
        "parameter_optimization",
        "performance_comparison",
        "portfolio_integration",
        "real_data_analysis",
    ]


def test_basic_usage(capsys):
    results = load("basic_usage").main(["--days", "1500"])
    assert len(results) == 3
    for pnl in results.values():
        assert pnl.shape == (1500,)
        assert np.all(np.isfinite(pnl))
        assert np.any(pnl != 0.0)
    printed = capsys.readouterr().out
    assert "European LS(250, 20)" in printed
    assert "correlation of daily returns" in printed


def test_basic_usage_saves_its_chart_next_to_the_script(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    shutil.copy(EXAMPLES / "basic_usage.py", tmp_path / "basic_usage.py")
    load("basic_usage", tmp_path).main(["--days", "800", "--plot"])
    assert (tmp_path / "basic_usage.png").stat().st_size > 10_000
    assert f"chart saved to {tmp_path / 'basic_usage.png'}" in capsys.readouterr().out


def test_performance_comparison(capsys):
    table = load("performance_comparison").main(["--paths", "10", "--years", "12"])
    assert list(table) == [
        "random walk",
        "trending, phi = 0.10",
        "mean reverting, phi = -0.10",
        "drift, Sharpe 1.0",
    ]
    for row in table.values():
        assert set(row) == {"closed form", "European", "American", "TSMOM"}
        assert all(math.isfinite(value) for value in row.values())
    assert table["random walk"]["closed form"] == 0.0
    assert table["trending, phi = 0.10"]["closed form"] == pytest.approx(0.67, abs=0.01)
    assert table["mean reverting, phi = -0.10"]["closed form"] < 0
    # 120 years of trending returns in all: the mean Sharpe ratio, 0.63 in
    # expectation, has a standard error of 0.09
    assert table["trending, phi = 0.10"]["European"] > 0.2
    assert table["mean reverting, phi = -0.10"]["European"] < -0.2
    assert "closed form" in capsys.readouterr().out


def test_parameter_optimization(capsys):
    table = load("parameter_optimization").main(["--years", "4", "--phi", "0.1", "--sharpe", "0"])
    assert set(table) == {"in sample", "out of sample", "closed form"}
    assert all(len(column) == 6 for column in table.values())
    # without drift the closed-form Sharpe ratio of an AR(1) alpha falls with the span
    assert table["closed form"] == sorted(table["closed form"], reverse=True)
    printed = capsys.readouterr().out
    assert "Best span in sample" in printed
    assert "standard error of about 0.50" in printed  # sqrt(260 / (4 * 260))


def test_portfolio_integration(capsys):
    portfolios = load("portfolio_integration").main(["--years", "6", "--sleeve", "0.25"])
    assert list(portfolios) == ["60/40", "trend sleeve", "60/40 + 0.25 x sleeve"]
    np.testing.assert_allclose(
        portfolios["60/40 + 0.25 x sleeve"],
        portfolios["60/40"] + 0.25 * portfolios["trend sleeve"],
    )
    assert "worst 16 % of quarters" in capsys.readouterr().out


@pytest.fixture
def price_file(tmp_path):
    rng = np.random.default_rng(5)
    close = 100 * np.exp(np.cumsum(0.0003 + 0.011 * rng.standard_normal(900)))
    lines = ["close,high,low"]
    lines += [f"{c!r},{c * 1.004!r},{c * 0.996!r}" for c in close.tolist()]
    path = tmp_path / "prices.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_real_data_analysis_on_a_file(price_file, capsys):
    table = load("real_data_analysis").main(["--csv", str(price_file)])
    assert list(table) == ["European", "American", "TSMOM"]
    for row in table.values():
        assert all(math.isfinite(value) for value in row.values())
        assert 0.0 <= row["in market"] <= 1.0
        assert row["turnover"] >= 0.0
    # continuous positions trade more than positions held until a stop
    assert table["European"]["turnover"] > table["American"]["turnover"]
    assert "900 days" in capsys.readouterr().out


def test_real_data_analysis_explains_a_missing_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", None)
    with pytest.raises(SystemExit, match=r'pip install "tfunify\[yahoo\]"'):
        load("real_data_analysis").main(["--ticker", "SPY"])


def test_real_data_analysis_explains_a_missing_file(tmp_path):
    with pytest.raises(SystemExit, match="CSV file not found"):
        load("real_data_analysis").main(["--csv", str(tmp_path / "none.csv")])
