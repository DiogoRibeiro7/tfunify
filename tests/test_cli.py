"""The ``tfu`` command, run in process and as an installed program."""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import io
import os
import subprocess
import sys

import numpy as np
import pytest

import tfunify
from tfunify import TSMOM, AmericanTF, AmericanTFConfig, EuropeanTF, EuropeanTFConfig, TSMOMConfig
from tfunify.cli import build_parser, main
from tfunify.data import download_csv


@pytest.fixture(autouse=True)
def plain_output(monkeypatch):
    """No colours in the help and the usage messages (Python 3.14 has them)."""
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def subcommands(parser):
    """The parsers of the commands, by name."""
    (action,) = (a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return action.choices


@pytest.fixture
def prices_csv(tmp_path):
    """A file with 400 days of open, high, low and close."""
    rng = np.random.default_rng(31)
    close = 100 * np.exp(np.cumsum(0.0005 + 0.01 * rng.standard_normal(400)))
    high = close * (1 + 0.004 * rng.random(400))
    low = close * (1 - 0.004 * rng.random(400))
    path = tmp_path / "prices.csv"
    lines = ["date,open,high,low,close,volume"]
    for day, (c, h, lo) in enumerate(zip(close, high, low)):
        date = dt.date(2020, 1, 1) + dt.timedelta(days=day)
        lines.append(f"{date},{float(c)!r},{float(h)!r},{float(lo)!r},{float(c)!r},1000")
    path.write_text("\n".join(lines) + "\n")
    return path, close, high, low


class TestSystems:
    def test_european(self, prices_csv, tmp_path, capsys):
        path, close, _, _ = prices_csv
        out = tmp_path / "eu.npz"
        code = main(
            ["european", "--csv", str(path), "--out", str(out), "--span-long", "50",
             "--span-short", "10", "--target", "0.2"]
        )  # fmt: skip
        assert code == 0
        expected = EuropeanTF(
            EuropeanTFConfig(span_long=50, span_short=10, sigma_target_annual=0.2)
        ).run_from_prices(close)
        with np.load(out) as saved:
            assert sorted(saved.files) == ["pnl", "signal", "volatility", "weights"]
            np.testing.assert_array_equal(saved["pnl"], expected.pnl)
            np.testing.assert_array_equal(saved["weights"], expected.weights)
            np.testing.assert_array_equal(saved["signal"], expected.signal)
            np.testing.assert_array_equal(saved["volatility"], expected.volatility)
        printed = capsys.readouterr().out
        assert printed.startswith("European TF: annual return ")
        assert f"results written to {out}" in printed

    def test_european_single_filter(self, prices_csv, tmp_path):
        path, close, _, _ = prices_csv
        out = tmp_path / "eu.npz"
        assert main(["european", "--csv", str(path), "--out", str(out), "--mode", "single",
                     "--span-long", "30", "-a", "252", "--span-sigma", "15"]) == 0  # fmt: skip
        expected = EuropeanTF(
            EuropeanTFConfig(mode="single", span_long=30, a=252, span_sigma=15)
        ).run_from_prices(close)
        with np.load(out) as saved:
            np.testing.assert_array_equal(saved["pnl"], expected.pnl)

    def test_longshort_flag_is_still_accepted(self, prices_csv, tmp_path):
        path, _, _, _ = prices_csv
        default, flagged = tmp_path / "a.npz", tmp_path / "b.npz"
        common = ["european", "--csv", str(path), "--span-long", "50", "--span-short", "10"]
        assert main([*common, "--out", str(default)]) == 0
        assert main([*common, "--out", str(flagged), "--longshort"]) == 0
        with np.load(default) as a, np.load(flagged) as b:
            np.testing.assert_array_equal(a["pnl"], b["pnl"])

    def test_american_uses_high_and_low(self, prices_csv, tmp_path, capsys):
        path, close, high, low = prices_csv
        out = tmp_path / "am.npz"
        code = main(
            ["american", "--csv", str(path), "--out", str(out), "--span-long", "40",
             "--span-short", "8", "--atr-period", "10", "--q", "1", "--p", "2",
             "--r-multiple", "0.02"]
        )  # fmt: skip
        assert code == 0
        cfg = AmericanTFConfig(span_long=40, span_short=8, atr_period=10, q=1, p=2, r_multiple=0.02)
        expected = AmericanTF(cfg).run(close, high, low)
        with np.load(out) as saved:
            assert sorted(saved.files) == [
                "atr", "fast", "pnl", "position", "slow", "stop", "weights",
            ]  # fmt: skip
            np.testing.assert_array_equal(saved["pnl"], expected.pnl)
            np.testing.assert_array_equal(saved["position"], expected.position)
            np.testing.assert_array_equal(saved["stop"], expected.stop)
        assert np.any(expected.position != 0)
        assert capsys.readouterr().out.startswith("American TF: annual return ")

    def test_tsmom(self, prices_csv, tmp_path, capsys):
        path, close, _, _ = prices_csv
        out = tmp_path / "ts.npz"
        code = main(
            ["tsmom", "--csv", str(path), "--out", str(out), "--L", "5", "--M", "8",
             "--signs", "period", "--span-sigma", "20", "--target", "0.1"]
        )  # fmt: skip
        assert code == 0
        expected = TSMOM(
            TSMOMConfig(L=5, M=8, signs="period", span_sigma=20, sigma_target_annual=0.1)
        ).run_from_prices(close)
        with np.load(out) as saved:
            assert sorted(saved.files) == ["pnl", "rebalance", "signal", "volatility", "weights"]
            np.testing.assert_array_equal(saved["pnl"], expected.pnl)
            np.testing.assert_array_equal(saved["rebalance"], expected.rebalance)
        assert capsys.readouterr().out.startswith("TSMOM: annual return ")

    @pytest.mark.parametrize(
        ("command", "default"),
        [
            ("european", "european_results.npz"),
            ("american", "american_results.npz"),
            ("tsmom", "tsmom_results.npz"),
        ],
    )
    def test_default_output_file(self, prices_csv, tmp_path, monkeypatch, command, default):
        path, _, _, _ = prices_csv
        monkeypatch.chdir(tmp_path)
        assert main([command, "--csv", str(path)]) == 0
        assert (tmp_path / default).is_file()

    @pytest.mark.parametrize("name", ["results", "results.dat", "results.npz"])
    def test_results_go_to_the_file_that_was_named(self, prices_csv, tmp_path, capsys, name):
        # given a name, NumPy appends ".npz" to it; the message would then point
        # at a file that does not exist
        path, _, _, _ = prices_csv
        out = tmp_path / "out" / name
        out.parent.mkdir()
        assert main(["american", "--csv", str(path), "--out", str(out)]) == 0
        assert [entry.name for entry in out.parent.iterdir()] == [name]
        assert f"results written to {out}" in capsys.readouterr().out
        with np.load(out) as saved:
            assert "pnl" in saved.files

    def test_output_that_is_not_a_console(self, prices_csv, tmp_path, monkeypatch):
        # a notebook or a test harness replaces sys.stdout by an object that
        # has none of the methods of a console stream
        path, _, _, _ = prices_csv
        stream = io.StringIO()
        monkeypatch.setattr(sys, "stdout", stream)
        assert main(["european", "--csv", str(path), "--out", str(tmp_path / "eu.npz")]) == 0
        assert stream.getvalue().startswith("European TF: annual return ")

    def test_a_path_the_console_cannot_show(self, prices_csv, tmp_path):
        # the run has succeeded when the message is printed; an output encoding
        # that lacks a character of the path must not turn it into a traceback
        path, _, _, _ = prices_csv
        out = tmp_path / "r\u00e9sultats \u03c3.npz"
        done = subprocess.run(
            [sys.executable, "-m", "tfunify", "tsmom", "--csv", str(path), "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, "PYTHONIOENCODING": "ascii"},
        )
        assert done.returncode == 0, done.stderr
        assert "r\\xe9sultats \\u03c3.npz" in done.stdout
        assert out.is_file()


class TestErrors:
    def test_missing_file(self, tmp_path, capsys):
        assert main(["european", "--csv", str(tmp_path / "none.csv")]) == 1
        captured = capsys.readouterr()
        assert captured.out == ""
        assert "tfu: error: CSV file not found" in captured.err

    def test_invalid_data(self, tmp_path, capsys):
        bad = tmp_path / "bad.csv"
        bad.write_text("close\n10\nabc\n")
        assert main(["tsmom", "--csv", str(bad)]) == 1
        assert "line 3: column 'close' holds 'abc'" in capsys.readouterr().err

    def test_invalid_parameters(self, prices_csv, tmp_path, capsys):
        path, _, _, _ = prices_csv
        out = tmp_path / "x.npz"
        code = main(["european", "--csv", str(path), "--out", str(out), "--span-long", "10",
                     "--span-short", "20"])  # fmt: skip
        assert code == 1
        assert "span_short must be smaller than span_long" in capsys.readouterr().err
        assert not out.exists()

    def test_series_too_short(self, tmp_path, capsys):
        short = tmp_path / "short.csv"
        short.write_text("close\n" + "\n".join(str(100 + i) for i in range(50)) + "\n")
        assert main(["tsmom", "--csv", str(short), "--out", str(tmp_path / "x.npz")]) == 1
        assert "M * L = 100 returns, got 49" in capsys.readouterr().err

    def test_download_without_yfinance(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setitem(sys.modules, "yfinance", None)  # makes the import fail
        assert main(["download", "SPY", "--out", str(tmp_path / "spy.csv")]) == 1
        assert 'pip install "tfunify[yahoo]"' in capsys.readouterr().err

    def test_download(self, tmp_path, capsys, monkeypatch):
        calls = []

        def fake_download(ticker, path, **kwargs):
            calls.append((ticker, str(path), kwargs))
            return path

        monkeypatch.setattr("tfunify.cli.download_csv", fake_download)
        out = tmp_path / "spy.csv"
        assert main(["download", "SPY", "--out", str(out), "--period", "2y", "--adjusted"]) == 0
        assert calls == [("SPY", str(out), {"period": "2y", "interval": "1d", "auto_adjust": True})]
        assert capsys.readouterr().out == f"SPY written to {out}\n"
        assert main(["download", "ES=F", "--interval", "1wk"]) == 0
        assert calls[1] == (
            "ES=F",
            "data.csv",
            {"period": "5y", "interval": "1wk", "auto_adjust": False},
        )

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["european"],  # --csv is required
            ["nonsense"],
            ["european", "--csv", "x.csv", "--mode", "triple"],
            ["tsmom", "--csv", "x.csv", "--L", "2.5"],
            ["tsmom", "--csv", "x.csv", "--signs", "weekly"],
            ["american", "--csv", "x.csv", "--atr-period", "ten"],
            ["download", "SPY", "--interval", "5m"],
            ["download"],  # the symbol is required
        ],
    )
    def test_usage_errors_exit_with_status_two(self, argv, capsys):
        with pytest.raises(SystemExit) as raised:
            main(argv)
        assert raised.value.code == 2
        assert "usage: tfu" in capsys.readouterr().err


class TestProgram:
    def test_help_lists_the_commands(self, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["--help"])
        assert raised.value.code == 0
        text = capsys.readouterr().out
        for command in ("european", "american", "tsmom", "download"):
            assert command in text

    def test_defaults_are_those_of_the_library(self):
        parser = build_parser()
        european = parser.parse_args(["european", "--csv", "x"])
        cfg = EuropeanTFConfig()
        assert (european.mode, european.span_long, european.span_short, european.span_sigma) == (
            cfg.mode, cfg.span_long, cfg.span_short, cfg.span_sigma,
        )  # fmt: skip
        assert (european.target, european.a) == (cfg.sigma_target_annual, cfg.a)
        american = parser.parse_args(["american", "--csv", "x"])
        acfg = AmericanTFConfig()
        assert (american.span_long, american.span_short, american.atr_period) == (
            acfg.span_long, acfg.span_short, acfg.atr_period,
        )  # fmt: skip
        assert (american.q, american.p, american.r_multiple) == (acfg.q, acfg.p, acfg.r_multiple)
        tsmom = parser.parse_args(["tsmom", "--csv", "x"])
        tcfg = TSMOMConfig()
        assert (tsmom.L, tsmom.M, tsmom.signs, tsmom.span_sigma) == (
            tcfg.L, tcfg.M, tcfg.signs, tcfg.span_sigma,
        )  # fmt: skip
        assert (tsmom.target, tsmom.a) == (tcfg.sigma_target_annual, tcfg.a)
        download = parser.parse_args(["download", "SPY"])
        library = inspect.signature(download_csv).parameters
        assert (download.period, download.interval, download.adjusted) == (
            library["period"].default, library["interval"].default, library["auto_adjust"].default,
        )  # fmt: skip

    def test_every_option_is_described_with_its_default(self):
        for name, command in subcommands(build_parser()).items():
            for action in command._actions:
                if not action.option_strings or isinstance(action, argparse._HelpAction):
                    continue
                where = f"tfu {name} {action.option_strings[0]}"
                assert action.help, where
                if action.default not in (None, False):
                    assert "%(default)s" in action.help, where

    @pytest.mark.parametrize("command", ["european", "american", "tsmom", "download"])
    def test_help_of_a_command(self, command, capsys):
        with pytest.raises(SystemExit) as raised:
            main([command, "--help"])
        assert raised.value.code == 0
        text = capsys.readouterr().out
        assert f"usage: tfu {command}" in text
        assert "(default: " in text

    def test_version(self, capsys):
        with pytest.raises(SystemExit) as raised:
            main(["--version"])
        assert raised.value.code == 0
        assert capsys.readouterr().out.strip() == f"tfunify {tfunify.__version__}"

    def test_runs_as_a_module(self, prices_csv, tmp_path):
        path, _, _, _ = prices_csv
        out = tmp_path / "eu.npz"
        done = subprocess.run(
            [sys.executable, "-m", "tfunify", "european", "--csv", str(path), "--out", str(out)],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 0, done.stderr
        assert done.stdout.startswith("European TF: annual return ")
        assert out.is_file()

    def test_module_reports_errors_with_status_one(self, tmp_path):
        done = subprocess.run(
            [sys.executable, "-m", "tfunify", "american", "--csv", str(tmp_path / "none.csv")],
            capture_output=True,
            text=True,
            check=False,
        )
        assert done.returncode == 1
        assert "tfu: error: CSV file not found" in done.stderr
        assert "Traceback" not in done.stderr
