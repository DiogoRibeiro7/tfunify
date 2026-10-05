"""The documentation pages: their code runs and the output they quote is real."""

from __future__ import annotations

import datetime as dt
import importlib.util
import re
from pathlib import Path

import numpy as np
import pytest

from tfunify.cli import main as tfu

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PAGES = sorted(DOCS.rglob("*.md")) if DOCS.is_dir() else []
WITH_CODE = [page for page in PAGES if "```python" in page.read_text(encoding="utf-8")]

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="needs a source checkout")


def page_id(page):
    return page.relative_to(DOCS).as_posix()  # the same on Windows


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A directory with the files the pages refer to: prices.csv and a results file."""
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(8)
    close = 100 * np.exp(np.cumsum(0.0003 + 0.01 * rng.standard_normal(600)))
    rows = ["date,open,high,low,close,volume"]
    start = dt.date(2020, 1, 1)
    rows += [
        f"{start + dt.timedelta(days=day)},{c!r},{c * 1.003!r},{c * 0.997!r},{c!r},100"
        for day, c in enumerate(close.tolist())
    ]
    (tmp_path / "prices.csv").write_text("\n".join(rows) + "\n")
    assert tfu(["european", "--csv", "prices.csv"]) == 0
    return tmp_path


def test_the_pages_with_code_are_the_expected_ones():
    assert [page_id(page) for page in WITH_CODE] == [
        "cli.md",
        "getting-started.md",
        "index.md",
        "systems/american.md",
        "systems/conventions.md",
        "systems/european.md",
        "systems/tsmom.md",
        "theory.md",
    ]


@pytest.mark.parametrize("page", WITH_CODE, ids=page_id)
def test_code_runs_and_prints_what_its_comments_say(page, workspace, capsys):
    text = page.read_text(encoding="utf-8")
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    capsys.readouterr()
    namespace: dict[str, object] = {"__name__": "__docs__"}
    for number, block in enumerate(blocks):
        exec(compile(block, f"{page_id(page)} block {number}", "exec"), namespace)
    printed = [line.strip() for line in capsys.readouterr().out.splitlines()]
    # comments that begin with a digit or a bracket quote the output
    quoted = re.findall(r"#\s+([\[\(\-\d][^\n]*)$", "\n".join(blocks), flags=re.M)
    for value in quoted:
        assert value.strip() in printed, f"{page_id(page)}: {value!r} is not printed"


def load_example(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "examples" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quoted_output(page, marker):
    """The lines of the ```text block of a page that contains ``marker``."""
    text = (DOCS / page).read_text(encoding="utf-8")
    for block in re.findall(r"```text\n(.*?)```", text, flags=re.S):
        if marker in block:
            return [line.rstrip() for line in block.splitlines() if line.strip()]
    raise AssertionError(f"{page} has no text block with {marker!r}")


def test_theory_page_quotes_the_output_of_the_comparison_example(capsys):
    load_example("performance_comparison").main([])
    printed = [line.rstrip() for line in capsys.readouterr().out.splitlines()]
    for line in quoted_output("theory.md", "random walk"):
        assert line in printed, line


def test_examples_page_quotes_the_output_of_the_span_example(capsys):
    load_example("parameter_optimization").main([])
    out = capsys.readouterr().out
    printed = [line.rstrip() for line in out.splitlines()]
    for line in quoted_output("examples.md", "in sample"):
        assert line in printed, line
    # the figures in the paragraph under the table
    assert "Sharpe ratio 1.00; the same span out of sample: 0.71; closed form: 0.63" in out
    assert "standard error of about 0.26" in out


def test_examples_page_lists_every_script():
    text = (DOCS / "examples.md").read_text(encoding="utf-8")
    for script in (ROOT / "examples").glob("*.py"):
        assert f"`{script.name}`" in text, script.name


def test_command_line_page_documents_every_option():
    from tfunify.cli import build_parser

    text = (DOCS / "cli.md").read_text(encoding="utf-8")
    parser = build_parser()
    commands = next(a for a in parser._actions if a.dest == "command").choices
    for name, sub in commands.items():
        assert f"`tfu {name}" in text, name
        for action in sub._actions:
            for option in action.option_strings:
                if option in ("-h", "--help"):
                    continue
                assert f"`{option}" in text, f"{name} {option}"


def test_reference_lists_every_public_name():
    import tfunify
    from tfunify import data, metrics, theory

    index = (DOCS / "reference" / "index.md").read_text(encoding="utf-8")
    for name in tfunify.__all__:
        if name in (
            "__version__",
            "data",
            "metrics",
            "theory",
            "PerformanceSummary",
            "performance_summary",
        ):
            continue
        assert f"[tfunify.{name}]" in index, name
    for module in (theory, metrics, data):
        for name in module.__all__:
            assert f"[{module.__name__}.{name}]" in index, f"{module.__name__}.{name}"
