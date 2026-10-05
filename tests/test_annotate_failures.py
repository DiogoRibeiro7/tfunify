"""The script CI uses to show test failures as annotations on the run."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "annotate_failures.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="needs a source checkout")

REPORT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest" errors="1" failures="1" tests="4">
<testcase classname="tests.test_a.TestX" name="test_ok" time="0.01"/>
<testcase classname="tests.test_a.TestX" name="test_bad[1e-07, x]" time="0.01">
<failure message="assert 1 == 2">def test_bad():
&gt;       assert 1 == 2
E       assert 1 == 2 (100% off)

tests/test_a.py:3: AssertionError</failure></testcase>
<testcase classname="tests.test_b" name="test_setup" time="0.0">
<error message="failed on setup">fixture 'nope' not found</error></testcase>
<testcase classname="tests.test_b" name="test_skipped"><skipped message="why"/></testcase>
</testsuite></testsuites>
"""


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("annotate_failures", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_failures_and_errors_are_found_and_passes_and_skips_are_not(tool):
    found = tool.failures(REPORT)
    assert [identifier for identifier, _ in found] == [
        "tests.test_a.TestX::test_bad[1e-07, x]",
        "tests.test_b::test_setup",
    ]
    assert "assert 1 == 2" in found[0][1]
    assert found[1][1] == "fixture 'nope' not found"


def test_commands_are_escaped(tool):
    first, second = tool.annotations(REPORT)
    title, message = first.removeprefix("::error title=").split("::", 1)
    # in a property, ':' and ',' would end the property; they are encoded
    assert title == "tests.test_a.TestX%3A%3Atest_bad[1e-07%2C x]"
    assert "\n" not in message
    assert "%0A" in message
    assert "(100%25 off)" in message
    assert second.startswith("::error title=tests.test_b%3A%3Atest_setup::")


def test_long_tracebacks_keep_their_end(tool):
    body = "\n".join(f"line {i}" for i in range(200))
    report = (
        '<testsuite><testcase classname="c" name="n">'
        f"<failure>{body}</failure></testcase></testsuite>"
    )
    ((_, details),) = tool.failures(report)
    lines = details.splitlines()
    assert lines[0] == "..."
    assert lines[-1] == "line 199"
    assert len(lines) == tool.MAX_LINES + 1


def test_more_failures_than_github_shows(tool):
    cases = "".join(
        f'<testcase classname="c" name="t{i}"><failure>boom</failure></testcase>' for i in range(14)
    )
    commands = tool.annotations(f"<testsuite>{cases}</testsuite>")
    assert len(commands) == 10
    assert commands[-1].startswith("::error title=5 more failed test(s)::c::t9, c::t10")


def test_command_line(tmp_path):
    report = tmp_path / "report.xml"
    report.write_text(REPORT, encoding="utf-8")
    run = subprocess.run(
        [sys.executable, str(SCRIPT), str(report)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert run.returncode == 0
    assert run.stdout.count("::error ") == 2
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), str(tmp_path / "none.xml")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert missing.returncode == 0
    assert missing.stdout.startswith("::warning::no test report")
    # a run that was killed leaves half a report: say so, and do not fail the
    # step that is only there to explain a failure
    report.write_text(REPORT[: len(REPORT) // 2], encoding="utf-8")
    broken = subprocess.run(
        [sys.executable, str(SCRIPT), str(report)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert broken.returncode == 0, broken.stderr
    assert broken.stdout.startswith("::warning::the test report at ")
    assert "could not be read" in broken.stdout
    assert broken.stdout.count("\n") == 1
    assert "Traceback" not in broken.stderr
    usage = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert usage.returncode == 2
