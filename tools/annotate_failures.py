"""Turn the failures in a JUnit XML report into GitHub Actions annotations.

Used by the CI workflow after a failed test run::

    python tools/annotate_failures.py test-results.xml

Each failed test becomes an ``::error`` workflow command, which GitHub shows
on the summary page of the run and on the pull request, so that the cause of
a failure can be read without opening the log of the job.  GitHub displays at
most ten error annotations per step; the rest are summarised in one line.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

MAX_ANNOTATIONS = 9  # one is kept for the summary of what was left out
MAX_LINES = 30


def _escape(text: str, *, is_property: bool = False) -> str:
    """Escape a value for a workflow command (GitHub's own rules)."""
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if is_property:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def failures(report: str) -> list[tuple[str, str]]:
    """``(test id, details)`` for every failed or errored test case in the report."""
    found = []
    for case in ET.fromstring(report).iter("testcase"):
        for outcome in case:
            if outcome.tag not in ("failure", "error"):
                continue
            identifier = f"{case.get('classname', '')}::{case.get('name', '')}".strip(":")
            lines = (outcome.text or outcome.get("message") or "").strip().splitlines()
            if len(lines) > MAX_LINES:  # the assertion is at the end of the traceback
                lines = ["...", *lines[-MAX_LINES:]]
            found.append((identifier, "\n".join(lines) or (outcome.get("message") or "")))
    return found


def annotations(report: str) -> list[str]:
    """The workflow commands to print for a report."""
    found = failures(report)
    commands = [
        f"::error title={_escape(identifier, is_property=True)}::{_escape(details)}"
        for identifier, details in found[:MAX_ANNOTATIONS]
    ]
    if len(found) > MAX_ANNOTATIONS:
        rest = ", ".join(identifier for identifier, _ in found[MAX_ANNOTATIONS:])
        commands.append(
            f"::error title={len(found) - MAX_ANNOTATIONS} more failed test(s)::{_escape(rest)}"
        )
    return commands


def main(arguments: list[str]) -> int:
    if len(arguments) != 1:
        print("usage: annotate_failures.py REPORT.xml", file=sys.stderr)
        return 2
    path = Path(arguments[0])
    if not path.is_file():
        print(f"::warning::no test report at {path}: the tests did not get as far as running")
        return 0
    try:
        commands = annotations(path.read_text(encoding="utf-8"))
    except (ET.ParseError, UnicodeDecodeError) as error:
        # a run that was killed leaves half a report; the failed step already
        # marks the job, so this one only says why there are no annotations
        print(f"::warning::the test report at {path} could not be read: {_escape(str(error))}")
        return 0
    for command in commands:
        print(command)
    return 0


if __name__ == "__main__":
    # UTF-8 whatever the code page of the platform: on Windows the default one
    # cannot encode every character a changelog or a traceback may contain.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
