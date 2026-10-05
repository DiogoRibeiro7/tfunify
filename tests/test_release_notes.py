"""The script the release workflow uses to turn the changelog into release notes."""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "release_notes.py"

pytestmark = pytest.mark.skipif(not SCRIPT.is_file(), reason="needs a source checkout")


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("release_notes", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CHANGELOG = """\
# Changelog

Preamble.

## [1.1.0] - 2030-02-01

A paragraph that is
wrapped over three
lines.

### Fixed

- **First.** An item that
  continues here.
- Second item.
  1. nested step
     that continues
- Third item, with code:

```python
x = 1
y = 2
```

| a | b |
|---|---|
| 1 | 2 |

## [1.0.0] - 2030-01-01

Older notes.

[1.1.0]: https://example.org/v1.1.0
[1.0.0]: https://example.org/v1.0.0
"""


def test_section_is_cut_at_the_next_version(tool):
    notes = tool.section(CHANGELOG, "1.1.0")
    assert notes.startswith("A paragraph that is")
    assert "Older notes" not in notes
    assert "## [" not in notes


def test_last_section_stops_before_the_link_definitions(tool):
    assert tool.section(CHANGELOG, "1.0.0") == "Older notes.\n"


def test_paragraphs_and_list_items_are_joined(tool):
    notes = tool.release_notes(CHANGELOG, "1.1.0").splitlines()
    assert notes[0] == "A paragraph that is wrapped over three lines."
    assert "- **First.** An item that continues here." in notes
    assert "- Second item." in notes
    assert "  1. nested step that continues" in notes


def test_headings_code_and_tables_are_left_alone(tool):
    notes = tool.release_notes(CHANGELOG, "1.1.0")
    assert "\n### Fixed\n\n- **First.**" in notes
    assert "```python\nx = 1\ny = 2\n```" in notes
    assert "| a | b |\n|---|---|\n| 1 | 2 |" in notes


def test_unknown_version(tool):
    with pytest.raises(ValueError, match=r"no section for version 9\.9\.9"):
        tool.section(CHANGELOG, "9.9.9")
    # a version is matched whole, not as the beginning of another
    with pytest.raises(ValueError, match=r"no section for version 1\.1"):
        tool.section(CHANGELOG, "1.1")


def test_a_heading_inside_a_code_block_does_not_end_the_section(tool):
    changelog = (
        "## [2.0.0] - 2030-03-01\n\nRename the section like this:\n\n"
        "```markdown\n## [x.y.z] - YYYY-MM-DD\n[x.y.z]: https://example.org\n```\n\n"
        "After the block.\n\n## [1.0.0] - 2030-01-01\n\nOlder notes.\n"
    )
    notes = tool.release_notes(changelog, "2.0.0")
    assert "## [x.y.z] - YYYY-MM-DD\n[x.y.z]: https://example.org\n```" in notes
    assert notes.endswith("After the block.\n")
    assert "Older notes" not in notes


@pytest.mark.parametrize(
    "changelog",
    [
        "## [1.0.0] - 2030-01-01\n",
        "## [1.0.0] - 2030-01-01\n\n\n## [0.9.0] - 2029-01-01\n\nNotes.\n",
        "## [1.0.0] - 2030-01-01\n\n[1.0.0]: https://example.org\n",
    ],
)
def test_an_empty_section_is_an_error(tool, changelog, tmp_path):
    # a release must not be published with nothing in its notes
    with pytest.raises(ValueError, match=r"section of the changelog for version 1\.0\.0 is empty"):
        tool.section(changelog, "1.0.0")
    path = tmp_path / "CHANGELOG.md"
    path.write_text(changelog, encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "1.0.0", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == ""
    assert "is empty" in completed.stderr


def test_declared_version_has_release_notes():
    """What the release workflow will run for the version in pyproject.toml."""
    declared = re.search(
        r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    assert declared is not None
    version = declared.group(1)
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), version, str(ROOT / "CHANGELOG.md")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    notes = completed.stdout
    assert notes.strip()
    assert "## [" not in notes
    # no hard-wrapped prose is left: every list item is on one line
    for line in notes.splitlines():
        assert not line.startswith("  ") or line.lstrip()[:1] in "-*0123456789`|", line


def test_command_line_errors():
    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "9.9.9", str(ROOT / "CHANGELOG.md")],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert missing.returncode == 1
    assert "no section for version 9.9.9" in missing.stderr
    usage = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert usage.returncode == 2


def test_output_is_utf8_whatever_the_code_page(tmp_path):
    """On Windows the default code page cannot encode Greek letters; the notes may hold them."""
    greek = "\u03bc and \u03c3."  # mu and sigma
    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(f"# Changelog\n\n## [1.0.0] - 2030-01-01\n\n{greek}\n", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "1.0.0", str(changelog)],
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "cp1252"},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.decode("utf-8").strip() == greek
