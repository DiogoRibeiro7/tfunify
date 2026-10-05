"""Print the release notes of one version from CHANGELOG.md.

Used by the release workflow::

    python tools/release_notes.py 0.2.0 > release-notes.md

The changelog is hard-wrapped at 80 columns, which reads well in an editor;
GitHub releases keep single line breaks, so paragraphs and list items are
joined back into single lines here.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

_BLOCK_START = re.compile(r"\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>|\||```|~~~|\[[^\]]+\]:\s)")


_HEADING = re.compile(r"## \[([^\]\n]+)\]")
_LINK_DEFINITION = re.compile(r"\[[^\]\n]+\]:\s")
_FENCES = ("```", "~~~")


def section(changelog: str, version: str) -> str:
    """The body of the ``## [version]`` section, without its heading.

    The section ends at the next ``## [...]`` heading or at the link
    definitions that close the file. Lines inside a code block end nothing.
    """
    body: list[str] | None = None
    in_code = False
    for line in changelog.splitlines():
        if body is None:
            heading = _HEADING.match(line)
            if heading is not None and heading.group(1) == version:
                body = []
            continue
        if line.lstrip().startswith(_FENCES):
            in_code = not in_code
        elif not in_code and (_HEADING.match(line) or _LINK_DEFINITION.match(line)):
            break
        body.append(line)
    if body is None:
        raise ValueError(f"the changelog has no section for version {version}")
    text = "\n".join(body).strip()
    if not text:
        raise ValueError(f"the section of the changelog for version {version} is empty")
    return text + "\n"


def unwrap(markdown: str) -> str:
    """Join the lines of each paragraph and list item; leave code blocks alone."""
    lines: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        fence = line.lstrip().startswith(_FENCES)
        continues = (
            not in_code
            and not fence
            and bool(lines)
            and bool(lines[-1].strip())
            and bool(line.strip())
            and not _BLOCK_START.match(line)
            and not lines[-1].lstrip().startswith(("#", "|", *_FENCES))
        )
        if continues:
            lines[-1] = f"{lines[-1].rstrip()} {line.strip()}"
        else:
            lines.append(line.rstrip())
        if fence:
            in_code = not in_code
    return "\n".join(lines) + "\n"


def release_notes(changelog: str, version: str) -> str:
    return unwrap(section(changelog, version))


def main(arguments: list[str]) -> int:
    if len(arguments) not in (1, 2):
        print("usage: release_notes.py VERSION [CHANGELOG]", file=sys.stderr)
        return 2
    path = Path(arguments[1]) if len(arguments) == 2 else Path("CHANGELOG.md")
    try:
        notes = release_notes(path.read_text(encoding="utf-8"), arguments[0])
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    sys.stdout.write(notes)
    return 0


if __name__ == "__main__":
    # UTF-8 whatever the code page of the platform: on Windows the default one
    # cannot encode every character a changelog or a traceback may contain.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main(sys.argv[1:]))
