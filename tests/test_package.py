"""The package as a user meets it: names, version, README and docstring examples."""

from __future__ import annotations

import doctest
import importlib
import re
from importlib.resources import files
from pathlib import Path

import pytest

import tfunify

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
MODULES = [
    "tfunify.american",
    "tfunify.cli",
    "tfunify.core",
    "tfunify.data",
    "tfunify.european",
    "tfunify.metrics",
    "tfunify.theory",
    "tfunify.tsmom",
]

needs_checkout = pytest.mark.skipif(not README.is_file(), reason="needs a source checkout")


def test_version_is_a_release_number():
    # final releases and pre-releases (0.3.0rc1); not the fallback of a source tree
    assert re.fullmatch(r"\d+\.\d+\.\d+((a|b|rc)\d+)?(\.post\d+)?", tfunify.__version__)


@needs_checkout
def test_the_version_is_declared_once_per_file_and_they_agree():
    declared = re.search(
        r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M
    )
    cited = re.search(
        r'^version: "?([^"\n]+)"?$', (ROOT / "CITATION.cff").read_text(encoding="utf-8"), re.M
    )
    assert declared is not None
    assert cited is not None
    assert cited.group(1) == declared.group(1)


def test_the_modules_are_reachable_from_the_package():
    # `import tfunify` is enough for tfunify.data.load_csv and the like
    for name in ("data", "metrics", "theory"):
        assert getattr(tfunify, name) is importlib.import_module(f"tfunify.{name}")
        assert name in tfunify.__all__


def test_all_names_exist_and_are_unique():
    for name in tfunify.__all__:
        assert hasattr(tfunify, name), name
    assert len(set(tfunify.__all__)) == len(tfunify.__all__)


@pytest.mark.parametrize("module_name", MODULES)
def test_modules_export_what_they_declare(module_name):
    module = importlib.import_module(module_name)
    for name in module.__all__:
        assert hasattr(module, name), f"{module_name}.{name}"


def test_everything_public_is_documented():
    undocumented = []
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in module.__all__:
            item = getattr(module, name)
            if callable(item) and not (item.__doc__ or "").strip():
                undocumented.append(f"{module_name}.{name}")
    assert undocumented == []


def test_the_systems_and_their_helpers_are_importable_from_the_top():
    for module_name in ("tfunify.american", "tfunify.core", "tfunify.european", "tfunify.tsmom"):
        module = importlib.import_module(module_name)
        for name in module.__all__:
            if name != "FloatArray":
                assert getattr(tfunify, name) is getattr(module, name), name


def test_package_is_typed():
    assert files("tfunify").joinpath("py.typed").is_file()


@pytest.mark.parametrize("module_name", MODULES)
def test_docstring_examples(module_name):
    module = importlib.import_module(module_name)
    results = doctest.testmod(module, optionflags=doctest.NORMALIZE_WHITESPACE)
    assert results.failed == 0


def test_the_classes_have_docstring_examples():
    attempted = sum(
        doctest.testmod(importlib.import_module(name)).attempted
        for name in ("tfunify.european", "tfunify.american", "tfunify.tsmom")
    )
    assert attempted >= 12


def readme_blocks():
    if not README.is_file():
        return []
    return re.findall(r"```python\n(.*?)```", README.read_text(encoding="utf-8"), flags=re.S)


BLOCKS = readme_blocks()


def run_block(index, capsys):
    exec(compile(BLOCKS[index], f"README.md block {index}", "exec"), {"__name__": "__readme__"})
    return capsys.readouterr().out.splitlines()


@needs_checkout
def test_readme_has_code_examples():
    assert len(BLOCKS) == 5


SUMMARY = (
    "annual return 130.00%, annual volatility 33.57%, Sharpe ratio 3.87, "
    "maximum drawdown 2.00% (4 days)"
)
# what each block prints; None stands for a line whose content is not quoted
PRINTED = [
    ["(5200,)", None],
    ["[-1, 0, 1]", "True"],
    ["290", "[100, 110, 120]"],
    ["0.34", "0.1", "0.22", "3.93", "0.88"],
    [SUMMARY, "3.87 0.02"],
]


@needs_checkout
@pytest.mark.parametrize("index", range(len(PRINTED)))
def test_readme_block_prints_what_its_comments_say(index, capsys):
    printed = run_block(index, capsys)
    expected = PRINTED[index]
    assert len(printed) == len(expected)
    for line, value in zip(printed, expected):
        if value is None:
            assert line.startswith("annual return ")
        else:
            assert line == value
            # ... and the README shows that value in a comment
            assert f"# {value}" in BLOCKS[index], value


@needs_checkout
def test_readme_names_exist():
    text = README.read_text(encoding="utf-8")
    for module, name in re.findall(r"`(tfunify(?:\.\w+)*)\.(\w+)`", text):
        assert hasattr(importlib.import_module(module), name), f"{module}.{name}"
    for name in ("sigma_floor_annual", "weight_cap", "warmup"):
        assert name in tfunify.EuropeanTFConfig.__dataclass_fields__
    assert f"`{'bias_correction=False'}`" in text


@needs_checkout
def test_readme_lists_every_example():
    text = README.read_text(encoding="utf-8")
    for script in (ROOT / "examples").glob("*.py"):
        assert f"`{script.name}`" in text, script.name
