# Development

## Setting up

```bash
git clone https://github.com/DiogoRibeiro7/tfunify.git
cd tfunify
python -m pip install -e ".[dev]"
```

## Checks

```bash
python -m pytest --cov        # tests, with coverage
ruff check .                  # lint
ruff format --check .         # formatting
mypy                          # types, strict
```

CI runs the same on every pull request: the tests on Python 3.10 to 3.14 on Linux, on macOS and Windows, and with the oldest supported NumPy; lint and types on the oldest and the newest Python; and a build of the distribution whose wheel is installed into a clean environment to run the command line.

To run the hooks before each commit: `pip install pre-commit && pre-commit install`.

For anything a user would notice, add a line to the `Unreleased` section at the top of `CHANGELOG.md`, creating the section if it is not there.

## How the tests are built

The numerics are tested against something that was computed another way:

- **Definitions as explicit sums.** `tests/reference.py` evaluates the formulas of the paper literally, one day and one lag at a time, and shares no code with the package. The systems must match it.
- **A path worked by hand** for the American system, and its rules checked on every day of random paths: every entry, exit, stop and size must be the one the definition prescribes, and nothing else may happen.
- **Special cases published in the paper**: white noise with drift, AR(1), the turnover figures, and the break-even cost (the cost per unit of turnover that uses up the expected return).
- **Moments by direct summation** over the filter weights, for five autocorrelation functions.
- **Simulation**, where a closed form predicts a statistic. The bounds of these tests are about five standard errors wide, so that they hold for any seed and not only for the one in the file.
- **Invariances**: results do not change when later data are appended; scaling the returns changes nothing; mirroring them mirrors the position; unchanged prices in front of a series change nothing.

`tests/test_regressions.py` has the figures that version 0.1.3 produced, one test for each of its numerical defects.

## Documentation

```bash
python -m pip install -e ".[docs]"
mkdocs serve                  # preview at http://127.0.0.1:8000
mkdocs build --strict         # what CI runs
```

The reference pages are generated from the docstrings, which are written in Markdown with NumPy-style sections. The site is built in strict mode whenever a pull request or a push changes the pages, the docstrings, the changelog or the configuration, and is published to GitHub Pages from the default branch.

Publishing needs GitHub Pages to be enabled once: *Settings > Pages*, with "GitHub Actions" as the source. Until then the workflow builds and checks the site and skips the deployment with a notice.

## Branches and releases

Work happens on `develop`, through pull requests. `main` holds the released code.

A release needs no manual step beyond a pull request:

1. On `develop`, set the new `version` in `pyproject.toml` and in `CITATION.cff`, and rename the `Unreleased` section of `CHANGELOG.md` to `## [x.y.z] - YYYY-MM-DD`. The tests check that the two versions agree and that the version has a changelog section.
2. Open a pull request from `develop` to `main` and merge it.
3. The release workflow sees a version without a tag, runs the whole CI on that commit, creates the tag `vx.y.z` and a GitHub release whose notes are that section of the changelog, and attaches the wheel and the source distribution.
4. If the repository variable `PUBLISH_TO_PYPI` is `true`, it also publishes to PyPI.

A push to `main` that does not change the version releases nothing. If a release run fails, fix the cause on `develop` and merge again: the version still has no tag, so the next run retries.

Both branches require signed commits. A fix that must reach `main` without what is on `develop` is branched from `main`, merged there through its own pull request, and `main` is then merged back into `develop`.

### Publishing to PyPI

Publishing uses PyPI's trusted publishing, so no token is stored. To turn it on:

1. On PyPI, add a trusted publisher for the project `tfunify`: owner `DiogoRibeiro7`, repository `tfunify`, workflow `release.yml`, environment `pypi`. For a project that does not exist yet this is done under "Publishing" in the account settings, as a pending publisher.
2. In the repository settings, create the variable `PUBLISH_TO_PYPI` with the value `true`.

Releases made after that are uploaded. A version that was released before is uploaded by starting the *Release* workflow by hand on `main` with its number in the `publish_version` field: the files attached to its GitHub release go to PyPI as they are.

[DEVELOPMENT_WORKFLOW.md](https://github.com/DiogoRibeiro7/tfunify/blob/develop/DEVELOPMENT_WORKFLOW.md) in the repository has the same in more detail.

## Changelog

See the [changelog](changelog.md).
