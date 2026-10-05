# Development workflow

How changes get into tfunify and how a release is made. The checks and the
release are run by GitHub Actions; nothing is tagged or published by hand.

## Branches

- **`develop`** is the default branch. All work lands here through pull
  requests.
- **`main`** holds the released code. It receives merges from `develop`, and
  from a hotfix branch when a fix cannot wait for what is on `develop`.
- **Feature branches** are cut from `develop` and merged back into it.

Both `develop` and `main` are protected and require signed commits. Commits
that GitHub cannot verify block a pull request whatever the merge method, so
sign them: `git commit -S`, or, for commits already made on a branch,

```bash
git rebase --exec 'git commit --amend --no-edit -S' develop
git push --force-with-lease
```

## Making a change

```bash
git checkout develop
git pull
git checkout -b feature/short-description

python -m pip install -e ".[dev]"
# ... edit, add tests ...
python -m pytest --cov
ruff check . && ruff format --check .
mypy

git push -u origin feature/short-description
# open a pull request against develop
```

Add a line to the `Unreleased` section of `CHANGELOG.md` (create the section if
it is not there) for anything a user would notice.

## What CI runs

On every pull request and on every push to `develop` and `main`
(`.github/workflows/ci.yml`):

| Job | What it does |
|---|---|
| `lint` | `ruff check`, `ruff format --check` and `mypy --strict` on Python 3.10 and 3.14 |
| `test (3.10)` ... `test (3.14)` | the test suite with coverage on Linux |
| `other-systems` | the test suite on macOS and Windows |
| `oldest-dependencies` | the test suite with NumPy 1.24 on Python 3.10 |
| `integration-test` | builds the distribution, installs the wheel into a clean environment and runs the `tfu` command on a generated price file |

The branch protection of `main` requires `test (3.10)`, `test (3.11)` and
`test (3.12)`. Those names come from the id of the job and its matrix, so the
`test` job in `ci.yml` must keep its id and must not be given a `name`.

When a test fails, its assertion is shown as an annotation on the summary page
of the run and on the pull request, so the log does not have to be opened.

The documentation site is built in strict mode by `.github/workflows/docs.yml`
whenever the docs, the docstrings or the changelog change, and is published to
GitHub Pages from the default branch. Publishing needs Pages to be enabled
once, under *Settings > Pages*, with "GitHub Actions" as the source; until then
the workflow builds and checks the site and skips the deployment with a notice.

## Releasing

1. On `develop`, set the new `version` in `pyproject.toml` and in
   `CITATION.cff`, and rename the `Unreleased` section of `CHANGELOG.md` to
   `## [x.y.z] - YYYY-MM-DD`. The tests check that the two versions agree and
   that the declared version has a changelog section that is not empty.
2. Open a pull request from `develop` to `main` and merge it.

That is all. `.github/workflows/release.yml` runs on every push to `main` and
compares the version in `pyproject.toml` with the existing tags. When the
version has no tag yet, it

1. runs the complete CI workflow on that commit,
2. builds the wheel and the source distribution and checks them,
3. creates the tag `vx.y.z` and a GitHub release whose notes are the section of
   the changelog for that version, with the two files attached,
4. publishes them to PyPI, if that has been switched on (see below).

A push to `main` that does not change the version releases nothing. If a
release run fails, fix the cause on `develop` and merge again: the version
still has no tag, so the next run retries.

After a release, merge `main` back into `develop` if the merge created a commit
that `develop` does not have.

### Version numbers

- **Major** (`x.0.0`): incompatible changes.
- **Minor** (`0.y.0`): new functionality; before 1.0, also incompatible changes.
- **Patch** (`0.0.z`): fixes.

### Publishing to PyPI

The `pypi` job of the release workflow is skipped unless the repository
variable `PUBLISH_TO_PYPI` is `true`. It uses PyPI's trusted publishing, so no
token is stored in the repository. To switch it on:

1. On PyPI, register a trusted publisher for the project `tfunify`: owner
   `DiogoRibeiro7`, repository `tfunify`, workflow `release.yml`, environment
   `pypi`. While the project does not exist on PyPI, this is done as a
   "pending publisher" under *Publishing* in the account settings.
2. In the repository, under *Settings > Secrets and variables > Actions >
   Variables*, create `PUBLISH_TO_PYPI` with the value `true`.

The next release is then uploaded. A version that is already tagged is not
released again. To publish it after the fact, start the *Release* workflow by
hand (*Actions > Release > Run workflow*) on `main` with its number, for
example `0.2.0`, in the `publish_version` field: the job downloads the files
attached to the GitHub release of that version and uploads them to PyPI as
they are.

## Hotfixes

Branch from `develop`, fix, raise the patch version, and follow the same two
pull requests (`develop`, then `main`). If `main` must be fixed without what is
on `develop`, branch from `main`, open the pull request against `main`, and
merge `main` back into `develop` afterwards.
