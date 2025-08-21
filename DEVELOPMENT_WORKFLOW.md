# Development Workflow

This document outlines the recommended workflow for developing and releasing `tfunify`.

## Branch Structure

- **`main`**: Production-ready code. All releases are made from this branch.
- **`develop`**: Integration branch for features. All development happens here.
- **`feature/*`**: Feature branches for individual features or fixes.

## Development Process

### 1\. Feature Development

```bash
# Start from develop
git checkout develop
git pull origin develop

# Create feature branch
git checkout -b feature/your-feature-name

# Make your changes
# ... code, test, commit ...

# Push feature branch
git push origin feature/your-feature-name

# Create Pull Request to develop branch
```

### 2\. Testing and Integration

All PRs to `develop` will trigger:

- Unit tests across Python 3.10, 3.11, 3.12
- Linting and type checking
- Integration tests

### 3\. Preparing for Release

```bash
# When ready to release, merge develop to main
git checkout main
git pull origin main
git merge develop

# Update version in pyproject.toml
poetry version patch  # or minor/major

# Update CHANGELOG.md with release notes

# Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "Bump version to X.Y.Z"

# Push to main
git push origin main
```

### 4\. Creating a Release

**Important**: Only create release tags from the `main` branch!

```bash
# Ensure you're on main and up to date
git checkout main
git pull origin main

# Create and push tag (this triggers the release workflow)
git tag vX.Y.Z
git push origin vX.Y.Z
```

The release workflow will:

1. ✅ Verify the tag is on the `main` branch
2. ✅ Run full test suite
3. ✅ Test CLI functionality
4. ✅ Verify version consistency
5. ✅ Build and test the package
6. ✅ Publish to PyPI
7. ✅ Create GitHub release

### 5\. After Release

```bash
# Merge main back to develop to keep branches in sync
git checkout develop
git merge main
git push origin develop
```

## Workflow Triggers

### CI Tests (`test.yml`)

- **Triggers**: Push/PR to `main` or `develop`
- **Purpose**: Continuous integration testing
- **Scope**: Multi-platform, comprehensive tests

### Main CI (`ci.yml`)

- **Triggers**: Push/PR to `main` or `develop`
- **Purpose**: Linting, type checking, basic tests
- **Scope**: Ubuntu only, fast feedback

### Release (`release.yml`)

- **Triggers**: Tag push (v_._._) _*AND__ tag must be on `main` branch
- **Purpose**: Production release to PyPI
- **Scope**: Full validation + publication

## Release Checklist

Before creating a release tag:

- [ ] All features merged to `develop`
- [ ] `develop` merged to `main`
- [ ] Version updated in `pyproject.toml`
- [ ] `CHANGELOG.md` updated
- [ ] All tests passing on `main`
- [ ] Currently on `main` branch
- [ ] Local `main` is up to date with remote

Then:

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

## Hotfixes

For urgent fixes to production:

```bash
# Create hotfix branch from main
git checkout main
git checkout -b hotfix/urgent-fix

# Make minimal fix
# ... fix, test, commit ...

# Merge to main
git checkout main
git merge hotfix/urgent-fix

# Update version (patch)
poetry version patch
git add pyproject.toml
git commit -m "Hotfix: bump version to X.Y.Z"

# Create release
git tag vX.Y.Z
git push origin vX.Y.Z

# Merge back to develop
git checkout develop
git merge main
git push origin develop
```

## Protection Rules (Recommended)

Consider setting up these branch protection rules in GitHub:

### `main` branch:

- Require pull request reviews
- Require status checks to pass (CI tests)
- Require up-to-date branches
- Include administrators in restrictions

### `develop` branch:

- Require status checks to pass (CI tests)
- Require up-to-date branches
