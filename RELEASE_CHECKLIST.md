# Release Checklist

This document captures the release workflow used for this repository.

## 1. Confirm the version

Before tagging, make sure the version is updated consistently:

- `pyproject.toml`
- `src/cryosieve/__init__.py`

## 2. Confirm the target commit

Check that the release should be cut from the intended commit and that the working tree is in the expected state.

Useful commands:

```bash
git status --short
git log --oneline --decorate -5
git rev-parse HEAD
```

## 3. Check whether the tag or release already exists

```bash
git tag -l "vX.Y.Z"
git ls-remote --tags origin "vX.Y.Z"
gh release view "vX.Y.Z" --repo mxhulab/cryosieve
```

## 4. Make sure GitHub CLI is ready

If needed:

```bash
brew install gh
gh auth login
gh auth status
```

## 5. Create and push the tag

Tag the intended commit, then push it:

```bash
git tag "vX.Y.Z" <commit>
git push origin "vX.Y.Z"
```

If the release should point at the current `HEAD`, use that commit explicitly or tag `HEAD` after verifying it is correct.

## 6. Create the GitHub Release

This repository publishes to PyPI from `.github/workflows/pippublish.yml`, and that workflow is triggered by the GitHub `release.published` event.

That means:

- pushing the tag is not enough
- the GitHub Release must also be created/published

Example:

```bash
gh release create "vX.Y.Z" \
  --repo mxhulab/cryosieve \
  --title "vX.Y.Z" \
  --generate-notes
```

## 7. Rewrite the release notes

Do not leave the release notes as a raw auto-generated engineering changelog.

Preferred tone for this repository:

- user-facing
- cryo-EM / paper-adjacent
- focused on practical impact

Emphasize:

- what users will notice
- installation or environment updates
- runtime / workflow improvements
- downstream analysis changes
- compatibility notes when results may differ slightly from older versions

## 8. Verify the published release

After publishing:

- open the GitHub release page
- confirm the title, tag, and notes are correct
- confirm the PyPI publish workflow has started or completed

Useful commands:

```bash
gh release view "vX.Y.Z" --repo mxhulab/cryosieve
gh run list --repo mxhulab/cryosieve --workflow pippublish.yml
```
