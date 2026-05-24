# Release procedure

Replace `x.y.z` with the new version throughout. Steps 1-3 only edit
files; make them on a release branch.

## Steps

1. Update the version number in `pyproject.toml` and `Cargo.toml`
   (the two must match).

2. Update `CHANGELOG.md`:
   - Promote `[Unreleased]` to a new `[x.y.z] - YYYY-MM-DD` section.
   - Update the comparison links at the bottom.

3. Regenerate `Cargo.lock`:
   ```bash
   cargo generate-lockfile
   ```
   This bumps the `phonors` entry to `x.y.z` and may also refresh
   transitive dependencies to their latest compatible versions, so
   sanity-check that it still compiles before committing:
   ```bash
   cargo check --release
   ```
   (A plain `cargo build` fails to link locally because the pyo3
   extension needs the Python interpreter symbols; the wheels are
   built with maturin in CI, not with bare `cargo build`.)

4. Commit on a release branch and open a PR:
   ```bash
   git switch -c release/vx.y.z
   git add pyproject.toml Cargo.toml Cargo.lock CHANGELOG.md
   git commit -m "Release vx.y.z"
   git push origin release/vx.y.z
   ```
   Merge the PR once CI is green.

5. Tag the merged commit on `main` and push the tag:
   ```bash
   git switch main && git pull
   git tag vx.y.z
   git push origin vx.y.z
   ```
   Pushing a `v*` tag triggers the release workflow
   (`.github/workflows/release.yml`), which builds wheels and the
   sdist for all supported platforms.

6. **Approve the PyPI upload.** The `publish-pypi` job runs in the
   `pypi` GitHub environment, which has a required reviewer, so the
   run pauses at that job. Open the run in the Actions tab, click
   "Review deployments", and approve `pypi`. The wheels are built but
   never uploaded until this approval is given.

7. Verify the release: check the new version on
   https://pypi.org/project/phonors/ and `pip install phonors` in a
   fresh virtualenv.

8. conda-forge updates itself. A bot opens a PR against
   `conda-forge/phonors-feedstock` referencing the new PyPI sdist
   within a few hours; review and merge it (fix the recipe first if
   the dependencies changed).

## Notes

- A TestPyPI rehearsal is available before tagging: trigger the
  workflow manually (`gh workflow run release.yml --ref main`, or
  Actions -> Release -> "Run workflow"). `workflow_dispatch`
  publishes to TestPyPI; only a `v*` tag publishes to PyPI.
