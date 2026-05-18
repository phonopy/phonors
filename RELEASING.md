# Release procedure

## Steps

1. Update the version number in `pyproject.toml` and `Cargo.toml`.

2. Update `CHANGELOG.md`:
   - Add a new `[x.y.z] - YYYY-MM-DD` section under `[Unreleased]`.
   - Update the comparison links at the bottom.

3. Regenerate `Cargo.lock`:
   ```bash
   cargo generate-lockfile
   ```

4. Commit the changes:
   ```bash
   git add pyproject.toml Cargo.toml Cargo.lock CHANGELOG.md
   git commit -m "Release vx.y.z"
   ```

5. Create and push the tag:
   ```bash
   git tag vx.y.z
   git push origin release
   git push origin vx.y.z
   ```

Pushing the tag triggers the GitHub Actions release workflow, which builds wheels for all supported platforms and publishes them to PyPI.
