# Releasing Autotune

Releases publish `@roberttlange/autotune` to npm and `autotune-cli` to PyPI from `.github/workflows/release.yml`. The versions in `package.json`, `python/pyproject.toml`, changelog section, annotated Git tag, registries, and GitHub Release must agree.

## Prerequisites

- The GitHub repository is public and the `master` branch is green.
- The npm account owns the `@roberttlange` scope and has two-factor authentication enabled.
- A GitHub ruleset protects tags matching `v*` from deletion and non-release updates.
- A protected GitHub environment named `npm` exists and requires reviewer approval before deployment.
- A protected GitHub environment named `pypi` exists and is configured as the PyPI trusted publisher for `RobertTLange/autotune`'s `release.yml` workflow.

## First Publication

npm trusted publishing can only be configured after the package exists. Bootstrap `0.1.0` with a short-lived granular npm access token:

1. Create a granular token with read/write package access, bypass 2FA enabled, and the shortest practical expiration.
2. Save it as the `NPM_TOKEN` secret in the GitHub `npm` environment.
3. Complete the release preparation and push the annotated tag as described below.
4. Confirm npm and the GitHub Release both show `0.1.0` with provenance.
5. In the npm package settings, configure a GitHub Actions trusted publisher:
   - organization or user: `RobertTLange`
   - repository: `autotune`
   - workflow: `release.yml`
   - environment: `npm`
   - allowed action: `npm publish`
6. Set npm publishing access to require 2FA and disallow tokens.
7. Delete the `NPM_TOKEN` GitHub secret and revoke the bootstrap token.

The release workflow automatically uses npm OIDC after the trusted publisher is configured. GitHub-hosted OIDC releases generate npm provenance without a long-lived token. The OIDC-enabled job only verifies and publishes the exact tarball produced by the unprivileged build job; it does not install dependencies or run package lifecycle scripts.

## Prepare a Release

1. Update `package.json`, `package-lock.json`, and `python/pyproject.toml` to the exact target version.
2. Move release notes into a dated `CHANGELOG.md` section named `## [X.Y.Z] - YYYY-MM-DD`.
3. Run the complete gate:

   ```bash
   npm ci
   npm run lint
   npm test
   npm run build
   npm audit --omit=dev
   npm pack --dry-run
   python -m pip install ./python[test]
   python -m mypy python/src
   python -m ruff check python/src python/tests
   (cd python && python -m pytest --cov=autotune_cli)
   python -m build python
   ```

4. Commit with `chore: release vX.Y.Z` and push `master`.
5. Wait for CI to pass.
6. Create and push an annotated tag:

   ```bash
   git tag -a vX.Y.Z -m "Release vX.Y.Z"
   git push origin vX.Y.Z
   ```

The tag must point to a commit on `origin/master`. The release workflow rejects lightweight tags and version mismatches.

## Verify

Watch the release workflow, then verify the registry and installed CLI:

```bash
gh run list --workflow release.yml
npm view @roberttlange/autotune@X.Y.Z version dist.integrity
npx --yes @roberttlange/autotune@X.Y.Z --version
gh release view vX.Y.Z
```

Do not reuse a version or move an existing release tag.
