# Release guide

mini-articraft publishes preview releases on GitHub. Each release attaches the wheel, the sdist,
and a `SHA256SUMS` file. The [release preview workflow](../.github/workflows/release-preview.yml)
builds and verifies each artifact before it publishes anything.

## Tag scheme

A release tag is `v` plus the package version, for example `v0.1.1a1`. The version must end in a
PEP 440 prerelease segment such as `a1`, `b1`, `rc1`, or `dev1`. The workflow rejects a tag that
does not match `__version__`, a version without a prerelease segment, and a tag that already
exists.

The legacy `sdk-preview-2026-07-21.1` tag predates this scheme and stays as it is.

## Create a preview release

1. Bump `__version__` in [`src/mini_articraft/__init__.py`](../src/mini_articraft/__init__.py) to
   the next prerelease version, for example `0.1.1a1`. The package version is dynamic, so the
   lockfile does not change. Merge the bump to `main`.
2. Run the manual workflow against that commit:

   ```shell
   gh workflow run release-preview.yml --ref main -f tag=v0.1.1a1
   gh run watch
   ```

## What the workflow enforces

The build job validates the tag, builds with `uv build --no-sources`, and runs
`twine check` on both artifacts. It installs the wheel into a clean virtual environment and runs
an isolated smoke test: the import must come from the environment, `mini_articraft.sdk` must not
load OpenUSD, and the [hinged box example](../examples/hinged_box/main.py) must pass its tests
and export a USDZ file. It smoke-tests the sdist the same way and writes `SHA256SUMS`.

A separate release job re-verifies the checksums and creates the GitHub prerelease. Only that
final job has `contents: write` permission.
