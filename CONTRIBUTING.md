# Contributing to VRI

Thanks for helping out! A few conventions keep the tree consistent across platforms.

## One-time setup

Enable the git hooks and provision the pinned formatter (clang-format 20.1.0) into a
repo-local virtualenv:

```sh
scripts/setup-hooks.sh        # Linux / macOS
```
```powershell
scripts\setup-hooks.ps1       # Windows
```

This points `core.hooksPath` at `scripts/hooks/` and installs clang-format 20.1.0 under
`scripts/.cf-venv/` (git-ignored). If you already have clang-format 20.x on `PATH`, it
uses that and skips the download. No usable Python? The script prints how to get
clang-format another way — until it's available the hook just **skips** (it never blocks
a commit), and CI stays the backstop.

## Code style

C++ is formatted with **clang-format 20.1.0** (the root `.clang-format`). CI pins that
exact version, so differences between clang-format majors matter — use the provisioned
one. The pre-commit hook checks the **staged** C/C++ files under `source/ examples/ tests/`
with the same command CI runs (`clang-format --dry-run --Werror`) and blocks a commit
that isn't formatted, printing the fix.

Check or fix the whole tree yourself:

```sh
scripts/check-format.sh          # check (exit 1 on violations)
scripts/check-format.sh --fix    # reformat in place
```
```powershell
scripts\check-format.ps1         # check
scripts\check-format.ps1 -Fix    # reformat in place
```

Bypass the hook for a single commit (e.g. a WIP checkpoint):

```sh
git commit --no-verify
```

Generated shader headers (`tests/shaders/`) and the Objective-C++ Metal backend are
exempt via their own `DisableFormat` `.clang-format` files; CI only checks
`source/ examples/ tests/`.

## Commits

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org)
(`feat:`, `fix:`, `docs:`, `chore:`, `style:`, …), matching the existing history.

## Building & running

See the [README](README.md#building) for build/run instructions and the backend matrix.
New to a backend's internals? [DETAIL.md](DETAIL.md) documents how each feature maps onto
each backend.
