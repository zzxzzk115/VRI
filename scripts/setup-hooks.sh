#!/bin/sh
# One-time contributor setup: enable the VRI git hooks and provision the pinned
# clang-format (20.1.0) into a repo-local venv, so the pre-commit format check works
# without a global install. Safe to re-run.
#   scripts/setup-hooks.sh
set -eu
cd "$(dirname "$0")/.."

git config core.hooksPath scripts/hooks
echo "hooks enabled (core.hooksPath = scripts/hooks)."

# Already have clang-format 20.x on PATH? Then nothing to provision.
if command -v clang-format >/dev/null 2>&1; then
    case "$(clang-format --version)" in
        *"version 20."*) echo "clang-format 20.x already on PATH; done."; exit 0 ;;
    esac
fi

# Find a usable Python for the pip wheel (a self-contained clang-format binary).
# Validate that the candidate actually runs - `command -v` also finds the Microsoft
# Store execution-alias stub, which is not a real interpreter.
py=""
for p in python3 python py; do
    command -v "$p" >/dev/null 2>&1 || continue
    "$p" -c "import sys" >/dev/null 2>&1 || continue
    py="$p"; break
done
if [ -z "$py" ]; then
    echo "note: no Python found, so clang-format 20.1.0 was not provisioned." >&2
    echo "      Get it any of these ways, then re-run (or just commit - the hook skips safely):" >&2
    echo "        - pip install clang-format==20.1.0   (needs Python)" >&2
    echo "        - install clang-format from your package manager / LLVM release" >&2
    echo "        - export CLANG_FORMAT=/path/to/clang-format" >&2
    exit 0
fi

echo "provisioning clang-format 20.1.0 into scripts/.cf-venv (via $py) ..."
"$py" -m venv scripts/.cf-venv
if [ -x scripts/.cf-venv/bin/python ]; then vpy=scripts/.cf-venv/bin/python; else vpy=scripts/.cf-venv/Scripts/python.exe; fi
"$vpy" -m pip install --quiet --disable-pip-version-check clang-format==20.1.0
echo "done - the hook and scripts/check-format.sh will use scripts/.cf-venv's clang-format."
