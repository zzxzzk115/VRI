#!/bin/sh
# Run clang-format over the CI-checked trees (source/ examples/ tests/), matching the
# CI job exactly. Check by default; --fix formats in place.
#   scripts/check-format.sh          # check; exit 1 on any violation
#   scripts/check-format.sh --fix    # reformat the files in place
# CI pins clang-format 20.1.0:  pip install clang-format==20.1.0
set -eu
cd "$(dirname "$0")/.."

if ! command -v clang-format >/dev/null 2>&1; then
    echo "clang-format not on PATH. Install the CI version: pip install clang-format==20.1.0" >&2
    exit 2
fi

files=$(git ls-files source examples tests | grep -E '\.(cpp|cc|h|hpp)$' || true)
[ -z "$files" ] && { echo "no C/C++ files to check"; exit 0; }

if [ "${1:-}" = "--fix" ]; then
    echo "$files" | xargs clang-format -i
    echo "clang-format -i applied."
else
    echo "$files" | xargs clang-format --dry-run --Werror
fi
