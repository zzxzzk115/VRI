#!/bin/sh
# Run clang-format over the CI-checked trees (source/ examples/ tests/), matching the
# CI job exactly. Check by default; --fix formats in place.
#   scripts/check-format.sh          # check; exit 1 on any violation
#   scripts/check-format.sh --fix    # reformat the files in place
# CI pins clang-format 20.1.0:  pip install clang-format==20.1.0
set -eu
cd "$(dirname "$0")/.."

# Resolve clang-format: $CLANG_FORMAT, else the repo-provisioned venv, else PATH.
cf="${CLANG_FORMAT:-}"
if [ -z "$cf" ]; then
    for c in scripts/.cf-venv/bin/clang-format \
             scripts/.cf-venv/Lib/site-packages/clang_format/data/bin/clang-format.exe \
             scripts/.cf-venv/Scripts/clang-format.exe; do
        [ -x "$c" ] && { cf="$c"; break; }
    done
fi
if [ -z "$cf" ] && command -v clang-format >/dev/null 2>&1; then cf="clang-format"; fi
if [ -z "$cf" ]; then
    echo "clang-format not found. Run scripts/setup-hooks.sh (or pip install clang-format==20.1.0)." >&2
    exit 2
fi

files=$(git ls-files source examples tests | grep -E '\.(cpp|cc|h|hpp)$' || true)
[ -z "$files" ] && { echo "no C/C++ files to check"; exit 0; }

if [ "${1:-}" = "--fix" ]; then
    echo "$files" | xargs "$cf" -i
    echo "clang-format -i applied."
else
    echo "$files" | xargs "$cf" --dry-run --Werror
fi
