#!/usr/bin/env bash
# Build the wasm examples and assemble the static website into _site/ (or $1).
# Used by .github/workflows/deploy_pages.yml; scripts/build_site.ps1 is the
# Windows twin for local verification.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/_site}"
ART="$ROOT/build/wasm/wasm32/release"

cd "$ROOT"

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo dev)"

xmake f -y -p wasm -m release --vri_build_examples=y --vri_build_tests=n --vri_build_tools=n
# Examples are set_default(false); --all builds them (tests/tools are disabled above).
xmake build -y --all

rm -rf "$OUT"
mkdir -p "$OUT/examples"

cp -r "$ROOT"/web/site/. "$OUT/"
cp "$ART"/example-*.html "$ART"/example-*.js "$ART"/example-*.wasm "$OUT/examples/"
cp "$ART"/example-*.data "$OUT/examples/" 2>/dev/null || true

# Stamp the git SHA into ?v={{VRI_SHA}} cache-busters.
grep -rl '{{VRI_SHA}}' "$OUT" | while read -r f; do
    sed -i "s/{{VRI_SHA}}/$SHA/g" "$f"
done

# Tell Pages not to run Jekyll (keeps files starting with _ etc. intact).
touch "$OUT/.nojekyll"

echo "site assembled at $OUT (build $SHA)"
