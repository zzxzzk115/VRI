# Build the wasm examples and assemble the static website into _site\ (or -OutDir).
# Windows twin of scripts/build_site.sh (which CI uses) for local verification:
#   .\scripts\build_site.ps1
#   python -m http.server -d _site
param(
    [string]$OutDir = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
if ($OutDir -eq "") { $OutDir = Join-Path $Root "_site" }
$Art = Join-Path $Root "build\wasm\wasm32\release"

Push-Location $Root
try {
    $Sha = "dev"
    if (Get-Command git -ErrorAction SilentlyContinue) {
        try { $Sha = (git rev-parse --short HEAD).Trim() } catch { $Sha = "dev" }
    }

    xmake f -y -p wasm -m release --vri_build_examples=y --vri_build_tests=n --vri_build_tools=n
    if ($LASTEXITCODE -ne 0) { throw "xmake configure failed" }
    # Examples are set_default(false); --all builds them (tests/tools are disabled above).
    xmake build -y --all
    if ($LASTEXITCODE -ne 0) { throw "xmake build failed" }

    if (Test-Path $OutDir) { Remove-Item -Recurse -Force $OutDir }
    New-Item -ItemType Directory -Force (Join-Path $OutDir "examples") | Out-Null

    Copy-Item -Recurse -Force (Join-Path $Root "web\site\*") $OutDir
    foreach ($pattern in @("example-*.html", "example-*.js", "example-*.wasm", "example-*.data")) {
        Get-ChildItem (Join-Path $Art $pattern) -ErrorAction SilentlyContinue |
            Copy-Item -Destination (Join-Path $OutDir "examples")
    }

    # Stamp the git SHA into ?v={{VRI_SHA}} cache-busters.
    Get-ChildItem $OutDir -Recurse -Include "*.html", "*.js", "*.css" | ForEach-Object {
        $text = Get-Content $_.FullName -Raw
        if ($text -match [regex]::Escape("{{VRI_SHA}}")) {
            $text -replace [regex]::Escape("{{VRI_SHA}}"), $Sha |
                Set-Content $_.FullName -Encoding utf8 -NoNewline
        }
    }

    # Tell Pages not to run Jekyll (keeps files starting with _ etc. intact).
    New-Item -ItemType File -Path (Join-Path $OutDir ".nojekyll") -Force | Out-Null

    Write-Host "site assembled at $OutDir (build $Sha)"
}
finally {
    Pop-Location
}
