<#
.SYNOPSIS
  Run clang-format over the CI-checked trees (source/ examples/ tests/), matching the
  CI job exactly. Check by default; -Fix formats in place.
.EXAMPLE
  scripts\check-format.ps1          # check; exit 1 on any violation
  scripts\check-format.ps1 -Fix     # reformat the files in place
.NOTES
  CI pins clang-format 20.1.0:  pip install clang-format==20.1.0
#>
param([switch]$Fix)
$ErrorActionPreference = "Stop"

$cf = (Get-Command clang-format -ErrorAction SilentlyContinue).Source
if (-not $cf) {
    Write-Error "clang-format not on PATH. Install the CI version: pip install clang-format==20.1.0"
    exit 2
}

Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    $files = git ls-files source examples tests | Where-Object { $_ -match '\.(cpp|cc|h|hpp)$' }
    if (-not $files) { Write-Host "no C/C++ files to check"; exit 0 }

    if ($Fix) {
        & $cf -i $files
        Write-Host "clang-format -i applied to $($files.Count) files."
        exit 0
    }

    & $cf --dry-run --Werror $files
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`nformat violations above. Fix with: scripts\check-format.ps1 -Fix" -ForegroundColor Yellow
    }
    exit $LASTEXITCODE
}
finally { Pop-Location }
