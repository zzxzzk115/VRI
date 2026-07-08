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

Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    # Resolve clang-format: $env:CLANG_FORMAT, else the repo-provisioned venv, else PATH.
    $cf = $env:CLANG_FORMAT
    if (-not $cf) {
        $cf = @("scripts\.cf-venv\Lib\site-packages\clang_format\data\bin\clang-format.exe",
                "scripts\.cf-venv\Scripts\clang-format.exe") |
              Where-Object { Test-Path $_ } | Select-Object -First 1
    }
    if (-not $cf) { $cf = (Get-Command clang-format -ErrorAction SilentlyContinue).Source }
    if (-not $cf) {
        Write-Error "clang-format not found. Run scripts\setup-hooks.ps1 (or pip install clang-format==20.1.0)."
        exit 2
    }
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
