<#
.SYNOPSIS
  One-time contributor setup: enable the VRI git hooks and provision the pinned
  clang-format (20.1.0) into a repo-local venv, so the pre-commit format check works
  without a global install. Safe to re-run.
.EXAMPLE
  scripts\setup-hooks.ps1
#>
$ErrorActionPreference = "Stop"
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    git config core.hooksPath scripts/hooks
    Write-Host "hooks enabled (core.hooksPath = scripts/hooks)."

    # Already have clang-format 20.x on PATH? Then nothing to provision.
    $cf = (Get-Command clang-format -ErrorAction SilentlyContinue).Source
    if ($cf -and ((& $cf --version) -match 'version 20\.')) {
        Write-Host "clang-format 20.x already on PATH; done."
        exit 0
    }

    # Find a usable Python (skip the Microsoft Store execution-alias stub).
    $py = $null
    foreach ($p in "python", "python3", "py") {
        $src = (Get-Command $p -ErrorAction SilentlyContinue).Source
        if ($src -and $src -notmatch 'WindowsApps') { $py = $src; break }
    }
    if (-not $py) {
        Write-Warning "No usable Python found, so clang-format 20.1.0 was not provisioned."
        Write-Host "  Get it any of these ways, then re-run (or just commit - the hook skips safely):"
        Write-Host "    - pip install clang-format==20.1.0   (needs Python)"
        Write-Host "    - install clang-format from your package manager / LLVM release"
        Write-Host "    - `$env:CLANG_FORMAT = 'C:\path\to\clang-format.exe'"
        exit 0
    }

    Write-Host "provisioning clang-format 20.1.0 into scripts/.cf-venv (via $py) ..."
    & $py -m venv scripts/.cf-venv
    $vpy = if (Test-Path scripts/.cf-venv/Scripts/python.exe) { "scripts/.cf-venv/Scripts/python.exe" } else { "scripts/.cf-venv/bin/python" }
    & $vpy -m pip install --quiet --disable-pip-version-check clang-format==20.1.0
    Write-Host "done - the hook and scripts\check-format.ps1 will use scripts/.cf-venv's clang-format."
}
finally { Pop-Location }
