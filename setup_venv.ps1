$ErrorActionPreference = "Stop"

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvDir = Join-Path $projectDir ".venv"
$pythonCmd = "python"

Write-Host "Creating virtual environment in $venvDir ..."
& $pythonCmd -m venv $venvDir

$venvPython = Join-Path $venvDir "Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    throw "Could not find venv Python at $venvPython"
}

Write-Host "Upgrading pip ..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt ..."
& $venvPython -m pip install -r (Join-Path $projectDir "requirements.txt")

Write-Host ""
Write-Host "Done. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "Then run:"
Write-Host "  python .\clinic_schedule_part3_column_generation.py"
