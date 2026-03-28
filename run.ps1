$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Activate venv if it exists (prefer .venv)
if (Test-Path ".venv\Scripts\Activate.ps1") {
  . ".venv\Scripts\Activate.ps1"
} elseif (Test-Path "venv\Scripts\Activate.ps1") {
  . "venv\Scripts\Activate.ps1"
}

Write-Host "Launching CrowdSafe AI..."
Write-Host "Dashboard: http://localhost:8000"

$argsText = ($args -join " ")
if ($argsText -match "--sim") {
  Write-Host "Mode: SIMULATION (no camera)"
} else {
  Write-Host "Mode: LIVE CAMERA (using default webcam)"
  Write-Host "Tip: Use --sim for simulation or --source PATH for video file"
}
Write-Host "Press Q in the video window (or Ctrl+C) to stop.`n"

if (Test-Path ".venv\Scripts\python.exe") {
  & ".venv\Scripts\python.exe" main.py @args
} elseif (Test-Path "venv\Scripts\python.exe") {
  & "venv\Scripts\python.exe" main.py @args
} else {
  python main.py @args
}
