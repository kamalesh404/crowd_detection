#!/bin/bash
# -----------------------------------------------
# CrowdSafe AI ? Quick Launch Script
# -----------------------------------------------

SCRIPT_PATH="${BASH_SOURCE[0]}"
SCRIPT_PATH="${SCRIPT_PATH//\\//}"
DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"
cd "$DIR"

# Activate venv if it exists (prefer .venv on Windows)
if [ -d ".venv" ]; then
  source .venv/Scripts/activate
elif [ -d "venv" ]; then
  source venv/bin/activate
fi

echo "Launching CrowdSafe AI..."
echo "Dashboard: http://localhost:8000"
if [[ "$@" == *"--sim"* ]]; then
  echo "Mode: SIMULATION (no camera)"
else
  echo "Mode: LIVE CAMERA (using default webcam)"
  echo "Tip: Use --sim for simulation or --source PATH for video file"
fi
echo "Press Q in the video window (or Ctrl+C) to stop."
echo ""

if [ -n "$VIRTUAL_ENV" ] && [ -x "$VIRTUAL_ENV/Scripts/python.exe" ]; then
  PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
elif command -v python >/dev/null 2>&1; then
  PYTHON="python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="python3"
else
  echo "Python not found in PATH."
  exit 1
fi

"$PYTHON" main.py "$@"
