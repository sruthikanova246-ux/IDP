@echo off
REM One-time setup: virtual environment, dependencies, AI model download, sample images.
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
  echo Python was not found. Install Python 3.10 or newer from https://www.python.org/ and tick "Add to PATH".
  exit /b 1
)

if not exist .venv (
  echo Creating virtual environment...
  python -m venv .venv || exit /b 1
)

echo Installing dependencies (PyTorch CPU is about 200 MB)...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || exit /b 1

echo Downloading and verifying the AI model (about 370 MB, first time only)...
.venv\Scripts\python download_model.py
if errorlevel 1 echo WARNING: model download failed. The app will still run, but AI analysis will be reported as unavailable.

.venv\Scripts\python make_samples.py
echo.
echo Setup complete. Start the app with run_windows.bat
