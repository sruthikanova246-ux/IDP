@echo off
REM Start the Digital Evidence Integrity System
cd /d "%~dp0"

if not exist .venv\Scripts\python.exe (
  echo The virtual environment is missing. Run setup_windows.bat first.
  exit /b 1
)

start "" http://127.0.0.1:5000/
.venv\Scripts\python app.py
