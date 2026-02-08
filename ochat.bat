@echo off
setlocal

set "SCRIPT_DIR=%~dp0"

:: Create venv and install dependencies if needed
if not exist "%SCRIPT_DIR%.venv" (
    echo Creating virtual environment...
    python -m venv "%SCRIPT_DIR%.venv"
    echo Installing dependencies...
    "%SCRIPT_DIR%.venv\Scripts\pip" install -q -r "%SCRIPT_DIR%requirements.txt"
    echo Done.
)

"%SCRIPT_DIR%.venv\Scripts\python" "%SCRIPT_DIR%ochat.py" %*
