@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [run_dev] Creating virtual environment...
    python -m venv .venv || goto :error
)

call ".venv\Scripts\activate.bat" || goto :error
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error

echo [run_dev] Starting application...
python app.py
exit /b %ERRORLEVEL%

:error
echo [run_dev] Failed with error %ERRORLEVEL%.
pause
exit /b %ERRORLEVEL%
