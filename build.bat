@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [build] Creating virtual environment...
    python -m venv .venv || goto :error
)

call ".venv\Scripts\activate.bat" || goto :error
python -m pip install --upgrade pip || goto :error
python -m pip install -r requirements.txt || goto :error

if not "%SKIP_OCR%"=="1" (
    echo [build] Installing OCR dependencies for team OCR build...
    python -m pip install -r requirements-ocr.txt || goto :error
) else (
    echo [build] SKIP_OCR=1, building standard non-OCR fallback package.
)

echo [build] Running tests...
python -m unittest discover -s tests || goto :error

echo [build] Cleaning old build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [build] Running PyInstaller...
pyinstaller LocalFullTextSearch.spec || goto :error

echo [build] Done. Output: dist\本地多格式全文搜索工具
pause
exit /b 0

:error
echo [build] Failed with error %ERRORLEVEL%.
pause
exit /b %ERRORLEVEL%
