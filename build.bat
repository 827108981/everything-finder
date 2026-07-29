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
python -m pytest -q || goto :error

echo [build] Cleaning old build output...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [build] Running PyInstaller...
pyinstaller LocalFullTextSearch.spec || goto :error

set "PACKAGE_DIR=dist\本地多格式全文搜索工具"
set "PACKAGE_EXE=%PACKAGE_DIR%\本地多格式全文搜索工具.exe"

echo [build] Generating and verifying distribution manifest...
python tools\verify_distribution.py "%PACKAGE_DIR%" --stage-release-files --write-manifest --verify || goto :error

echo [build] Running frozen package smoke tests...
set "QT_QPA_PLATFORM=offscreen"
"%PACKAGE_EXE%" --self-test || goto :error
"%PACKAGE_EXE%" --validate-core || goto :error

echo [build] Done. Output: dist\本地多格式全文搜索工具
pause
exit /b 0

:error
echo [build] Failed with error %ERRORLEVEL%.
pause
exit /b %ERRORLEVEL%
