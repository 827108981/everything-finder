@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [build] Creating virtual environment...
    python -m venv .venv || goto :error
)

call ".venv\Scripts\activate.bat" || goto :error
for /f %%V in ('python -c "from local_full_text_search.version import __version__; print(__version__)"') do set "APP_VERSION=%%V"
if not defined APP_VERSION goto :error
if not "%SKIP_INSTALL%"=="1" (
    python -m pip install --upgrade pip || goto :error
    python -m pip install -r requirements.txt || goto :error
)

if not "%SKIP_OCR%"=="1" (
    if not "%SKIP_INSTALL%"=="1" (
        echo [build] Installing OCR dependencies for team OCR build...
        python -m pip install -r requirements-ocr.txt || goto :error
    )
    echo [build] Verifying OCR model manifest...
    python tools\generate_ocr_model_manifest.py ocr_models --verify || goto :error
) else (
    echo [build] SKIP_OCR=1, building standard non-OCR fallback package.
)

echo [build] Running tests...
python -m pytest -q || goto :error

for /f "delims=" %%N in ('python -c "print('\u672c\u5730\u591a\u683c\u5f0f\u5168\u6587\u641c\u7d22\u5de5\u5177')"') do set "PACKAGE_NAME=%%N"
if not defined PACKAGE_NAME goto :error
set "BUILD_DIR=build\LocalFullTextSearch-%APP_VERSION%"
set "STAGE_ROOT=dist\.staging-%APP_VERSION%"
set "PACKAGE_DIR=%STAGE_ROOT%\%PACKAGE_NAME%"
set "FINAL_PACKAGE_DIR=dist\%PACKAGE_NAME%-%APP_VERSION%"
set "PACKAGE_EXE=%PACKAGE_DIR%\%PACKAGE_NAME%.exe"
set "PACKAGE_ZIP=dist\%PACKAGE_NAME%-%APP_VERSION%.zip"

echo [build] Cleaning version-specific build output...
if exist "%BUILD_DIR%" rmdir /s /q "%BUILD_DIR%"
if exist "%STAGE_ROOT%" rmdir /s /q "%STAGE_ROOT%"
if exist "%FINAL_PACKAGE_DIR%" rmdir /s /q "%FINAL_PACKAGE_DIR%"
if exist "%PACKAGE_ZIP%" del /q "%PACKAGE_ZIP%"
if exist "%BUILD_DIR%" goto :cleanup_error
if exist "%STAGE_ROOT%" goto :cleanup_error
if exist "%FINAL_PACKAGE_DIR%" goto :cleanup_error
if exist "%PACKAGE_ZIP%" goto :cleanup_error

echo [build] Running PyInstaller...
pyinstaller --noconfirm --workpath "%BUILD_DIR%" --distpath "%STAGE_ROOT%" LocalFullTextSearch.spec || goto :error

echo [build] Generating and verifying distribution manifest...
python tools\verify_distribution.py "%PACKAGE_DIR%" --stage-release-files --write-manifest --verify || goto :error

echo [build] Running frozen package smoke tests...
python tools\run_frozen_validations.py "%PACKAGE_EXE%" || goto :error

echo [build] Refreshing manifest with validation evidence...
python tools\verify_distribution.py "%PACKAGE_DIR%" --stage-release-files --write-manifest --verify || goto :error

echo [build] Creating and verifying release archive...
python tools\create_release_archive.py "%PACKAGE_DIR%" "%PACKAGE_ZIP%" --verify || goto :error

move "%PACKAGE_DIR%" "%FINAL_PACKAGE_DIR%" >nul || goto :error
if exist "%STAGE_ROOT%" rmdir /s /q "%STAGE_ROOT%"

echo [build] Done. Output: %FINAL_PACKAGE_DIR% and %PACKAGE_ZIP%
if /I not "%NO_PAUSE%"=="1" pause
exit /b 0

:cleanup_error
echo [build] Unable to clean version-specific output. Close any running copy and retry.
goto :error

:error
echo [build] Failed with error %ERRORLEVEL%.
if /I not "%NO_PAUSE%"=="1" pause
exit /b %ERRORLEVEL%
