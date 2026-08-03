@echo off
setlocal
cd /d "%~dp0"
set "LFTS_APP_DATA_DIR=%~dp0failure-fallback-demo-data"
for %%F in ("%~dp0*.exe") do set "LFTS_DEMO_EXE=%%~fF"
if not defined LFTS_DEMO_EXE exit /b 1
start "" "%LFTS_DEMO_EXE%" --failure-fallback-demo
