@echo off
setlocal

set "PORT=8000"
set "ROOT_DIR=%~dp0"
cd /d "%ROOT_DIR%"

where python >nul 2>nul
if errorlevel 1 (
  echo Python is required to run this script.
  echo Install Python or run your own static server in this folder.
  pause
  exit /b 1
)

start "" http://localhost:%PORT%/index.html
start "" /min python -m http.server "%PORT%"

echo.
echo Running game at http://localhost:%PORT%/index.html
echo Server is using python -m http.server on port %PORT%.
echo Close this window when you are done.

pause
