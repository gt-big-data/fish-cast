@echo off
rem Double-click to set up GitHub for FishCast (see scripts\setup_github_windows.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\setup_github_windows.ps1"
echo.
pause
