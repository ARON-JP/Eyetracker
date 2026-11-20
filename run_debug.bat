@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo 初回セットアップを実行します...
  powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)
".venv\Scripts\python.exe" "%~dp0GazeDot_debug.py"
pause
