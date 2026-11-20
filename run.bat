@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo 初回セットアップを実行します...
  powershell -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
)
start "" ".venv\Scripts\pythonw.exe" "%~dp0GazeDot.pyw"
