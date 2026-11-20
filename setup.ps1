# GazeDot first-run setup (Windows PowerShell)
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

Write-Host "== GazeDot setup ==" -ForegroundColor Cyan

$version = $null
foreach ($v in @("3.11", "3.12", "3.10")) {
    try {
        py "-$v" -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $version = $v; break }
    } catch { }
}
if (-not $version) {
    Write-Host "Python 3.10-3.12 が見つかりません。https://www.python.org/downloads/ から 3.11 をインストールしてください。" -ForegroundColor Red
    exit 1
}
Write-Host "Python $version を使用します"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py "-$version" -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install --upgrade pip --quiet
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt --quiet

New-Item -ItemType Directory -Force models | Out-Null
if (-not (Test-Path "models\face_landmarker.task")) {
    Write-Host "顔ランドマークモデルを取得中..." -ForegroundColor Cyan
    Invoke-WebRequest `
        -Uri "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" `
        -OutFile "models\face_landmarker.task"
}

Write-Host "完了しました。run.bat で起動できます。" -ForegroundColor Green
