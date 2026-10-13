param(
    [switch]$Prod,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $backend ".venv\Scripts\python.exe"

function Step($text) { Write-Host "==> $text" -ForegroundColor Cyan }

function Wait-Port($port, $seconds) {
    $deadline = (Get-Date).AddSeconds($seconds)
    while ((Get-Date) -lt $deadline) {
        $client = New-Object System.Net.Sockets.TcpClient
        try { $client.Connect("127.0.0.1", $port); return $true } catch { Start-Sleep -Milliseconds 500 } finally { $client.Dispose() }
    }
    return $false
}

foreach ($port in 8000, 5173) {
    if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
        if ($Prod -and $port -eq 5173) { continue }
        Write-Host "Порт $port уже занят — возможно, ассистент уже запущен. Закройте старые окна и повторите." -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path $python)) {
    Step "Создаю виртуальное окружение backend\.venv"
    & python -m venv (Join-Path $backend ".venv")
    & $python -m pip install --disable-pip-version-check -q -r (Join-Path $backend "requirements-dev.txt")
}

if (-not (Test-Path (Join-Path $backend ".env"))) {
    Copy-Item (Join-Path $backend ".env.example") (Join-Path $backend ".env")
    Write-Host "Создан backend\.env — впишите ключ GigaChat в GIGACHAT_CREDENTIALS (без ключа работают правила)." -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Step "Устанавливаю зависимости фронтенда"
    Push-Location $frontend; npm install; Pop-Location
}

$xtts = Join-Path $backend ".cache\tts\xtts_v2\model.pth"
if (-not (Test-Path $xtts)) {
    & $python -c "import torch, TTS" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Step "Скачиваю модели озвучки и поиска по смыслу (около 2,5 ГБ, только при первом запуске)"
        Push-Location $backend
        & $python -m assistant.models_download
        Pop-Location
    } else {
        Write-Host "torch не установлен — озвучка голосами и поиск по смыслу выключены (см. README, раздел «Голос»)." -ForegroundColor Yellow
    }
}

$backendCmd ="Set-Location '$backend'; & '$python' -m uvicorn app:app --host 127.0.0.1 --port 8000"

if ($Prod) {
    Step "Собираю фронтенд"
    Push-Location $frontend
    $env:VITE_USE_MOCKS = "false"; $env:VITE_API_URL = ""
    npm run build
    Pop-Location
    Step "Запускаю сервер (фронт и API на одном порту)"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
    $url = "http://127.0.0.1:8000"
} else {
    Step "Запускаю backend и frontend в отдельных окнах"
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontend'; npm run dev"
    $url = "http://127.0.0.1:5173"
}

if (Wait-Port 8000 180) { Write-Host "Backend готов" -ForegroundColor Green } else { Write-Host "Backend не ответил за 3 минуты — смотрите его окно" -ForegroundColor Red }
if (-not $Prod -and -not (Wait-Port 5173 60)) { Write-Host "Frontend не ответил — смотрите его окно" -ForegroundColor Red }

Write-Host "Откройте $url" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process $url }
