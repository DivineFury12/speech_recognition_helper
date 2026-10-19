param(
    [switch]$Prod,
    [switch]$NoBrowser,
    [switch]$NoVoice
)

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$python = Join-Path $backend ".venv\Scripts\python.exe"
$dbContainer = "assistant-db"
$dbPort = 15432

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

Step "Проверяю Docker для базы PostgreSQL"
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $desktop) { Start-Process $desktop }
    $deadline = (Get-Date).AddMinutes(3)
    do { Start-Sleep -Seconds 3; & docker info *> $null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) { Write-Host "Docker не запустился. Запустите Docker Desktop и повторите." -ForegroundColor Red; exit 1 }
}
$exists = & docker ps -a --filter "name=^$dbContainer$" --format "{{.Names}}"
if (-not $exists) {
    Step "Создаю базу $dbContainer (PostgreSQL + pgvector, порт $dbPort)"
    & docker run -d --name $dbContainer -e POSTGRES_USER=assistant -e POSTGRES_PASSWORD=assistant `
        -e POSTGRES_DB=assistant -p "127.0.0.1:${dbPort}:5432" -v assistant-pgdata:/var/lib/postgresql/data `
        pgvector/pgvector:0.8.7-pg17-bookworm | Out-Null
} else {
    & docker start $dbContainer | Out-Null
}
$deadline = (Get-Date).AddSeconds(60)
do { Start-Sleep -Seconds 1; & docker exec $dbContainer pg_isready -U assistant *> $null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)

if (-not (Test-Path $python)) {
    Step "Создаю виртуальное окружение backend\.venv"
    & python -m venv (Join-Path $backend ".venv")
    & $python -m pip install --disable-pip-version-check -q -r (Join-Path $backend "requirements.txt") pytest
    Write-Host "Для голоса и поиска по смыслу нужны torch и GigaAM — команды в README, раздел «Голос»." -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $backend ".env"))) {
    Copy-Item (Join-Path $backend ".env.example") (Join-Path $backend ".env")
    Write-Host "Создан backend\.env — впишите ключ GigaChat в GIGACHAT_AUTH_KEY." -ForegroundColor Yellow
}

if (-not $NoVoice) {
    $missing = & $python -c "import importlib.util; print(', '.join(name for name in ('torch', 'transformers', 'gigaam') if importlib.util.find_spec(name) is None))"
    if ($LASTEXITCODE -ne 0 -or $missing) {
        Write-Host "Не установлены зависимости голосового режима: $missing" -ForegroundColor Red
        Write-Host "Установите torch и gigaam по инструкции в README или запустите текстовый режим: .\run.ps1 -Prod -NoVoice" -ForegroundColor Yellow
        exit 1
    }
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Step "Устанавливаю зависимости фронтенда"
    Push-Location $frontend; npm install; Pop-Location
}

$env:DATABASE_URL = "postgresql+psycopg://assistant:assistant@127.0.0.1:$dbPort/assistant"
Step "Обновляю схему базы (alembic upgrade head)"
Push-Location $backend
& $python -m alembic upgrade head
Pop-Location

$envLine = "`$env:DATABASE_URL = '$($env:DATABASE_URL)'; "
if ($NoVoice) { $envLine += "`$env:VOICE_ENABLED = 'false'; " }
if ($Prod) { $envLine += "`$env:FRONTEND_DIST = '$(Join-Path $frontend 'dist')'; " }
$backendCmd = "$envLine`$host.UI.RawUI.WindowTitle = 'Ассистент: backend'; Set-Location '$backend'; & '$python' -m uvicorn app:app --host 127.0.0.1 --port 8000"

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
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle = 'Ассистент: frontend'; Set-Location '$frontend'; npm run dev"
    $url = "http://127.0.0.1:5173"
}

if (Wait-Port 8000 240) { Write-Host "Backend готов" -ForegroundColor Green } else { Write-Host "Backend не ответил за 4 минуты — смотрите его окно" -ForegroundColor Red }
if (-not $Prod -and -not (Wait-Port 5173 60)) { Write-Host "Frontend не ответил — смотрите его окно" -ForegroundColor Red }

Write-Host "Откройте $url" -ForegroundColor Green
if (-not $NoBrowser) { Start-Process $url }
