param(
    [switch]$Release,
    [switch]$Prod,
    [switch]$NoVoice,
    [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backend = Join-Path $root 'backend'
if ($Prod) {
    Write-Host 'Фронтенд теперь собирается в Docker; -Prod больше не требуется.' -ForegroundColor Yellow
}
$envFile = Join-Path $backend '.env'
if (-not (Test-Path -LiteralPath $envFile)) {
    Copy-Item -LiteralPath (Join-Path $backend '.env.example') -Destination $envFile
    Write-Host 'Создан backend\.env. Заполните GIGACHAT_AUTH_KEY и повторите запуск.' -ForegroundColor Yellow
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host 'Docker не найден. Установите Docker Desktop или Docker Engine.' -ForegroundColor Red
    exit 1
}
& docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Docker недоступен. Запустите Docker Desktop или Docker Engine.' -ForegroundColor Red
    exit 1
}

$compose = @('compose', '-f', (Join-Path $backend 'docker-compose.yml'))
if ($Release) {
    $compose += @('-f', (Join-Path $backend 'compose.release.yml'))
}
if ($NoVoice) {
    $env:VOICE_ENABLED = 'false'
    $env:TTS_ENABLED = 'false'
}

Push-Location $backend
try {
    & docker @compose config --quiet
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & docker @compose up -d --build --remove-orphans
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & docker @compose ps
} finally {
    Pop-Location
}

if ($Release) {
    Write-Host 'Стек запущен. Проверьте HTTPS-адрес вашего домена и статус контейнеров.' -ForegroundColor Green
} else {
    $url = 'http://localhost:8080'
    Write-Host "Откройте $url" -ForegroundColor Green
    if (-not $NoBrowser) { Start-Process $url }
}
