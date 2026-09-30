# Starts the MLBench Studio frontend and API on this computer.
# It is intentionally localhost-only: no service is exposed to the network.

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
$frontend = Join-Path $projectRoot 'frontend'
$logs = Join-Path $projectRoot 'tmp'

function Test-ListeningPort([int] $Port) {
    return $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1)
}

if (-not (Test-Path $python)) {
    throw "Python virtual environment was not found: $python. Install dependencies first."
}

New-Item -ItemType Directory -Force -Path $logs | Out-Null

if (-not (Test-ListeningPort 8765)) {
    Start-Process -FilePath $python `
        -ArgumentList @('-m', 'uvicorn', 'backend.server.app:app', '--host', '127.0.0.1', '--port', '8765') `
        -WorkingDirectory $projectRoot `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs 'backend-stdout.log') `
        -RedirectStandardError (Join-Path $logs 'backend-stderr.log') | Out-Null
    Write-Host 'Backend started: 127.0.0.1:8765'
} else {
    Write-Host 'Backend already running: 127.0.0.1:8765'
}

if (-not (Test-ListeningPort 5173)) {
    Start-Process -FilePath 'pnpm.cmd' `
        -ArgumentList @('run', 'dev', '--', '--host', '127.0.0.1') `
        -WorkingDirectory $frontend `
        -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logs 'frontend-stdout.log') `
        -RedirectStandardError (Join-Path $logs 'frontend-stderr.log') | Out-Null
    Write-Host 'Frontend started: 127.0.0.1:5173'
} else {
    Write-Host 'Frontend already running: 127.0.0.1:5173'
}

for ($attempt = 1; $attempt -le 12; $attempt++) {
    try {
        $null = Invoke-RestMethod -Uri 'http://127.0.0.1:8765/api/health' -TimeoutSec 2
        break
    } catch {
        if ($attempt -eq 12) { throw 'Backend did not start. Check tmp/backend-stderr.log.' }
        Start-Sleep -Seconds 1
    }
}

Write-Host ''
Write-Host 'Local experiment studio is ready:'
Write-Host 'http://127.0.0.1:5173/mlbench-studio/'
