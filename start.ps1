$ErrorActionPreference = 'Stop'
$studioRoot = $PSScriptRoot
$pythonPath = $env:DRAMA_PYTHON
if (-not $pythonPath -and $env:IRODORI_HOME) {
    $candidate = Join-Path $env:IRODORI_HOME '.venv\Scripts\python.exe'
    if (Test-Path -LiteralPath $candidate) { $pythonPath = $candidate }
}
if (-not $pythonPath) { $pythonPath = (Get-Command python -ErrorAction Stop).Source }
$env:PYTHONIOENCODING = 'utf-8'
$serviceRoot = if ($env:DRAMA_DATA_DIR) { $env:DRAMA_DATA_DIR } else { Join-Path $studioRoot 'data' }
New-Item -ItemType Directory -Path $serviceRoot -Force | Out-Null
try { $null = Invoke-RestMethod 'http://127.0.0.1:8787/api/health' -TimeoutSec 2 }
catch {
    Start-Process -FilePath $pythonPath -ArgumentList ('"' + (Join-Path $studioRoot 'server.py') + '"') -WorkingDirectory $studioRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $serviceRoot 'server.log') -RedirectStandardError (Join-Path $serviceRoot 'server-error.log')
    $ready = $false
    for ($attempt=0;$attempt -lt 30;$attempt++) {
        Start-Sleep -Milliseconds 500
        try { $null = Invoke-RestMethod 'http://127.0.0.1:8787/api/health' -TimeoutSec 1; $ready = $true; break } catch {}
    }
    if (-not $ready) { throw 'Server startup failed. Check the server-error.log file in the data directory.' }
}
Start-Process 'http://127.0.0.1:8787'
