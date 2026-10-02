# Setup + scrape for the Polymarket BTC 5-minute dataset (Windows PowerShell).
# Safe to re-run: responses are cached in .\.cache, so an interrupted scrape resumes.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot ".")

$Py      = if ($env:PY) { $env:PY } else { "python" }
$Count   = if ($env:COUNT) { $env:COUNT } else { 500 }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { 6 }
$Rps     = if ($env:RPS) { $env:RPS } else { 6 }

Write-Host "== python =="
& $Py --version

Write-Host "`n== dependencies =="
& $Py -c "import requests" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "installing requests (falls back to stdlib urllib if this fails)"
    & $Py -m pip install --user requests
    if ($LASTEXITCODE -ne 0) { Write-Warning "pip install failed; the script will use urllib" }
} else {
    Write-Host "requests already installed"
}

Write-Host "`n== scrape: $Count markets =="
& $Py scripts\scrape_btc5m.py --count $Count --workers $Workers --rps $Rps

Write-Host "`n== done =="
Get-ChildItem data\ | Format-Table Name, Length
