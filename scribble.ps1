#!/usr/bin/env pwsh
# Usage: ./scribble.ps1 VIDEO_OR_LINK [-o OUTDIR] [--no-video] ...   (./scribble.ps1 -h for all options)
# Cross-platform launcher (PowerShell 7+ on Windows, macOS, Linux; also Windows PowerShell 5.1).
# Arguments, including links containing '&', are passed through untouched.

$script = Join-Path (Join-Path $PSScriptRoot 'scripts') 'make_scribble.py'  # 2-arg form works in 5.1 too

# Pick a Python 3. On Windows `python3` is often the Microsoft Store stub, so prefer `python`
# there; elsewhere `python` may be missing or Python 2, so prefer `python3`.
$onWindows = $IsWindows -or $PSVersionTable.PSEdition -eq 'Desktop'
$candidates = if ($onWindows) { 'python', 'py', 'python3' } else { 'python3', 'python' }
$python = $candidates | Where-Object { Get-Command $_ -CommandType Application -ErrorAction SilentlyContinue } |
    Select-Object -First 1
if (-not $python) {
    Write-Error 'Python 3 not found on PATH. Install it, then: python -m pip install -r requirements.txt'
    exit 1
}

& $python -I $script @args
exit $LASTEXITCODE
