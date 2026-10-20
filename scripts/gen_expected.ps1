# Regenerate a task's Octave answer key inside the Ubuntu-Bench WSL distro.
#   .\scripts\gen_expected.ps1 t01_movavg
param([Parameter(Mandatory)][string]$Task)

$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $repo "tasks\$Task\generate_expected.m"
if (-not (Test-Path $script)) { throw "missing: $script" }

$wslScript = (wsl -d Ubuntu-Bench -- wslpath -a ($script -replace '\\', '/')).Trim()
wsl -d Ubuntu-Bench -- octave --no-gui --no-window-system $wslScript
if ($LASTEXITCODE -ne 0) { throw "octave failed ($LASTEXITCODE)" }
