# Build the sandbox image inside the Ubuntu-Bench WSL distro (needs network, once).
#   .\scripts\build_sandbox.ps1
param([string]$Tag = "matlab2py-sandbox:1")

$repo = Split-Path -Parent $PSScriptRoot
$dockerDir = (wsl -d Ubuntu-Bench --exec wslpath -a (Join-Path $repo "docker" | ForEach-Object { $_ -replace '\\', '/' })).Trim()
wsl -d Ubuntu-Bench --exec docker build -t $Tag $dockerDir
if ($LASTEXITCODE -ne 0) { throw "docker build failed ($LASTEXITCODE)" }
wsl -d Ubuntu-Bench --exec docker image inspect --format '{{.Id}}  {{.Size}} bytes' $Tag
