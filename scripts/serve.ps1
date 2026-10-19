# Serve a converted GGUF with llama-server (Vulkan build) on localhost.
#   .\scripts\serve.ps1                     # OLMo 3 7B Instruct on :8080
#   .\scripts\serve.ps1 -Model olmo3-7b-think
param(
    [string]$Model = "olmo3-7b-instruct",
    [int]$Port = 8080,
    [int]$Ctx = 8192,
    [string]$Tools = "Z:\Coding\_tools",
    [string]$Build = "b11734"
)

$repo = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $Tools "llama.cpp\$Build\vulkan\llama-server.exe"
$gguf = Join-Path $Tools "models\$Model\$Model-Q8_0.gguf"
$template = Join-Path $repo "configs\olmo3_chat_template_patched.jinja"

foreach ($p in $exe, $gguf, $template) {
    if (-not (Test-Path $p)) { throw "missing: $p" }
}

# One slot, no prompt cache and a fixed seed, so repeated runs are as comparable as possible.
# Temperature is set per request, not here.
$serverArgs = @(
    "-m", $gguf,
    "--chat-template-file", $template,
    "-ngl", "99",
    "-c", $Ctx,
    "-np", "1",
    "-s", "0",
    "--no-cache-prompt",
    "--host", "127.0.0.1",
    "--port", $Port
)

Write-Host "$exe $($serverArgs -join ' ')"
& $exe @serverArgs
