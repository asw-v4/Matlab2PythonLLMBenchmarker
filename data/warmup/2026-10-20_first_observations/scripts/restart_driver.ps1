$env:UV_CACHE_DIR='Z:\Coding\_tools\uv\cache'; $env:UV_PYTHON_INSTALL_DIR='Z:\Coding\_tools\uv\python'; Remove-Item Env:VIRTUAL_ENV -ErrorAction SilentlyContinue
$t = 'Z:\Coding\_tools'
$sp = 'C:\Users\award\AppData\Local\Temp\claude\z--Coding\d4b45327-265e-463d-8a49-88f52820c4e9\scratchpad'
$repo = 'Z:\Coding\Portfolio\Matlab2PythonLLMBenchmarker'
$exe = "$t\llama.cpp\b11734\vulkan\llama-server.exe"
$args = @('-m', "$t\models\olmo3-7b-instruct\olmo3-7b-instruct-Q8_0.gguf",
          '--chat-template-file', "$repo\configs\olmo3_chat_template_patched.jinja",
          '-ngl','99','-c','8192','-np','1','-s','0','--no-cache-prompt','--host','127.0.0.1','--port','8080')
Set-Location $repo
foreach ($run in 1..6) {
  $p = Start-Process -FilePath $exe -ArgumentList $args -PassThru -WindowStyle Hidden -RedirectStandardOutput "$sp\rr_srv_$run.out" -RedirectStandardError "$sp\rr_srv_$run.err"
  $ready = $false
  foreach ($i in 1..120) {
    try { $h = Invoke-RestMethod http://127.0.0.1:8080/health -TimeoutSec 2; if ($h.status -eq 'ok') { $ready = $true; break } } catch {}
    Start-Sleep -Milliseconds 500
  }
  if (-not $ready) { "run $run : server never became ready"; Stop-Process -Id $p.Id -Force; continue }
  & 'Z:\miniforge3\Library\bin\uv.exe' run python "$sp\restart_runs.py" "$sp\restart_$run.json" 2>$null
  Stop-Process -Id $p.Id -Force
  Start-Sleep -Seconds 3
  "run $run done"
}
