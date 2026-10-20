import json, sys
from pathlib import Path

sys.path.insert(0, r"Z:\Coding\Portfolio\Matlab2PythonLLMBenchmarker\scripts")
from check_isolation import ask

d = Path(r"Z:\Coding\Portfolio\Matlab2PythonLLMBenchmarker\tasks\t01_movavg")
plain = (d / "spec.md").read_text(encoding="utf-8")
amb = (d / "spec_ambiguous.md").read_text(encoding="utf-8")
url = "http://127.0.0.1:8080"

seq = [("amb", amb), ("plain", plain), ("amb", amb), ("plain", plain), ("plain", plain), ("plain", plain), ("amb", amb)]
out = []
for i, (label, spec) in enumerate(seq, 1):
    r = ask(url, spec, 600)
    out.append({"n": i, "label": label, "text": r["choices"][0]["message"]["content"],
                "tokens": r["usage"]["completion_tokens"], "cache_n": r["timings"].get("cache_n", 0)})
Path(sys.argv[1]).write_text(json.dumps(out), encoding="utf-8")
