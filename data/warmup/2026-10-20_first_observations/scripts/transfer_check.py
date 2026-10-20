import json, sys
from pathlib import Path

sys.path.insert(0, r"Z:\Coding\Portfolio\Matlab2PythonLLMBenchmarker\scripts")
from check_isolation import ask

sp = Path(sys.argv[1])
d = Path(r"Z:\Coding\Portfolio\Matlab2PythonLLMBenchmarker\tasks\t01_movavg")
plain = (d / "spec.md").read_text(encoding="utf-8")
amb = (d / "spec_ambiguous.md").read_text(encoding="utf-8")
url = "http://127.0.0.1:8080"

ref = json.loads((sp / "restart_1.json").read_text(encoding="utf-8"))
# restart_1 request positions: 1 amb(B), 2 plain(B), 3 amb(A), 4 plain(A)
steady = {"amb": ref[2]["text"], "plain": ref[3]["text"]}
first = {"amb": ref[0]["text"], "plain": ref[1]["text"]}

def txt(spec):
    return ask(url, spec, 600)["choices"][0]["message"]["content"]

for rnd in (1, 2):
    for label, spec in (("amb", amb), ("plain", plain)):
        t = txt(spec)
        mode = "steady (A)" if t == steady[label] else "first-mode (B)" if t == first[label] else "NEITHER"
        print(f"round {rnd} {label:5}: {mode}")
