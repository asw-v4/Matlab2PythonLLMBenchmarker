"""Check that requests to llama-server don't carry over state from earlier requests.

Needs a running server (scripts/serve.ps1). Sends each task's spec and ambiguous spec in an
interleaved order and checks, per request:
  - nothing was served from the server's prompt cache (cache_n == 0, cached_tokens == 0)
  - the whole prompt was processed (prompt_n == prompt_tokens)
It also compares replies to the same prompt before and after the other spec was sent.
Exit code 1 if any isolation check fails. Differences in reply text are reported but only
count as failures if cache reuse was seen, since GPU output is not guaranteed bit-identical.

    uv run python scripts/check_isolation.py [--task t01_movavg] [--url http://127.0.0.1:8080]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SYSTEM = "You translate MATLAB to Python. End your reply with a final line CONFIDENCE: 0-100."


def ask(url: str, spec: str, max_tokens: int) -> dict:
    body = {
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": spec}],
        "temperature": 0, "seed": 0, "max_tokens": max_tokens,
    }
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def summarise(label: str, r: dict) -> dict:
    t, u = r["timings"], r["usage"]
    return {
        "label": label,
        "text": r["choices"][0]["message"]["content"],
        "cache_n": t.get("cache_n", 0),
        "cached_tokens": u.get("prompt_tokens_details", {}).get("cached_tokens", 0),
        "prompt_n": t["prompt_n"],
        "prompt_tokens": u["prompt_tokens"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="t01_movavg")
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--max-tokens", type=int, default=600)
    args = ap.parse_args()

    task_dir = REPO / "tasks" / args.task
    plain = (task_dir / "spec.md").read_text(encoding="utf-8")
    amb = (task_dir / "spec_ambiguous.md").read_text(encoding="utf-8")

    # ambiguous alone, plain, ambiguous again, plain again, ambiguous back-to-back with itself
    order = [("amb#1", amb), ("plain#1", plain), ("amb#2", amb), ("plain#2", plain), ("amb#3", amb), ("amb#4", amb)]
    rows = [summarise(label, ask(args.url, spec, args.max_tokens)) for label, spec in order]

    failed = False
    print(f"{'request':9} {'prompt_tokens':>13} {'prompt_n':>9} {'cache_n':>8} {'cached_tokens':>14}")
    for r in rows:
        bad = r["cache_n"] != 0 or r["cached_tokens"] != 0 or r["prompt_n"] != r["prompt_tokens"]
        failed |= bad
        print(f"{r['label']:9} {r['prompt_tokens']:>13} {r['prompt_n']:>9} {r['cache_n']:>8} {r['cached_tokens']:>14}" + ("   <-- CACHE REUSE" if bad else ""))

    by = {r["label"]: r["text"] for r in rows}
    print()
    print("amb#1 == amb#2 (separated by plain):", by["amb#1"] == by["amb#2"])
    print("amb#3 == amb#4 (back to back):      ", by["amb#3"] == by["amb#4"])
    print("amb#1 == amb#3:                      ", by["amb#1"] == by["amb#3"])
    print("plain#1 == plain#2:                  ", by["plain#1"] == by["plain#2"])
    print("ISOLATION " + ("FAILED" if failed else "OK: no prompt-cache reuse between requests"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
