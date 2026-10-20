"""Canary warm-up for llama-server, with a full log of what happened.

A freshly started server was seen to answer some prompts differently the first time than on
later repeats, repeatably (see docs/warmup_study_notes.md). The canary sends one fixed prompt
repeatedly, with fixed sampling, until the last few replies are identical, and records every
request so this can be studied later. It does not make other prompts safe: a warmed server still
gave a different first reply to one task prompt. The canary prompt is unrelated to any task.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from bench.llama_client import chat

SCHEMA_VERSION = 1

CANARY_SYSTEM = (
    "You translate MATLAB to Python. Reply with the Python function in one code block, "
    "then a short note, then a final line CONFIDENCE: 0-100."
)
CANARY_USER = """Translate this MATLAB function to Python.

```matlab
function s = sumsq(x)
  s = 0;
  for i = 1:length(x)
    s = s + x(i)^2;
  end
end
```

Write a Python function `sumsq(x)`. `x` is a 2-D numpy array of shape `(1, n)`. Return a float.
"""

CANARY_PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": 400}


def canary_messages() -> list[dict]:
    return [{"role": "system", "content": CANARY_SYSTEM}, {"role": "user", "content": CANARY_USER}]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canary_sha() -> str:
    """Identifies the exact canary (messages and sampling) so logs from different versions can't be mixed."""
    return sha256_text(json.dumps({"messages": canary_messages(), "params": CANARY_PARAMS}, sort_keys=True))


def first_diff_index(a: str, b: str) -> int | None:
    """Index of the first differing character, or None if the strings are equal."""
    if a == b:
        return None
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def trailing_run(hashes: list[str]) -> tuple[int, int]:
    """(1-based index where the final run of identical values starts, its length). (0, 0) if empty."""
    if not hashes:
        return 0, 0
    n = 1
    while n < len(hashes) and hashes[-1 - n] == hashes[-1]:
        n += 1
    return len(hashes) - n + 1, n


def fingerprint(hashes: list[str]) -> str:
    """Compact pattern of distinct replies, lettered by first appearance, e.g. 'AABBBB'."""
    seen: dict[str, str] = {}
    out = []
    for h in hashes:
        if h not in seen:
            seen[h] = chr(ord("A") + len(seen)) if len(seen) < 26 else "?"
        out.append(seen[h])
    return "".join(out)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def collect_git_info(repo: Path) -> dict | None:
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True,
                                    text=True, check=True).stdout.strip())
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return None


def extract_props(props: dict) -> dict:
    """The parts of the server's /props worth keeping with a log."""
    gen = props.get("default_generation_settings", {})
    template = props.get("chat_template") or ""
    return {
        "build_info": props.get("build_info"),
        "model_path": props.get("model_path"),
        "total_slots": props.get("total_slots"),
        "n_ctx": gen.get("n_ctx"),
        "sampler_defaults": gen.get("params"),
        "chat_template_sha256": sha256_text(template) if template else None,
    }


def build_header(*, url: str, props: dict | None, git: dict | None, context: dict | None,
                 note: str | None, settings: dict) -> dict:
    return {
        "type": "header",
        "schema_version": SCHEMA_VERSION,
        "started_utc": utc_now(),
        "url": url,
        "canary_sha256": canary_sha(),
        "canary_params": CANARY_PARAMS,
        "settings": settings,
        "server": extract_props(props) if props else None,
        "git": git,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
        "context": context or {},
        "note": note,
    }


def run_warmup(url: str, *, n_max: int = 12, stable_run: int = 4, min_requests: int = 6,
               ask: Callable = chat, sink: Callable[[dict], None] | None = None,
               metrics: Callable[[], str | None] | None = None) -> tuple[dict, list[dict]]:
    """Send the canary until the last `stable_run` replies are identical (and at least
    `min_requests` have been sent), or `n_max` is reached. Returns (summary, request records).

    `sink` is called with each record as it is produced, so a crash keeps partial data.
    """
    if not 1 <= stable_run <= min_requests <= n_max:
        raise ValueError("need 1 <= stable_run <= min_requests <= n_max")
    emit = sink or (lambda rec: None)
    messages = canary_messages()
    records: list[dict] = []
    hashes: list[str] = []
    texts: list[str] = []
    done = False
    for i in range(1, n_max + 1):
        t_utc = utc_now()
        t0 = time.perf_counter()
        resp = ask(url, messages, **CANARY_PARAMS)
        latency = time.perf_counter() - t0
        text = resp["choices"][0]["message"]["content"]
        timings = resp.get("timings", {})
        usage = resp.get("usage", {})
        h = sha256_text(text)
        rec = {
            "type": "request", "i": i, "t_utc": t_utc, "latency_s": round(latency, 4),
            "reply_sha256": h, "reply_chars": len(text), "reply": text,
            "finish_reason": resp["choices"][0].get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n"),
            "prompt_per_second": timings.get("prompt_per_second"),
            "predicted_per_second": timings.get("predicted_per_second"),
            "first_diff_vs_previous": first_diff_index(texts[-1], text) if texts else None,
            "metrics_raw": (metrics() or None) if metrics else None,
        }
        records.append(rec)
        hashes.append(h)
        texts.append(text)
        emit(rec)
        start, length = trailing_run(hashes)
        if i >= min_requests and length >= stable_run:
            done = True
            break

    start, length = trailing_run(hashes)
    summary = {
        "type": "summary",
        "finished_utc": utc_now(),
        "ok": done,
        "n_requests": len(records),
        "fingerprint": fingerprint(hashes),
        "stable_from": start,
        "stable_run_length": length,
        "transient_requests": start - 1,
        "distinct_replies": len(set(hashes)),
        "final_reply_sha256": hashes[-1],
        "first_vs_final_diff": first_diff_index(texts[0], texts[-1]),
    }
    emit(summary)
    return summary, records
