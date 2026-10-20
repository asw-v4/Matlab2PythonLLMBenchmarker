"""Run the benchmark over a set of tasks, without Inspect: the loop that Inspect will later drive.

For each task, spec variant (plain / ambiguous) and prompt condition (free / format):
    fresh conversation -> confirm the reply (repeat until two in a row match) -> evaluate -> log.
"""
from __future__ import annotations

import dataclasses
import platform
from pathlib import Path
from typing import Callable

from bench import prompts, report
from bench import sandbox as sandbox_mod
from bench.confirm import confirm_reply
from bench.evaluate import AMBIGUOUS, PLAIN, evaluate_reply
from bench.runlog import SCHEMA_VERSION, RunLog, hash_task, sha256_bytes
from bench.warmup import extract_props, utc_now

SPEC_FILES = {PLAIN: "spec.md", AMBIGUOUS: "spec_ambiguous.md"}
DEFAULT_PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": 1500}


def spec_text(task, variant: str) -> str:
    return (Path(task.dir) / SPEC_FILES[variant]).read_text(encoding="utf-8")


def build_run_header(*, tasks, conditions, variants, params, max_attempts, url, props=None, manifest=None,
                     template_file_sha=None, sandbox_cfg=None, image_id=None, git=None, warmup=None,
                     lock_file: Path | None = None, note=None, context=None, dry_run=False) -> dict:
    cfg = dataclasses.asdict(sandbox_cfg) if sandbox_cfg else None
    if cfg:
        cfg.pop("workroot", None)
    return {
        "type": "header", "schema_version": SCHEMA_VERSION, "started_utc": utc_now(), "dry_run": dry_run,
        "url": url, "note": note, "context": context or {},
        "server": extract_props(props) if props else None,
        "model": manifest,
        "chat_template_file_sha256": template_file_sha,
        "prompts": {c: prompts.sha256(prompts.system_prompt(c)) for c in conditions},
        "conditions": list(conditions), "variants": list(variants),
        "tasks": {t.id: {"sha256": hash_task(t.dir), "octave_version": t.octave_version, "n_cases": len(t.cases)}
                  for t in tasks},
        "sampling": params, "max_attempts": max_attempts,
        "sandbox": {"config": cfg, "image_id": image_id},
        "lock_file_sha256": sha256_bytes(Path(lock_file).read_bytes()) if lock_file and Path(lock_file).exists() else None,
        "warmup": warmup, "git": git,
        "host": {"platform": platform.platform(), "python": platform.python_version()},
    }


def run_sample(task, variant: str, condition: str, *, url: str, ask: Callable, runner: Callable,
               params: dict, max_attempts: int) -> dict:
    spec = spec_text(task, variant)
    messages = prompts.build_messages(condition, spec)
    confirmed = confirm_reply(url, messages, params=params, max_attempts=max_attempts, ask=ask)

    ev = evaluate_reply(confirmed.reply, task, variant=variant, condition=condition, runner=runner)
    first_ev = None
    if confirmed.first_reply_differs:
        first_ev = evaluate_reply(confirmed.first_reply, task, variant=variant, condition=condition, runner=runner)

    replies = {}
    for a in confirmed.attempts:
        replies.setdefault(a.reply_sha256, a.text)
    return {
        "type": "sample", "task_id": task.id, "variant": variant, "condition": condition,
        "prompt": {"system_sha256": prompts.sha256(messages[0]["content"]),
                   "user_sha256": prompts.sha256(messages[1]["content"])},
        "confirmation": {
            "confirmed": confirmed.confirmed, "n_attempts": confirmed.n_attempts,
            "first_reply_differs": confirmed.first_reply_differs, "truncated": confirmed.truncated,
            "attempts": [a.as_dict(with_text=False) for a in confirmed.attempts],
        },
        "replies": replies,                      # each distinct reply once, keyed by its hash
        "reply_sha256": confirmed.attempts[-1].reply_sha256,
        "evaluation": ev.as_dict(),
        "first_reply_evaluation": first_ev.as_dict() if first_ev else None,
    }


def run_pilot(tasks, *, variants, conditions, url: str, ask: Callable, runner: Callable, params: dict,
              max_attempts: int, log: RunLog | None = None,
              on_sample: Callable[[dict], None] | None = None) -> list[dict]:
    samples = []
    for task in sorted(tasks, key=lambda t: t.id):
        for variant in variants:
            for condition in conditions:
                s = run_sample(task, variant, condition, url=url, ask=ask, runner=runner, params=params,
                               max_attempts=max_attempts)
                samples.append(s)
                if log:
                    log.write(s)
                if on_sample:
                    on_sample(s)
    return samples


def summary_record(samples: list[dict]) -> dict:
    s = report.summarise(samples)
    groups = [{"task_id": k[0], "variant": k[1], "condition": k[2], **g} for k, g in s["groups"].items()]
    return {"type": "summary", "finished_utc": utc_now(), "n_samples": s["n_samples"],
            "calibration_plain": dataclasses.asdict(s["calibration_plain"]), "groups": groups}


def golden_replies(tasks) -> dict[str, str]:
    """For a dry run: the reply a perfect model would give to each spec, keyed by the user message."""
    out = {}
    for t in tasks:
        code = (Path(t.dir) / "solution.py").read_text(encoding="utf-8")
        reply = f"```python\n{code}\n```\n\nCONFIDENCE: 100\n"
        for variant in (PLAIN, AMBIGUOUS):
            out[prompts.build_messages("free", spec_text(t, variant))[1]["content"]] = reply
    return out


def dry_run_ask(tasks) -> Callable:
    table = golden_replies(tasks)

    def ask(url, messages, **params):
        user = next(m["content"] for m in messages if m["role"] == "user")
        return {"choices": [{"message": {"content": table[user]}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 0, "prompt_tokens": 0}, "timings": {"prompt_n": 0, "cache_n": 0}}

    return ask
