"""Run the pilot against a running llama-server (scripts/serve.ps1), or dry-run it with golden answers.

    uv run python scripts/run_pilot.py --dry-run                 # no model: replays the golden solutions
    uv run python scripts/run_pilot.py --tasks t01_movavg        # the real thing, one task
    uv run python scripts/run_pilot.py                           # all tasks, plain and ambiguous specs

Writes runs/pilot/<UTC time>/run.jsonl (header, one record per sample, summary) and prints a report.
Not part of the Inspect integration: that will drive the same functions.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bench import pilot, prompts, report, sandbox, warmup  # noqa: E402
from bench.llama_client import chat, get_json  # noqa: E402
from bench.runlog import RunLog  # noqa: E402
from bench.taskio import load_task  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tasks", nargs="*", help="task folder names (default: all)")
    ap.add_argument("--variants", nargs="+", default=["plain", "ambiguous"], choices=["plain", "ambiguous"])
    ap.add_argument("--conditions", nargs="+", default=["free"], choices=sorted(prompts.CONDITIONS))
    ap.add_argument("--max-attempts", type=int, default=6, help="most requests per prompt when confirming")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--out", type=Path, help="run folder (default runs/pilot/<UTC time>)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the canary warm-up")
    ap.add_argument("--dry-run", action="store_true", help="no model: replay the golden solutions")
    ap.add_argument("--manifest", type=Path, default=REPO / "models" / "manifests" / "olmo3-7b-instruct.json")
    ap.add_argument("--template", type=Path, default=REPO / "configs" / "olmo3_chat_template_patched.jinja")
    ap.add_argument("--note")
    ap.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    task_dirs = sorted(p for p in (REPO / "tasks").iterdir() if (p / "task.toml").exists())
    if args.tasks:
        task_dirs = [p for p in task_dirs if p.name in set(args.tasks)]
        if not task_dirs:
            print(f"no such task(s): {args.tasks}", file=sys.stderr)
            return 1
    tasks = [load_task(p) for p in task_dirs]

    cfg = sandbox.SandboxConfig()
    if not sandbox.docker_available(cfg):
        print("sandbox image not found; build it with scripts/build_sandbox.ps1", file=sys.stderr)
        return 1

    params = {**pilot.DEFAULT_PARAMS, "max_tokens": args.max_tokens}
    props = warmup_summary = ask = None
    if args.dry_run:
        ask = pilot.dry_run_ask(tasks)
    else:
        try:
            props = get_json(args.url, "/props")
        except OSError as e:
            print(f"cannot reach {args.url}: {e}\nstart the server with scripts/serve.ps1", file=sys.stderr)
            return 1
        ask = chat
        if not args.no_warmup:
            print("warming up with the canary ...")
            warmup_summary, _ = warmup.run_warmup(args.url)
            print(f"  canary fingerprint {warmup_summary['fingerprint']}, stable from request "
                  f"{warmup_summary['stable_from']}, ok={warmup_summary['ok']}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else None
    template_sha = warmup.sha256_text(args.template.read_text(encoding="utf-8")) if args.template.exists() else None
    out = args.out or REPO / "runs" / "pilot" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    header = pilot.build_run_header(
        tasks=tasks, conditions=args.conditions, variants=args.variants, params=params,
        max_attempts=args.max_attempts, url=args.url, props=props, manifest=manifest, template_file_sha=template_sha,
        sandbox_cfg=cfg, image_id=sandbox.image_id(cfg), git=warmup.collect_git_info(REPO), warmup=warmup_summary,
        lock_file=REPO / "uv.lock", note=args.note, context=dict(kv.split("=", 1) for kv in args.context),
        dry_run=args.dry_run)

    def progress(s: dict) -> None:
        e, c = s["evaluation"], s["confirmation"]
        what = e["outcome"] if s["variant"] == "ambiguous" else ("pass" if e["passed"] else e["signature"])
        extra = "  [first reply differed]" if c["first_reply_differs"] else "  [UNCONFIRMED]" if not c["confirmed"] else ""
        print(f"  {s['task_id']:20} {s['variant']:9} {s['condition']:6} {what:22} "
              f"conf={e['confidence']} attempts={c['n_attempts']}{extra}")

    print(f"{len(tasks)} task(s), variants {args.variants}, conditions {args.conditions}"
          + ("  (DRY RUN)" if args.dry_run else ""))
    with RunLog(out / "run.jsonl") as log:
        log.write(header)
        samples = pilot.run_pilot(tasks, variants=args.variants, conditions=args.conditions, url=args.url, ask=ask,
                                  runner=lambda code, task: sandbox.run_code(code, task, cfg), params=params,
                                  max_attempts=args.max_attempts, log=log, on_sample=progress)
        log.write(pilot.summary_record(samples))
    print()
    print(report.format_report(report.summarise(samples)))
    print(f"\nlog: {out / 'run.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
