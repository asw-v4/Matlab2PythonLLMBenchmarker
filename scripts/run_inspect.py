"""Run the pilot through Inspect, against a running llama-server (scripts/serve.ps1), or dry-run it.

    uv run python scripts/run_inspect.py --dry-run               # no server: a mock model gives the golden answers
    uv run python scripts/run_inspect.py --tasks t01_movavg      # the real model, one task
    uv run python scripts/run_inspect.py                         # all tasks, plain and ambiguous specs

Writes Inspect's log (runs/inspect/<UTC time>/*.eval) and prints our report. Inspect's own viewer works on the
log:  uv run inspect view --log-dir runs/inspect/<UTC time>
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import ModelOutput, get_model  # noqa: E402

from bench import inspect_pilot, pilot, prompts, report, sandbox, warmup  # noqa: E402
from bench.llama_client import get_json  # noqa: E402


def mock_model(tasks):
    """A model that answers every spec with the golden solution, for a dry run."""
    table = pilot.golden_replies(tasks)

    def answer(messages, tools, tool_choice, config):
        user = next(m.text for m in messages if m.role == "user")
        return ModelOutput.from_content(model="mockllm", content=table[user])

    return get_model("mockllm/model", custom_outputs=answer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--tasks", nargs="*", help="task folder names (default: all)")
    ap.add_argument("--variants", nargs="+", default=["plain", "ambiguous"], choices=["plain", "ambiguous"])
    ap.add_argument("--conditions", nargs="+", default=["free"], choices=sorted(prompts.CONDITIONS))
    ap.add_argument("--max-attempts", type=int, default=6, help="most requests per prompt when confirming")
    ap.add_argument("--out", type=Path, help="log folder (default runs/inspect/<UTC time>)")
    ap.add_argument("--no-warmup", action="store_true", help="skip the canary warm-up")
    ap.add_argument("--dry-run", action="store_true", help="no server: a mock model returns the golden solutions")
    ap.add_argument("--manifest", type=Path, default=REPO / "models" / "manifests" / "olmo3-7b-instruct.json")
    ap.add_argument("--template", type=Path, default=REPO / "configs" / "olmo3_chat_template_patched.jinja")
    ap.add_argument("--note")
    ap.add_argument("--context", action="append", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()

    ids = args.tasks or inspect_pilot.all_task_ids()
    unknown = sorted(set(ids) - set(inspect_pilot.all_task_ids()))
    if unknown:
        print(f"no such task(s): {unknown}", file=sys.stderr)
        return 1
    tasks = [inspect_pilot._task(i) for i in ids]

    cfg = sandbox.SandboxConfig()
    if not sandbox.docker_available(cfg):
        print("sandbox image not found; build it with scripts/build_sandbox.ps1", file=sys.stderr)
        return 1

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.exists() else None
    props = warmup_summary = None
    if args.dry_run:
        model, model_kwargs = mock_model(tasks), {}
    else:
        try:
            props = get_json(args.url, "/props")
        except OSError as e:
            print(f"cannot reach {args.url}: {e}\nstart the server with scripts/serve.ps1", file=sys.stderr)
            return 1
        model = f"llamaserver/{(manifest or {}).get('name', 'model')}"
        model_kwargs = {"model_base_url": args.url}
        if not args.no_warmup:
            print("warming up with the canary ...")
            warmup_summary, _ = warmup.run_warmup(args.url)
            print(f"  canary fingerprint {warmup_summary['fingerprint']}, stable from request "
                  f"{warmup_summary['stable_from']}, ok={warmup_summary['ok']}")

    template_sha = warmup.sha256_text(args.template.read_text(encoding="utf-8")) if args.template.exists() else None
    header = pilot.build_run_header(
        tasks=tasks, conditions=args.conditions, variants=args.variants, params=json.loads(
            inspect_pilot.PILOT_CONFIG.model_dump_json(exclude_none=True)), max_attempts=args.max_attempts,
        url=args.url, props=props, manifest=manifest, template_file_sha=template_sha, sandbox_cfg=cfg,
        image_id=sandbox.image_id(cfg), git=warmup.collect_git_info(REPO), warmup=warmup_summary,
        lock_file=REPO / "uv.lock", note=args.note, context=dict(kv.split("=", 1) for kv in args.context),
        dry_run=args.dry_run)

    out = args.out or REPO / "runs" / "inspect" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    task = inspect_pilot.pilot_task(ids, tuple(args.variants), tuple(args.conditions), args.max_attempts,
                                    runner_cfg=cfg)
    print(f"{len(tasks)} task(s), variants {args.variants}, conditions {args.conditions}"
          + ("  (DRY RUN)" if args.dry_run else ""))
    # one sample at a time, so requests reach the server in a fixed order
    logs = inspect_eval(task, model=model, log_dir=str(out), max_samples=1, display="plain",
                        metadata={"run_header": header}, **model_kwargs)
    log = read_eval_log(logs[0].location)
    print()
    print(report.format_report(report.summarise(inspect_pilot.records_from_log(log))))
    print(f"\nstatus: {log.status}\nlog: {logs[0].location}")
    return 0 if log.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
