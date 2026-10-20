"""Warm up a running llama-server with the canary prompt and log everything.

    uv run python scripts/warmup.py --note "fresh start"
    uv run python scripts/warmup.py --context server_pid=1234 --context vulkan_driver=32.0.31041.1004

Log: one JSON object per line (header, one record per request, summary), written as it goes.
Exit code 0 if the replies stabilised, 2 if they did not within --n-max requests.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bench import warmup as w  # noqa: E402
from bench.llama_client import get_json, get_text  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--n-max", type=int, default=12, help="most canary requests to send")
    ap.add_argument("--stable-run", type=int, default=4, help="identical replies in a row needed at the end")
    ap.add_argument("--min-requests", type=int, default=6, help="never stop before this many requests")
    ap.add_argument("--out", type=Path, help="log path (default runs/warmup/<UTC time>.jsonl)")
    ap.add_argument("--note", help="free-text note stored in the header")
    ap.add_argument("--context", action="append", default=[], metavar="KEY=VALUE",
                    help="extra facts to store in the header; repeatable")
    ap.add_argument("--manifest", type=Path, default=REPO / "models" / "manifests" / "olmo3-7b-instruct.json")
    ap.add_argument("--template", type=Path, default=REPO / "configs" / "olmo3_chat_template_patched.jinja")
    args = ap.parse_args()

    context = dict(kv.split("=", 1) for kv in args.context)
    if args.manifest.exists():
        m = json.loads(args.manifest.read_text(encoding="utf-8"))
        context.setdefault("gguf_sha256", m["output"]["sha256"])
        context.setdefault("model_name", m["name"])
    if args.template.exists():
        context.setdefault("chat_template_file_sha256", w.sha256_text(args.template.read_text(encoding="utf-8")))

    try:
        props = get_json(args.url, "/props")
    except OSError as e:
        print(f"cannot reach {args.url}: {e}", file=sys.stderr)
        return 1

    out = args.out or REPO / "runs" / "warmup" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"n_max": args.n_max, "stable_run": args.stable_run, "min_requests": args.min_requests}
    header = w.build_header(url=args.url, props=props, git=w.collect_git_info(REPO), context=context,
                            note=args.note, settings=settings)

    with open(out, "w", encoding="utf-8", newline="\n") as fh:
        def sink(rec: dict) -> None:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if rec["type"] == "request":
                diff = rec["first_diff_vs_previous"]
                print(f"#{rec['i']:>2}  {rec['latency_s']:6.2f}s  {rec['completion_tokens']:>4} tok  "
                      f"{rec['reply_sha256'][:8]}  cache_n={rec['cache_n']}  "
                      f"diff_vs_prev={'-' if diff is None else diff}")

        sink_header = json.dumps(header) + "\n"
        fh.write(sink_header)
        fh.flush()
        summary, _ = w.run_warmup(args.url, n_max=args.n_max, stable_run=args.stable_run,
                                  min_requests=args.min_requests, sink=sink,
                                  metrics=lambda: get_text(args.url, "/metrics"))

    print(f"\nfingerprint {summary['fingerprint']}  stable from request {summary['stable_from']}  "
          f"({summary['transient_requests']} transient)  ok={summary['ok']}")
    print(f"log: {out}")
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
