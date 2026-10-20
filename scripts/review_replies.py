"""Check the text rules by hand after a run.

    uv run python scripts/review_replies.py runs/inspect/<time>            # writes review.md and labels.csv
    uv run python scripts/review_replies.py runs/pilot/<time>/run.jsonl    # the non-Inspect log works too
    (edit labels.csv: fill in hand_outcome for each row, using the outcomes listed in review.md, and
     hand_documented with yes or no for whether the code's own comments or docstrings mention the vague point)
    uv run python scripts/review_replies.py runs/inspect/<time> --labels runs/inspect/<time>/labels.csv

Only replies to the vague specs are reviewed (add --all for the plain ones too).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from bench import review  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", type=Path, help="an Inspect log (.eval file or its folder) or a run.jsonl")
    ap.add_argument("--labels", type=Path, help="a filled-in labels.csv: report agreement instead of writing the sheet")
    ap.add_argument("--all", action="store_true", help="include replies to the plain specs")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the text judgements from the stored replies with the current rules "
                         "(for a log written before a rule change; the code's results are kept)")
    ap.add_argument("--out", type=Path, help="folder for review.md and labels.csv (default: next to the log)")
    args = ap.parse_args()

    samples = review.select(review.load_samples(args.log), include_plain=args.all)
    if args.rescore:
        samples = [review.rescore(s) for s in samples]
    if not samples:
        print("no samples to review in that log", file=sys.stderr)
        return 1

    if args.labels:
        text = args.labels.read_text(encoding="utf-8-sig")        # -sig: editors and Excel may add a byte-order mark
        labels = review.read_labels(text)
        unknown = sorted(set(labels) - {s.id for s in samples})
        if unknown:
            print(f"labels for samples not in this log: {unknown}", file=sys.stderr)
            return 1
        print(review.format_agreement(review.agreement(samples, labels)))
        documented = review.read_documented_labels(text)
        if documented:
            print(review.format_documented_agreement(review.agreement_documented(samples, documented)))
        return 0

    out = args.out or (args.log if args.log.is_dir() else args.log.parent)
    out.mkdir(parents=True, exist_ok=True)
    (out / "review.md").write_text(review.review_sheet(samples), encoding="utf-8", newline="\n")
    labels = out / "labels.csv"
    if labels.exists():
        print(f"{len(samples)} replies to review\n  read:  {out / 'review.md'}\n"
              f"  kept your existing {labels.name} (it is never overwritten; delete it first to start again)")
        return 0
    labels.write_text(review.labels_template(samples), encoding="utf-8", newline="\n")
    print(f"{len(samples)} replies to review\n  read:  {out / 'review.md'}\n  fill:  {labels}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
