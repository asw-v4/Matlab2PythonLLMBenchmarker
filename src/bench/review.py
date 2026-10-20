"""Check the text rules by hand.

The rules that decide "asked", "stated an assumption" and "described the choice" are keyword rules, so after a
run every reply to a vague spec is read by a person. This module builds the review sheet (the reply's prose
with the code collapsed, next to what the rules said and which words matched), a blank labels file to fill in,
and the agreement between the two.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from bench import parsing

OUTCOMES = ("asked", "asked-and-proceeded", "stated-assumption", "described-choice", "silent", "no-answer")


@dataclass(frozen=True)
class ReviewSample:
    id: str
    task_id: str
    variant: str
    condition: str
    reply: str
    evaluation: dict

    @property
    def rule_outcome(self) -> str | None:
        return self.evaluation.get("outcome")


def redact_code(reply: str) -> str:
    """The reply with each code block replaced by a one-line placeholder."""
    out, pos = [], 0
    for m in parsing.FENCE.finditer(reply):
        out.append(reply[pos:m.start()])
        out.append(f"[code block, {len(m.group(2).splitlines())} lines]")
        pos = m.end()
    tail = reply[pos:]
    m = parsing.OPEN_FENCE.search(tail)
    if m:
        out.append(tail[:m.start()])
        out.append(f"[code block, never closed, {len(m.group(2).splitlines())} lines]")
    else:
        out.append(tail)
    return "".join(out).strip()


def samples_from_jsonl(path: Path) -> list[ReviewSample]:
    """Samples from a run.jsonl written by scripts/run_pilot.py."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("type") != "sample":
            continue
        reply = rec["replies"].get(rec["reply_sha256"], "")
        out.append(ReviewSample(f"{rec['task_id']}:{rec['variant']}:{rec['condition']}", rec["task_id"],
                                rec["variant"], rec["condition"], reply, rec["evaluation"]))
    return out


def samples_from_inspect_log(log) -> list[ReviewSample]:
    """Samples from an Inspect EvalLog written by scripts/run_inspect.py."""
    from bench.inspect_pilot import SCORER_NAME
    out = []
    for s in log.samples or []:
        sc = (s.scores or {}).get(SCORER_NAME)
        if sc is None or not sc.metadata:
            continue
        out.append(ReviewSample(str(s.id), s.metadata["task_id"], s.metadata["variant"], s.metadata["condition"],
                                s.output.completion, sc.metadata["evaluation"]))
    return out


def load_samples(path: Path) -> list[ReviewSample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        return samples_from_jsonl(path)
    from inspect_ai.log import read_eval_log
    if path.is_dir():
        found = sorted(path.glob("*.eval"))
        if not found:
            raise FileNotFoundError(f"no .eval log in {path}")
        path = found[-1]
    return samples_from_inspect_log(read_eval_log(str(path)))


def select(samples: list[ReviewSample], include_plain: bool = False) -> list[ReviewSample]:
    return [s for s in samples if include_plain or s.variant == "ambiguous"]


def review_sheet(samples: list[ReviewSample]) -> str:
    lines = ["# Review sheet", "",
             "Read each reply's prose and decide which outcome fits. Write it in labels.csv (column hand_outcome).", "",
             "Outcomes:",
             "- `asked`: no code, a clarifying question",
             "- `asked-and-proceeded`: a clarifying question, and code",
             "- `stated-assumption`: code, and the prose flags the vague point as a choice or assumption",
             "- `described-choice`: code, and the prose says what was done about the vague point without flagging it",
             "- `silent`: code with none of those",
             "- `no-answer`: no code and no question", ""]
    for s in samples:
        e = s.evaluation
        lines += [f"## {s.id}", "",
                  f"- rule outcome: **{s.rule_outcome}**   (choice: {e.get('choice_summary')}, per output: {e.get('choices')})",
                  f"- question sentences matched: {e.get('question_sentences') or 'none'}",
                  f"- uncertainty words matched: {e.get('assumption_cues') or 'none'}",
                  f"- vague-point words found: {e.get('mention_terms') or 'none'}",
                  f"- has code: {e.get('has_code')}, confidence: {e.get('confidence')}", "",
                  "```text", redact_code(s.reply) or "(empty reply)", "```", "",
                  "hand outcome: ______", ""]
    return "\n".join(lines)


def labels_template(samples: list[ReviewSample]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["sample_id", "rule_outcome", "hand_outcome"])
    for s in samples:
        w.writerow([s.id, s.rule_outcome, ""])
    return buf.getvalue()


def read_labels(text: str) -> dict[str, str]:
    """sample_id -> hand outcome, for rows that have one. Unknown outcomes are an error."""
    labels = {}
    for row in csv.DictReader(io.StringIO(text)):
        hand = (row.get("hand_outcome") or "").strip()
        if not hand:
            continue
        if hand not in OUTCOMES:
            raise ValueError(f"{row['sample_id']}: {hand!r} is not one of {', '.join(OUTCOMES)}")
        labels[row["sample_id"]] = hand
    return labels


def agreement(samples: list[ReviewSample], labels: dict[str, str]) -> dict:
    """How often the rules' outcome equals the hand label, over the samples that have a label."""
    labelled = [s for s in samples if s.id in labels]
    confusion = Counter((s.rule_outcome, labels[s.id]) for s in labelled)
    agree = sum(n for (rule, hand), n in confusion.items() if rule == hand)
    return {
        "n_samples": len(samples), "n_labelled": len(labelled), "n_agree": agree,
        "rate": (agree / len(labelled)) if labelled else None,
        "confusion": dict(confusion),
        "disagreements": [(s.id, s.rule_outcome, labels[s.id]) for s in labelled if s.rule_outcome != labels[s.id]],
        "unlabelled": [s.id for s in samples if s.id not in labels],
    }


def format_agreement(a: dict) -> str:
    lines = [f"labelled {a['n_labelled']} of {a['n_samples']} samples; the rules agreed on {a['n_agree']}"
             + (f" ({100 * a['rate']:.0f}%)" if a["rate"] is not None else "")]
    for sid, rule, hand in a["disagreements"]:
        lines.append(f"  differs: {sid}: rules said {rule}, by hand {hand}")
    if a["unlabelled"]:
        lines.append(f"  not yet labelled: {', '.join(a['unlabelled'])}")
    return "\n".join(lines)
