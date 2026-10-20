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


def comments_in(reply: str) -> str:
    """The comments and docstrings of every Python block in the reply (what the code says about itself)."""
    blocks, _ = parsing._blocks(reply)
    notes = [parsing.code_notes(body) for lang, body, _ in blocks if lang in parsing.PYTHON_LANGS]
    return "\n".join(n for n in notes if n)


def yes_no(flag) -> str:
    return "yes" if flag else "no"


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


def rescore(sample: ReviewSample) -> ReviewSample:
    """Recompute the text judgements from the stored reply with the CURRENT rules and vocabulary.

    The code's results (pass/fail, signature, which reading it chose) are left as they were; only what depends on the
    words does: asked, flagged, described, documented in code, and the outcome built from them. Lets a log from before
    a rule change be reviewed without running the model again.
    """
    from bench import ambiguity
    from bench.inspect_pilot import _task
    task = _task(sample.task_id)
    vocab = task.ambiguity_vocabulary
    e = dict(sample.evaluation)
    flags = parsing.text_flags(sample.reply, vocab)
    asks = flags.asks_clarifying_question or e.get("control") == "question"
    code = parsing.extract_code(sample.reply, task.function).code
    in_code = parsing.code_flags(code, vocab) if code is not None else None
    e.update(asks_clarifying_question=asks, states_assumption=flags.states_assumption,
             mentions_open_point=flags.mentions_open_point, question_sentences=list(flags.question_sentences),
             assumption_cues=list(flags.assumption_cues), mention_terms=list(flags.mention_terms),
             documented_in_code=bool(in_code and in_code.mentions_open_point),
             flagged_in_code=bool(in_code and in_code.states_assumption),
             code_mention_terms=list(in_code.mention_terms) if in_code else [],
             code_assumption_cues=list(in_code.assumption_cues) if in_code else [])
    if sample.variant == "ambiguous":
        e["outcome"] = ambiguity.outcome(has_code=code is not None, asks_clarifying_question=asks,
                                         states_assumption=flags.states_assumption,
                                         mentions_open_point=flags.mentions_open_point)
    return ReviewSample(sample.id, sample.task_id, sample.variant, sample.condition, sample.reply, e)


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
             "- `no-answer`: no code and no question", "",
             "**`hand_outcome` is about the PROSE only: the text outside the code block.** Anything inside the code, its",
             "comments and docstrings, never counts for it. So a reply with code and no text around it is `silent`, however",
             "much its comments say. What the comments say goes in the other column: `hand_documented` (yes or no: do the",
             "code's own comments or docstrings mention the vague point).", ""]
    for s in samples:
        e = s.evaluation
        notes = comments_in(s.reply)
        outside = parsing.prose(s.reply).strip()
        lines += [f"## {s.id}", "",
                  "- text outside the code block: " + ("**none**, so `silent` if there is code" if not outside
                                                       else f"**yes**, {len(outside)} characters (shown below)"),
                  f"- rule outcome: **{s.rule_outcome}**   (choice: {e.get('choice_summary')}, per output: {e.get('choices')})",
                  f"- question sentences matched: {e.get('question_sentences') or 'none'}",
                  f"- uncertainty words matched: {e.get('assumption_cues') or 'none'}",
                  f"- vague-point words found in the prose: {e.get('mention_terms') or 'none'}",
                  f"- documented in code: **{yes_no(e.get('documented_in_code'))}** "
                  f"(words: {e.get('code_mention_terms') or 'none'}; flagged as a choice: {yes_no(e.get('flagged_in_code'))})",
                  f"- has code: {e.get('has_code')}, confidence: {e.get('confidence')}", "",
                  "Prose (code collapsed):", "", "```text", redact_code(s.reply) or "(empty reply)", "```", "",
                  "Comments and docstrings in the code:", "", "```text", notes or "(none)", "```", "",
                  "hand outcome: ______      hand documented in code (yes/no): ______", ""]
    return "\n".join(lines)


LABEL_COLUMNS = ["sample_id", "rule_outcome", "hand_outcome", "rule_documented", "hand_documented"]


def labels_template(samples: list[ReviewSample]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(LABEL_COLUMNS)
    for s in samples:
        w.writerow([s.id, s.rule_outcome, "", yes_no(s.evaluation.get("documented_in_code")), ""])
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


def read_documented_labels(text: str) -> dict[str, bool]:
    """sample_id -> whether the code documents the vague point, by hand, for rows that have yes or no."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        hand = (row.get("hand_documented") or "").strip().lower()
        if not hand:
            continue
        if hand not in ("yes", "no"):
            raise ValueError(f"{row['sample_id']}: hand_documented must be yes or no, not {hand!r}")
        out[row["sample_id"]] = hand == "yes"
    return out


def agreement_documented(samples: list[ReviewSample], labels: dict[str, bool]) -> dict:
    labelled = [s for s in samples if s.id in labels]
    wrong = [(s.id, bool(s.evaluation.get("documented_in_code")), labels[s.id]) for s in labelled
             if bool(s.evaluation.get("documented_in_code")) != labels[s.id]]
    return {"n_labelled": len(labelled), "n_agree": len(labelled) - len(wrong),
            "rate": ((len(labelled) - len(wrong)) / len(labelled)) if labelled else None, "disagreements": wrong}


def format_documented_agreement(a: dict) -> str:
    text = f"documented-in-code flag: labelled {a['n_labelled']}, the rule agreed on {a['n_agree']}"
    if a["rate"] is not None:
        text += f" ({100 * a['rate']:.0f}%)"
    for sid, rule, hand in a["disagreements"]:
        text += f"\n  differs: {sid}: rule said {yes_no(rule)}, by hand {yes_no(hand)}"
    return text


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
