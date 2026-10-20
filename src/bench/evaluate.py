"""Everything we conclude from one reply: parse it, run its code, score it, name its failure, and for a
vague spec say what the model did about the vagueness."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Callable

from bench import ambiguity as amb
from bench import parsing, sandbox
from bench.signatures import classify_sample

PLAIN, AMBIGUOUS = "plain", "ambiguous"
NO_CODE = "no-code"


@dataclass
class Evaluation:
    task_id: str
    variant: str                       # "plain" or "ambiguous"
    condition: str                     # "free" or "format"
    # what the reply contained
    has_code: bool
    code_reason: str                   # "ok", "no-code-block" or "no-function"
    code_truncated: bool
    confidence: float | None
    confidence_note: str               # "ok", "missing" or "out-of-range"
    control: str | None                # format condition only: "question", "code" or "neither"
    asks_clarifying_question: bool
    states_assumption: bool
    mentions_open_point: bool
    question_sentences: list[str] = field(default_factory=list)
    assumption_cues: list[str] = field(default_factory=list)
    mention_terms: list[str] = field(default_factory=list)
    # what the code's own comments and docstrings say about the vague point (kept apart from the prose judgement)
    documented_in_code: bool = False
    flagged_in_code: bool = False
    code_mention_terms: list[str] = field(default_factory=list)
    code_assumption_cues: list[str] = field(default_factory=list)
    # what the code did
    case_status: list[str] = field(default_factory=list)
    case_errors: list[str | None] = field(default_factory=list)
    passed: bool = False
    signature: str = NO_CODE
    case_signatures: list[str] = field(default_factory=list)
    # for a vague spec
    choices: dict[str, str] | None = None
    choice_summary: str | None = None
    outcome: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def evaluate_reply(reply: str, task, *, variant: str, condition: str,
                   runner: Callable = sandbox.run_code, runner_cfg=None) -> Evaluation:
    if variant not in (PLAIN, AMBIGUOUS):
        raise ValueError(f"variant must be {PLAIN!r} or {AMBIGUOUS!r}")
    extracted = parsing.extract_code(reply, task.function)
    confidence, note = parsing.parse_confidence(reply)
    flags = parsing.text_flags(reply, vocabulary=task.ambiguity_vocabulary)
    control = parsing.parse_control(reply) if condition == "format" else None
    asks = flags.asks_clarifying_question or control == "question"

    ev = Evaluation(
        task_id=task.id, variant=variant, condition=condition,
        has_code=extracted.code is not None, code_reason=extracted.reason, code_truncated=extracted.truncated,
        confidence=confidence, confidence_note=note, control=control,
        asks_clarifying_question=asks, states_assumption=flags.states_assumption,
        mentions_open_point=flags.mentions_open_point,
        question_sentences=list(flags.question_sentences), assumption_cues=list(flags.assumption_cues),
        mention_terms=list(flags.mention_terms),
    )

    if extracted.code is not None:
        in_code = parsing.code_flags(extracted.code, vocabulary=task.ambiguity_vocabulary)
        ev.documented_in_code, ev.flagged_in_code = in_code.mentions_open_point, in_code.states_assumption
        ev.code_mention_terms, ev.code_assumption_cues = list(in_code.mention_terms), list(in_code.assumption_cues)

    runs = None
    if extracted.code is not None:
        runs = runner(extracted.code, task, runner_cfg) if runner_cfg is not None else runner(extracted.code, task)
        sample = classify_sample(runs, task)
        ev.case_status = [r.status for r in runs]
        ev.case_errors = [r.error for r in runs]
        ev.passed, ev.signature = sample.passed, sample.signature
        ev.case_signatures = [c.signature for c in sample.cases]

    if variant == AMBIGUOUS:
        ev.choices = amb.output_choices(runs, task) if runs is not None else {}
        ev.choice_summary = amb.summary_label(ev.choices) if runs is not None else None
        ev.outcome = amb.outcome(has_code=ev.has_code, asks_clarifying_question=asks,
                                 states_assumption=flags.states_assumption,
                                 mentions_open_point=flags.mentions_open_point)
    return ev
