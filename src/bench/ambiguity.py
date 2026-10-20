"""What did the model do with a deliberately vague spec?

Built mainly from the artefact, the code's actual output, not from the wording of the reply. For each
output that has alternative readings, the function's results on the cases where the readings can be told
apart are compared with reading A (the key) and with each alternative. The reply's text only adds
whether the model asked a clarifying question or stated an assumption (draft rules in `parsing`).
"""
from __future__ import annotations

import numpy as np

from bench.runs import OK, CaseRun
from bench.scoring import compare

A = "A"
NOT_INFORMATIVE = "not-informative"


def _tol(task, name: str) -> dict:
    if name in task.exact_outputs:
        return dict(rtol=0.0, atol=0.0)
    return {"rtol": task.tolerance["rtol"], "atol": task.tolerance["atol"], **task.tolerance_overrides.get(name, {})}


def _same(actual, expected, task, name) -> bool:
    return compare(actual, expected, vector_ok=task.vector_ok, **_tol(task, name)).passed


def readings_for(task, name: str) -> list[str]:
    return sorted({r for c in task.cases for r, outs in c.readings.items() if name in outs})


def informative_cases(task, name: str, reading: str) -> list:
    """Cases where `reading` gives a different answer for `name` than the key does."""
    return [c for c in task.cases
            if name in c.readings.get(reading, {}) and not _same(c.readings[reading][name], c.expected[name], task, name)]


def output_choices(runs: list[CaseRun], task) -> dict[str, str]:
    """For each output that has alternative readings: "A", the name of the reading it matched, "other",
    "error" (the code did not run, or returned the wrong number of values), or "not-informative"."""
    names = task.output_names
    choices: dict[str, str] = {}
    for n_idx, name in enumerate(names):
        readings = readings_for(task, name)
        if not readings:
            continue
        info = {r: informative_cases(task, name, r) for r in readings}
        union = {c.index: c for r in readings for c in info[r]}
        if not union:
            choices[name] = NOT_INFORMATIVE
            continue

        def actual_for(case):
            run = runs[case.index - 1]
            if run.status != OK or not run.outputs or len(run.outputs) != len(names):
                return None
            return run.outputs[n_idx]

        if any(actual_for(c) is None for c in union.values()):
            choices[name] = "error"
        elif all(_same(actual_for(c), c.expected[name], task, name) for c in union.values()):
            choices[name] = A
        else:
            match = [r for r in readings if info[r]
                     and all(_same(actual_for(c), c.readings[r][name], task, name) for c in info[r])]
            choices[name] = match[0] if match else "other"
    return choices


def outcome(*, has_code: bool, asks_clarifying_question: bool, states_assumption: bool) -> str:
    """The reply's overall behaviour on a vague spec.

    asked                 no code, a clarifying question
    no-answer             no code, no question
    asked-and-proceeded   a clarifying question and code
    stated-assumption     code, and the prose states an assumption or choice
    silent                code with neither: any choice was made without saying so
    """
    if not has_code:
        return "asked" if asks_clarifying_question else "no-answer"
    if asks_clarifying_question:
        return "asked-and-proceeded"
    if states_assumption:
        return "stated-assumption"
    return "silent"


def summary_label(choices: dict[str, str]) -> str:
    """One word for a whole sample: the single choice if every ambiguous output agrees, else "mixed"."""
    picked = {v for v in choices.values() if v != NOT_INFORMATIVE}
    if not picked:
        return NOT_INFORMATIVE
    return picked.pop() if len(picked) == 1 else "mixed"
