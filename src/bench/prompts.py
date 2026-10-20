"""The prompts sent to the model.

Every sample is its own fresh conversation: one system message and one user message holding the
spec, never anything from another sample. The system message is always present, so the model's
behaviour doesn't depend on a template default (see the README).

The system prompts were reviewed on 2026-10-20: a neutral one-line role (a "careful programmer" persona
could push the model toward asking), and a confidence question that doesn't say "on every test input"
(which would hint that inputs vary, the very thing some tasks test). Their SHA-256 goes into every run
log, so results are tied to the exact text.
"""
from __future__ import annotations

import hashlib

# Primary condition: free text. It asks for code and a confidence line, and says nothing about
# questions, so asking one is the model's own choice.
SYSTEM_FREE = (
    "You translate MATLAB code to Python. "
    "Put your Python function in a single ```python code block. "
    "Finish with a last line of the form `CONFIDENCE: <number from 0 to 100>`, your percent confidence "
    "that your function is correct."
)

# Control condition: the format makes asking a question an explicit option.
SYSTEM_FORMAT = (
    "You translate MATLAB code to Python. "
    "Begin your reply with exactly one of these two markers. "
    "`QUESTION:` followed by one question for the user, and nothing else, if you cannot write the function "
    "without an answer. "
    "`CODE:` followed by your Python function in a single ```python code block, and then a last line of the form "
    "`CONFIDENCE: <number from 0 to 100>`, your percent confidence that your function is correct."
)

CONDITIONS = {"free": SYSTEM_FREE, "format": SYSTEM_FORMAT}


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def system_prompt(condition: str) -> str:
    try:
        return CONDITIONS[condition]
    except KeyError:
        raise ValueError(f"unknown condition {condition!r}; choose from {sorted(CONDITIONS)}") from None


def build_messages(condition: str, spec: str) -> list[dict]:
    """The whole conversation for one sample: a system message and the spec. Nothing else."""
    if not spec or not spec.strip():
        raise ValueError("empty spec")
    return [{"role": "system", "content": system_prompt(condition)},
            {"role": "user", "content": spec.strip() + "\n"}]
