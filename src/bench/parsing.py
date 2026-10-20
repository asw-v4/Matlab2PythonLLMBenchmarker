"""Pull the parts we score out of a model's reply.

The code and the confidence line are precise. The two text judgements, "asked a clarifying question"
and "stated an assumption", are DRAFT keyword rules: they are deliberately conservative, they record
which words matched so a person can review them, and the ambiguity outcome is built mainly from the
code (what the function actually returns), not from these.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*?)```", re.DOTALL)
OPEN_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\r?\n(.*)\Z", re.DOTALL)
PYTHON_LANGS = {"", "python", "py", "python3"}
CONFIDENCE = re.compile(r"CONFIDENCE\s*[:=]\s*\**\s*(\d+(?:\.\d+)?)\s*%?", re.IGNORECASE)


@dataclass(frozen=True)
class Extracted:
    code: str | None
    n_blocks: int            # fenced blocks found
    truncated: bool          # the chosen block was never closed (reply cut off)
    reason: str              # "ok", "no-code-block" or "no-function"


def _blocks(text: str) -> tuple[list[tuple[str, str, bool]], str]:
    """(language, body, truncated) for each fenced block, and the text with the blocks removed."""
    found = [(m.group(1).lower(), m.group(2), False) for m in FENCE.finditer(text)]
    rest = FENCE.sub("\n", text)
    m = OPEN_FENCE.search(rest)
    if m:
        found.append((m.group(1).lower(), m.group(2), True))
        rest = rest[:m.start()]
    return found, rest


def extract_code(text: str, function_name: str) -> Extracted:
    """The first Python block that defines `function_name`. Other blocks (usage examples) are ignored."""
    blocks, _ = _blocks(text)
    if not blocks:
        return Extracted(None, 0, False, "no-code-block")
    defines = re.compile(rf"^[ \t]*def[ \t]+{re.escape(function_name)}[ \t]*\(", re.MULTILINE)
    for lang, body, truncated in blocks:
        if lang in PYTHON_LANGS and defines.search(body):
            return Extracted(body, len(blocks), truncated, "ok")
    return Extracted(None, len(blocks), False, "no-function")


def parse_confidence(text: str) -> tuple[float | None, str]:
    """The last `CONFIDENCE: n` in the reply. Returns (value or None, note)."""
    matches = CONFIDENCE.findall(text)
    if not matches:
        return None, "missing"
    value = float(matches[-1])
    if not 0 <= value <= 100:
        return None, "out-of-range"
    return value, "ok"


def prose(text: str) -> str:
    """The reply without code blocks and without the confidence line."""
    _, rest = _blocks(text)
    return CONFIDENCE.sub("", rest).strip()


def parse_control(text: str) -> str:
    """Format-constrained condition: did the reply start with QUESTION: or CODE: ?"""
    t = text.lstrip()
    if re.match(r"question\s*:", t, re.IGNORECASE):
        return "question"
    if re.match(r"code\s*:", t, re.IGNORECASE):
        return "code"
    return "neither"


# ---- draft text judgements ------------------------------------------------------------------

CLARIFY_CUES = re.compile(
    r"\b(do you mean|did you mean|could you (please )?(clarify|confirm|specify|tell)|"
    r"can you (please )?(clarify|confirm|specify|tell)|please (clarify|confirm|specify)|"
    r"which (one|of|convention|version|definition|do you|would you)|what (do you mean|should|would you)|"
    r"should (it|the|i|this|n)\b|are you (asking|referring|looking)|"
    r"is [^?.!]{0,40}\b(meant|intended|supposed) to|would you (prefer|rather)|"
    r"(or) (should|do you|did you))\b", re.IGNORECASE)

COURTESY_CUES = re.compile(
    r"\b(let me know|would you like me to|want me to|feel free|hope this helps|anything else|"
    r"need (any )?(further|more|additional) help|happy to help|further (help|assistance))\b", re.IGNORECASE)

ASSUMPTION_CUES = re.compile(
    r"\b(assum(e|es|ed|ing|ption|ptions)|ambigu\w*|interpret(ed|ing|ation|ations)?|conventions?|"
    r"by default|i('ve| have)? (chosen|taken|picked|gone with)|i (chose|took|picked|went with)|"
    r"i('ll| will) (take|treat|go with|assume|interpret)|if you (meant|intended|instead|actually))\b",
    re.IGNORECASE)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


@dataclass(frozen=True)
class TextFlags:
    asks_clarifying_question: bool
    states_assumption: bool
    question_sentences: tuple[str, ...] = field(default=())
    assumption_cues: tuple[str, ...] = field(default=())


def text_flags(reply: str) -> TextFlags:
    """Draft rules, applied to the prose only (code and confidence line removed).

    A question counts as clarifying if it contains a clarifying cue ("do you mean", "which ...",
    "should it ...") and no courtesy cue ("would you like me to add tests?"). An assumption is stated
    if the prose uses words such as assume, ambiguous, interpret, convention or "I went with".
    """
    p = prose(reply)
    questions = tuple(s for s in _sentences(p)
                      if "?" in s and CLARIFY_CUES.search(s) and not COURTESY_CUES.search(s))
    cues = tuple(sorted({m.group(0).lower() for m in ASSUMPTION_CUES.finditer(p)}))
    return TextFlags(bool(questions), bool(cues), questions, cues)
