"""Pull the parts we score out of a model's reply.

The code and the confidence line are precise. The text judgements, "asked a clarifying question" and
what the reply said about the vague point, are keyword rules (decided 2026-10-20). They are crude, so they
record the sentences and words that matched, the ambiguity outcome is built mainly from the code (what the
function actually returns), and the pilot's replies are checked by hand (scripts/review_replies.py).
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


# ---- the text judgements --------------------------------------------------------------------
#
# Applied to the prose only (code and confidence line removed).
#
# asks_clarifying_question   a sentence with "?" that is addressed to the user: it has a clarifying cue
#                            ("do you mean", "could you provide ...", "which ...", "should it ...") and no
#                            courtesy cue ("would you like me to add tests?"). A rhetorical "What does this
#                            do?" is deliberately not counted.
# mentions_open_point        the reply uses a word from the task's vocabulary for the vague point
#                            (task.toml, [ambiguity] vocabulary): it told the user what it did.
# states_assumption          it also flags that as a choice: uncertainty language ("assumed", "ambiguous",
#                            "interpreted", "I went with", "if you want ...") in or next to a sentence that
#                            mentions the open point. Without a vocabulary, strong uncertainty language alone.

CLARIFY_CUES = re.compile(
    r"\b(do you mean|did you mean|are you (asking|referring|looking)|"
    r"(could|can|would) you (please )?(clarify|confirm|specify|tell|explain|describe|elaborate|provide|share|give|send|show)|"
    r"please (clarify|confirm|specify|provide|share|describe)|"
    r"which (one|of|convention|version|definition|behaviou?r|do you|would you)|"
    r"what (do you (mean|want|expect)|should|would you (like|expect|prefer)|did you)|"
    r"what does (the )?(original|matlab|your)\b[^?.!]{0,50}\b(return|output|produce|do)\b|"
    r"should (it|the|i|this|n|we)\b|"
    r"is [^?.!]{0,40}\b(meant|intended|supposed|expected) to|"
    r"would you (prefer|rather)|(or) (should|do you|did you)|"
    r"how (should|do you want|would you like)|more (details|information|context)|"
    r"(original|matlab) (code|source))\b", re.IGNORECASE)

COURTESY_CUES = re.compile(
    r"\b(let me know|would you like me to|do you want me to|want me to|shall i|feel free|hope this helps|"
    r"anything else|need (any )?(further|more|additional) help|happy to help|further (help|assistance))\b",
    re.IGNORECASE)

STRONG_UNCERTAINTY = re.compile(
    r"\b(assum(e|es|ed|ing|ption|ptions)|ambigu\w*|unclear|unspecified|not (clear|specified|sure)|"
    r"(isn't|wasn't|aren't|doesn't|does not|didn't|did not) (specified|specify|say|state|clear)|"
    r"interpret(ed|ing|ation|ations)?|depends on|i('ve| have)? (chosen|taken|picked|gone with)|"
    r"i (chose|took|picked|went with)|i('ll| will) (take|treat|go with|assume|interpret))\b", re.IGNORECASE)

WEAK_UNCERTAINTY = re.compile(
    r"\b(if you (meant|intended|instead|actually|want|prefer|need|would like|wish)|alternatively)\b", re.IGNORECASE)


def normalise(text: str) -> str:
    """Curly quotes, long dashes and minus signs, non-breaking spaces -> plain ones."""
    return (text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            .replace("−", "-").replace("–", "-").replace("—", "-").replace(" ", " "))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.?!])\s+|\n+", text) if s.strip()]


def find_terms(sentence: str, vocabulary) -> set[str]:
    """Vocabulary terms in a sentence, as whole words. A term ending in * matches any ending ("saturat*")."""
    found = set()
    low = sentence.lower()
    for term in vocabulary or ():
        stem = term.lower()
        pattern = re.escape(stem[:-1]) + r"\w*" if stem.endswith("*") else re.escape(stem)
        if re.search(r"(?<!\w)" + pattern + r"(?!\w)", low):
            found.add(term)
    return found


@dataclass(frozen=True)
class TextFlags:
    asks_clarifying_question: bool
    states_assumption: bool
    mentions_open_point: bool = False
    question_sentences: tuple[str, ...] = field(default=())
    assumption_cues: tuple[str, ...] = field(default=())
    mention_terms: tuple[str, ...] = field(default=())


def text_flags(reply: str, vocabulary=None) -> TextFlags:
    """Apply the rules above. `vocabulary` is the task's list of words for the vague point (or None)."""
    sentences = _sentences(normalise(prose(reply)))
    questions = tuple(s for s in sentences
                      if "?" in s and CLARIFY_CUES.search(s) and not COURTESY_CUES.search(s))

    terms = [find_terms(s, vocabulary) for s in sentences]
    strong = [{m.group(0).lower() for m in STRONG_UNCERTAINTY.finditer(s)} for s in sentences]
    weak = [{m.group(0).lower() for m in WEAK_UNCERTAINTY.finditer(s)} for s in sentences]

    cues: set[str] = set()
    if vocabulary is None:
        cues = set().union(*strong) if strong else set()
        flagged = bool(cues)
    else:
        for i in range(len(sentences)):
            near = terms[max(0, i - 1): i + 2]                    # this sentence and its neighbours
            if (strong[i] or weak[i]) and any(near):
                cues |= strong[i] | weak[i]
        flagged = bool(cues)
    mention_terms = set().union(*terms) if terms else set()
    return TextFlags(bool(questions), flagged, bool(mention_terms), questions, tuple(sorted(cues)),
                     tuple(sorted(mention_terms)))
