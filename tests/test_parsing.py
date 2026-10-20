import pytest

from bench import parsing as ps

GOOD = """Here is the translation.

```python
import numpy as np

def movavg(x, k):
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])
```

The window includes both ends in MATLAB, so the slice stops at i + k.

CONFIDENCE: 85
"""


# ---- code extraction

def test_extracts_the_function_block():
    e = ps.extract_code(GOOD, "movavg")
    assert e.reason == "ok" and e.n_blocks == 1 and not e.truncated
    assert "def movavg(x, k):" in e.code and e.code.strip().endswith("range(n - k + 1)]])")
    assert "```" not in e.code


def test_no_code_block():
    e = ps.extract_code("I can't tell what you mean.", "movavg")
    assert e.code is None and e.reason == "no-code-block" and e.n_blocks == 0


def test_a_block_that_does_not_define_the_function_is_not_code():
    e = ps.extract_code("```python\nprint('hi')\n```", "movavg")
    assert e.code is None and e.reason == "no-function" and e.n_blocks == 1


def test_untagged_fence_counts_as_python():
    assert ps.extract_code("```\ndef f(x):\n    return x\n```", "f").code is not None


def test_other_languages_are_ignored():
    text = "```matlab\nfunction y = f(x)\n  y = x;\nend\n```\n```python\ndef f(x):\n    return x\n```"
    e = ps.extract_code(text, "f")
    assert e.code.startswith("def f") and e.n_blocks == 2


def test_picks_the_block_that_defines_the_function_not_the_usage_example():
    text = ("```python\nx = movavg(a, 3)\nprint(x)\n```\nand\n"
            "```python\ndef movavg(x, k):\n    return x\n```\n")
    assert "def movavg" in ps.extract_code(text, "movavg").code


def test_first_defining_block_wins():
    text = "```python\ndef f(x):\n    return 1\n```\n```python\ndef f(x):\n    return 2\n```"
    assert "return 1" in ps.extract_code(text, "f").code


def test_function_name_must_match_exactly():
    assert ps.extract_code("```python\ndef movavg2(x):\n    pass\n```", "movavg").code is None
    assert ps.extract_code("```python\ndef movavg (x):\n    pass\n```", "movavg").code is not None


def test_truncated_reply_with_unclosed_fence_is_flagged():
    e = ps.extract_code("```python\ndef f(x):\n    return x +", "f")
    assert e.code is not None and e.truncated and e.n_blocks == 1


def test_windows_line_endings():
    assert ps.extract_code("```python\r\ndef f(x):\r\n    return x\r\n```", "f").code is not None


# ---- confidence

@pytest.mark.parametrize("text,expected", [
    ("CONFIDENCE: 85", 85.0), ("confidence: 7", 7.0), ("CONFIDENCE:100", 100.0), ("CONFIDENCE: 0", 0.0),
    ("CONFIDENCE: 72.5", 72.5), ("CONFIDENCE: 90%", 90.0), ("**CONFIDENCE: 60**", 60.0),
    ("CONFIDENCE = 40", 40.0),
])
def test_confidence_forms(text, expected):
    assert ps.parse_confidence(text) == (expected, "ok")


def test_confidence_takes_the_last_one():
    assert ps.parse_confidence("CONFIDENCE: 10\nmore\nCONFIDENCE: 90") == (90.0, "ok")


@pytest.mark.parametrize("text,note", [
    ("no confidence here", "missing"), ("CONFIDENCE: high", "missing"), ("CONFIDENCE: n/a", "missing"),
    ("CONFIDENCE: 150", "out-of-range"),
])
def test_confidence_missing_or_invalid(text, note):
    assert ps.parse_confidence(text) == (None, note)


# ---- prose and control format

def test_prose_removes_code_and_confidence_line():
    p = ps.prose(GOOD)
    assert "def movavg" not in p and "CONFIDENCE" not in p and "Here is the translation." in p


def test_prose_drops_a_truncated_block_too():
    assert "return x +" not in ps.prose("Before.\n```python\ndef f(x):\n    return x +")


@pytest.mark.parametrize("text,kind", [
    ("QUESTION: Do you want n-1?", "question"), ("  question: which one?", "question"),
    ("CODE:\n```python\n```", "code"), ("code: x", "code"),
    ("Here you go", "neither"), ("", "neither"), ("I have a QUESTION: no", "neither"),
])
def test_control_format(text, kind):
    assert ps.parse_control(text) == kind


# ---- text judgements (draft rules)

def flags(text):
    return ps.text_flags(text)


def test_a_real_clarifying_question_is_detected():
    f = flags("Before I write this: do you mean the sample or the population standard deviation?")
    assert f.asks_clarifying_question and f.question_sentences


@pytest.mark.parametrize("text", [
    "Which convention should the function use for n?",
    "Could you clarify whether n counts points or panels?",
    "Should it keep the full length of the input, or only the valid windows?",
    "Is the output meant to include the starting state?",
])
def test_other_clarifying_questions(text):
    assert flags(text).asks_clarifying_question


@pytest.mark.parametrize("text", [
    "Would you like me to add some tests?",
    "Let me know if you need anything else?",
    "Do you want me to explain how it works?",
    "Hope this helps! Feel free to ask more?",
])
def test_courtesy_questions_do_not_count(text):
    assert not flags(text).asks_clarifying_question


def test_a_question_inside_code_does_not_count():
    f = flags("```python\ndef f(x):\n    # should it round? do you mean half even?\n    return x\n```\nCONFIDENCE: 50")
    assert not f.asks_clarifying_question


def test_no_question_mark_no_question():
    assert not flags("Do you mean the sample version. I will take the sample version.").asks_clarifying_question


@pytest.mark.parametrize("text,cue", [
    ("I assumed n is the number of subintervals.", "assumed"),
    ("This is ambiguous, so I went with the MATLAB default.", "ambiguous"),
    ("I interpreted 'panels' as parabola fits.", "interpreted"),
    ("I've chosen the sample standard deviation.", "i've chosen"),
    ("By default MATLAB divides by n-1.", "by default"),
    ("If you meant the population version, change ddof.", "if you meant"),
])
def test_stated_assumptions(text, cue):
    f = flags(text)
    assert f.states_assumption and cue in f.assumption_cues


@pytest.mark.parametrize("text", [
    "The function returns a numpy array.", "It loops over the windows and averages each one.",
    "Note that the result is a row vector.", "Here is the code you asked for.",
])
def test_ordinary_explanations_are_not_assumptions(text):
    assert not flags(text).states_assumption


def test_assumptions_inside_code_comments_do_not_count():
    f = flags("```python\ndef f(x):\n    # assume x is sorted\n    return x\n```")
    assert not f.states_assumption


def test_flags_record_what_matched_for_review():
    f = flags("I assumed the usual convention. Which one do you mean?")
    assert f.assumption_cues == ("assumed", "convention") and len(f.question_sentences) == 1


def test_empty_reply():
    f = flags("")
    assert not f.asks_clarifying_question and not f.states_assumption


@pytest.mark.parametrize("text", [
    "Is the output meant to include the starting state?",
    "Is it supposed to be a row or a column?",
    "Is n intended to count panels?",
])
def test_meant_to_questions_are_clarifying(text):
    assert flags(text).asks_clarifying_question


def test_a_plain_question_about_something_else_is_not_flagged():
    assert not flags("Is this the right place to put the import?").asks_clarifying_question
