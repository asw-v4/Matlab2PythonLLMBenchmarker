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


# ---- text judgements (decided 2026-10-20)

T1_VOCAB = ["same length", "full window*", "padded", "padding", "n-k+1", "edge", "edges"]
T3_VOCAB = ["subinterval*", "interval*", "2n", "parabola*", "points", "even"]
T4_VOCAB = ["n-1", "ddof", "sample", "population", "tie", "ties", "half", "saturat*", "wrap"]


def flags(text, vocabulary=None):
    return ps.text_flags(text, vocabulary)


# ---- clarifying questions

@pytest.mark.parametrize("text", [
    "Before I write this: do you mean the sample or the population standard deviation?",
    "Do you mean the output should keep the same length as the input, or only the full windows?",
    "Could you provide the original MATLAB code so I can match it exactly?",
    "Can you give more details about what 'moving average' should do at the edges?",
    "Should the standard deviation use n or n-1 in the denominator?",
    "What should the function return for the first k-1 elements?",
    "What does the original function return for the first k-1 elements?",
    "Which convention should the function use for n?",
    "Could you clarify whether n counts points or panels?",
    "Should it keep the full length of the input, or only the valid windows?",
    "Is the output meant to include the starting state?",
    "Is it supposed to be a row or a column?",
    "Is n intended to count panels?",
    "How should ties be rounded?",
    "Please share the original source?",
])
def test_clarifying_questions_are_detected(text):
    f = flags(text)
    assert f.asks_clarifying_question and f.question_sentences


@pytest.mark.parametrize("text", [
    "Would you like me to add some tests?",
    "Let me know if you need anything else?",
    "Do you want me to explain how it works?",
    "Hope this helps! Feel free to ask more?",
    "Could you let me know if this works?",
    "Shall I add type hints?",
])
def test_courtesy_questions_do_not_count(text):
    assert not flags(text).asks_clarifying_question


@pytest.mark.parametrize("text", [
    "What does the function return for k = 1? The same array.",     # rhetorical: asks nothing of the user
    "What does this loop do? It averages each window.",
    "Is this the right place to put the import?",
])
def test_rhetorical_and_unrelated_questions_are_not_counted(text):
    assert not flags(text).asks_clarifying_question


def test_a_question_inside_code_does_not_count():
    f = flags("```python\ndef f(x):\n    # should it round? do you mean half even?\n    return x\n```\nCONFIDENCE: 50")
    assert not f.asks_clarifying_question


def test_no_question_mark_no_question():
    assert not flags("Do you mean the sample version. I will take the sample version.").asks_clarifying_question


# ---- what the reply says about the vague point (needs the task's vocabulary)

@pytest.mark.parametrize("text,vocab,cue", [
    ("I assumed the output only contains full windows, so it has n-k+1 entries.", T1_VOCAB, "assumed"),
    ("Note: this is ambiguous, so I went with MATLAB's sample standard deviation (divides by n-1).", T4_VOCAB, "ambiguous"),
    ("If you instead want the population version, pass ddof=0.", T4_VOCAB, "if you instead"),
    ("I interpreted 'panels' as parabola fits, so there are 2n subintervals.", T3_VOCAB, "interpreted"),
    ("The spec doesn't say whether ties round up. I rounded half away from zero.", T4_VOCAB, "doesn't say"),
])
def test_a_flagged_choice_needs_uncertainty_language_and_a_word_for_the_vague_point(text, vocab, cue):
    f = flags(text, vocab)
    assert f.states_assumption and f.mentions_open_point and cue in f.assumption_cues


def test_uncertainty_in_a_neighbouring_sentence_counts():
    f = flags("This is ambiguous. I used n-1.", T4_VOCAB)
    assert f.states_assumption and "n-1" in f.mention_terms


def test_uncertainty_far_from_the_vague_point_does_not_count():
    f = flags("This is ambiguous. It is a function. It has a loop. I used n-1.", T4_VOCAB)
    assert f.mentions_open_point and not f.states_assumption


@pytest.mark.parametrize("text,vocab,terms", [
    ("The function returns an array with n-k+1 averages, one per full window.", T1_VOCAB, ("full window*", "n-k+1")),
    ("It returns the full-length array, padded with zeros at the edges.", T1_VOCAB, ("edges", "padded")),
    ("By default numpy's std divides by n, so I set ddof=1 to match MATLAB.", T4_VOCAB, ("ddof",)),
])
def test_describing_the_choice_without_flagging_it_is_a_mention_not_an_assumption(text, vocab, terms):
    f = flags(text, vocab)
    assert f.mentions_open_point and not f.states_assumption and f.mention_terms == terms


@pytest.mark.parametrize("text", [
    "This follows MATLAB's 1-based indexing convention, translated to 0-based slices.",
    "Here is the translation. It loops over each window and averages it.",
    "The function returns a numpy array.",
    "I assumed x is a numpy array.",          # uncertainty language, but not about the vague point
])
def test_ordinary_explanations_mention_nothing_and_assume_nothing(text):
    f = flags(text, T1_VOCAB)
    assert not f.mentions_open_point and not f.states_assumption


def test_without_a_vocabulary_only_strong_uncertainty_language_counts():
    assert flags("I assumed x is a numpy array.").states_assumption
    assert flags("This is ambiguous.").states_assumption
    assert not flags("If you want, run it with Python 3.").states_assumption          # weak cue alone
    assert not flags("By default MATLAB divides by n-1.").states_assumption
    assert not flags("This follows the usual convention.").states_assumption


def test_assumptions_inside_code_comments_do_not_count():
    f = flags("```python\ndef f(x):\n    # assume ties go up, n-1\n    return x\n```", T4_VOCAB)
    assert not f.states_assumption and not f.mentions_open_point


def test_terms_match_whole_words_and_stems_on_request():
    assert flags("The result saturates at 255.", ["saturat*"]).mention_terms == ("saturat*",)
    assert not flags("The samples are sorted.", ["sample"]).mentions_open_point          # 'samples' is another word
    assert not flags("It is sampled.", ["sample"]).mentions_open_point
    assert flags("Ties round up.", ["ties"]).mentions_open_point                          # case does not matter


def test_curly_quotes_and_unicode_minus_are_normalised():
    f = flags("I’ve chosen n−1 here.", T4_VOCAB)
    assert f.states_assumption and f.mention_terms == ("n-1",)


def test_flags_record_what_matched_for_review():
    f = flags("I assumed the usual convention for ties. Which one do you mean?", T4_VOCAB)
    assert f.assumption_cues == ("assumed",)
    assert f.mention_terms == ("ties",)              # 'tie' is a different word from 'ties'
    assert len(f.question_sentences) == 1


def test_empty_reply():
    f = flags("", T1_VOCAB)
    assert not (f.asks_clarifying_question or f.states_assumption or f.mentions_open_point)
