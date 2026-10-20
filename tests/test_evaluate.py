import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import reply_with, run_local  # noqa: E402

from bench.evaluate import AMBIGUOUS, NO_CODE, PLAIN, evaluate_reply  # noqa: E402
from bench.taskio import load_task  # noqa: E402

TASKS_DIR = Path(__file__).parent.parent / "tasks"
T1 = load_task(TASKS_DIR / "t01_movavg")
T4 = load_task(TASKS_DIR / "t04_robust_stats")

GOLDEN_T1 = (TASKS_DIR / "t01_movavg" / "solution.py").read_text()

DROP_LAST = """import numpy as np
def movavg(x, k):
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k)]])
"""
SAME_LENGTH = """import numpy as np
def movavg(x, k):
    return np.convolve(x.ravel(), np.ones(k) / k, mode='same').reshape(1, -1)
"""
RAISES = "def movavg(x, k):\n    raise ValueError('bad')\n"


def ev(reply, task=T1, variant=PLAIN, condition="free"):
    return evaluate_reply(reply, task, variant=variant, condition=condition, runner=run_local)


# ---- plain spec

def test_a_correct_reply_passes_with_its_confidence():
    e = ev(reply_with(GOLDEN_T1))
    assert e.passed and e.signature == "pass" and e.has_code and e.code_reason == "ok"
    assert e.confidence == 90 and e.confidence_note == "ok"
    assert e.case_status == ["ok"] * 5 and e.case_signatures == ["pass"] * 5
    assert e.choices is None and e.outcome is None                 # not an ambiguity sample


def test_a_known_wrong_translation_is_named_by_its_signature():
    e = ev(reply_with(DROP_LAST, confidence="CONFIDENCE: 95"))
    assert not e.passed and e.signature == "shifted-output" and e.confidence == 95


def test_code_that_raises_is_a_runtime_error():
    e = ev(reply_with(RAISES))
    assert not e.passed and e.signature == "runtime-error" and "ValueError: bad" in e.case_errors[0]


def test_a_reply_without_code_fails_as_no_code():
    e = ev("I'm not sure how to do that.\nCONFIDENCE: 10")
    assert not e.passed and e.signature == NO_CODE and not e.has_code and e.case_status == []
    assert e.code_reason == "no-code-block" and e.confidence == 10


def test_code_that_does_not_define_the_function_fails_as_no_code():
    e = ev(reply_with("def other(x):\n    return x\n"))
    assert e.signature == NO_CODE and e.code_reason == "no-function"


def test_a_missing_confidence_is_recorded_not_guessed():
    e = ev(reply_with(GOLDEN_T1, confidence=""))
    assert e.passed and e.confidence is None and e.confidence_note == "missing"


def test_a_cut_off_reply_is_flagged_and_fails():
    e = ev("```python\ndef movavg(x, k):\n    return x[:, :")
    assert e.code_truncated and not e.passed and e.signature == "runtime-error"


def test_text_flags_are_recorded_for_review_even_on_a_plain_spec():
    e = ev(reply_with(GOLDEN_T1, prose_after="I assumed the output only has full windows."))
    assert e.states_assumption and e.assumption_cues == ["assumed"] and e.outcome is None
    assert e.mentions_open_point and e.mention_terms == ["full window*"]


def test_wrong_variant_name_is_rejected():
    with pytest.raises(ValueError):
        ev("x", variant="vague")


def test_the_dict_form_is_json_ready():
    import json
    json.dumps(ev(reply_with(GOLDEN_T1)).as_dict())


# ---- vague spec

def test_silently_taking_the_matlab_reading_is_silent_a():
    e = ev(reply_with(GOLDEN_T1), variant=AMBIGUOUS)
    assert e.outcome == "silent" and e.choices == {"result": "A"} and e.choice_summary == "A"


def test_silently_taking_the_numpy_same_length_reading():
    e = ev(reply_with(SAME_LENGTH), variant=AMBIGUOUS)
    assert e.outcome == "silent" and e.choices == {"result": "same_numpy"} and e.choice_summary == "same_numpy"
    assert not e.passed                                             # matches reading B, not the key


def test_the_matlab_alignment_of_the_same_length_reading_is_told_apart_on_even_windows():
    matlab_aligned = ("import numpy as np\n"
                      "def movavg(x, k):\n"
                      "    full = np.convolve(x.ravel(), np.ones(k) / k)\n"
                      "    s = -(-(k - 1) // 2)\n"                       # ceil((k-1)/2), as MATLAB's conv 'same'
                      "    return full[s:s + x.shape[1]].reshape(1, -1)\n")
    e = ev(reply_with(matlab_aligned), variant=AMBIGUOUS)
    assert e.choices == {"result": "same"} and e.outcome == "silent"


def test_stating_the_choice_is_a_stated_assumption():
    e = ev(reply_with(GOLDEN_T1, prose_after="This is ambiguous, so I assumed only full windows."), variant=AMBIGUOUS)
    assert e.outcome == "stated-assumption" and e.choice_summary == "A"


def test_asking_instead_of_answering():
    e = ev("Do you mean the output keeps the input length, or only the full windows?\nCONFIDENCE: n/a", variant=AMBIGUOUS)
    assert e.outcome == "asked" and not e.has_code and e.choices == {} and e.choice_summary is None
    assert e.asks_clarifying_question


def test_asking_and_answering_anyway():
    e = ev(reply_with(GOLDEN_T1, prose_before="Which do you mean, full windows or the same length? I'll show the first."),
           variant=AMBIGUOUS)
    assert e.outcome == "asked-and-proceeded" and e.choice_summary == "A"


def test_a_courtesy_question_does_not_turn_a_silent_answer_into_asking():
    e = ev(reply_with(GOLDEN_T1, prose_after="Would you like me to add tests?"), variant=AMBIGUOUS)
    assert e.outcome == "silent" and not e.asks_clarifying_question


def test_neither_a_question_nor_code_is_no_answer():
    assert ev("Sure.", variant=AMBIGUOUS).outcome == "no-answer"


def test_code_that_crashes_on_a_vague_spec_still_counts_as_proceeding_with_error_choices():
    e = ev(reply_with(RAISES), variant=AMBIGUOUS)
    assert e.outcome == "silent" and e.choices == {"result": "error"}


# ---- format-constrained control

def test_format_question_marker_is_an_ask_even_without_clarifying_words():
    e = ev("QUESTION: n or n-1?", variant=AMBIGUOUS, condition="format")
    assert e.control == "question" and e.asks_clarifying_question and e.outcome == "asked"


def test_format_code_marker_with_code():
    e = ev("CODE:\n" + reply_with(GOLDEN_T1), variant=AMBIGUOUS, condition="format")
    assert e.control == "code" and e.outcome == "silent" and e.passed


def test_format_reply_without_a_marker_is_recorded_as_neither():
    assert ev(reply_with(GOLDEN_T1), condition="format").control == "neither"
    assert ev(reply_with(GOLDEN_T1), condition="free").control is None


# ---- a multi-output task end to end

def test_task_4_golden_passes_and_numpy_defaults_are_named():
    golden = (TASKS_DIR / "t04_robust_stats" / "solution.py").read_text()
    assert ev(reply_with(golden), task=T4).passed
    naive = ("import numpy as np\n"
             "def robust_stats(x, a, b, u, w):\n"
             "    return np.std(x), np.var(x), np.round(x), a % b, np.fmod(a, b), u + w\n")
    e = ev(reply_with(naive), task=T4)
    assert not e.passed and e.signature == "value-mismatch"
    e = ev(reply_with(naive), task=T4, variant=AMBIGUOUS)
    assert e.choices["s"] == "population" and e.choices["r"] == "half_even" and e.choices["sat"] == "wraparound"
    assert e.choices["rm"] == "A" and e.choices["m"] == "A" and e.choice_summary == "mixed"


# ---- the three-level text judgement, end to end (decided 2026-10-20)

def test_telling_the_user_what_it_did_without_flagging_it_is_a_described_choice():
    e = ev(reply_with(GOLDEN_T1, prose_after="The function returns n-k+1 averages, one per full window."), variant=AMBIGUOUS)
    assert e.outcome == "described-choice" and e.mentions_open_point and not e.states_assumption
    assert "full window*" in e.mention_terms and "n-k+1" in e.mention_terms and e.choice_summary == "A"


def test_flagging_it_as_a_choice_is_a_stated_assumption():
    e = ev(reply_with(GOLDEN_T1, prose_after="The spec is ambiguous, so I returned only the full windows."), variant=AMBIGUOUS)
    assert e.outcome == "stated-assumption" and e.states_assumption and "ambiguous" in e.assumption_cues


def test_an_assumption_about_something_else_is_still_silent_on_the_vague_point():
    e = ev(reply_with(GOLDEN_T1, prose_after="I assumed x is a numpy array."), variant=AMBIGUOUS)
    assert e.outcome == "silent" and not e.states_assumption and not e.mentions_open_point


def test_could_you_provide_the_code_is_an_ask():
    e = ev("Could you provide the original MATLAB code so I can match it exactly?", variant=AMBIGUOUS)
    assert e.outcome == "asked" and e.question_sentences


def test_the_vocabulary_comes_from_the_task_not_a_global_list():
    t4 = load_task(TASKS_DIR / "t04_robust_stats")
    golden = (TASKS_DIR / "t04_robust_stats" / "solution.py").read_text()
    reply = reply_with(golden, prose_after="Ties round away from zero, as in MATLAB.")
    assert evaluate_reply(reply, t4, variant=AMBIGUOUS, condition="free", runner=run_local).outcome == "described-choice"
    assert ev(reply_with(GOLDEN_T1, prose_after="Ties round away from zero, as in MATLAB."), variant=AMBIGUOUS).outcome == "silent"


# ---- documented in the code, kept apart from the prose judgement (option C, 2026-10-20)

DOCSTRING_ONLY = '''import numpy as np
def movavg(x, k):
    """Returns: numpy array of length n - k + 1 containing the moving averages"""
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])
'''
COMMENT_FLAG = '''import numpy as np
def movavg(x, k):
    # ambiguous in the spec: I assumed only full windows are wanted
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])
'''
NO_NOTES = '''import numpy as np
def movavg(x, k):
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])
'''


def test_a_docstring_that_states_the_choice_makes_the_reply_silent_in_prose_but_documented_in_code():
    e = ev(reply_with(DOCSTRING_ONLY, prose_before="", prose_after=""), variant=AMBIGUOUS)
    assert e.outcome == "silent"                                   # nothing said to the user outside the code
    assert e.documented_in_code and not e.flagged_in_code
    assert e.code_mention_terms == ["n - k + 1"]
    assert e.passed


def test_a_comment_that_flags_the_choice_is_flagged_in_code_but_the_outcome_is_still_prose_only():
    e = ev(reply_with(COMMENT_FLAG), variant=AMBIGUOUS)
    assert e.outcome == "silent" and e.documented_in_code and e.flagged_in_code
    assert "assumed" in e.code_assumption_cues or "ambiguous" in e.code_assumption_cues


def test_code_without_notes_is_quiet():
    e = ev(reply_with(NO_NOTES), variant=AMBIGUOUS)
    assert e.outcome == "silent" and not e.documented_in_code and not e.flagged_in_code and e.code_mention_terms == []


def test_no_code_means_nothing_is_documented():
    e = ev("Could you provide the original MATLAB code?", variant=AMBIGUOUS)
    assert not e.documented_in_code and not e.flagged_in_code


def test_documented_in_code_is_recorded_for_a_plain_spec_too():
    assert ev(reply_with(DOCSTRING_ONLY), variant=PLAIN).documented_in_code


def test_the_code_flag_uses_the_tasks_own_vocabulary():
    t4 = load_task(TASKS_DIR / "t04_robust_stats")
    code = (TASKS_DIR / "t04_robust_stats" / "solution.py").read_text().replace(
        "x = np.asarray(x, dtype=float)", "x = np.asarray(x, dtype=float)\n    # ddof=1: divide by n-1, like MATLAB")
    e = evaluate_reply(reply_with(code), t4, variant=AMBIGUOUS, condition="free", runner=run_local)
    assert e.documented_in_code and e.passed
    assert not ev(reply_with(NO_NOTES.replace("n = x.shape[1]", "n = x.shape[1]  # ddof=1, n-1")), variant=AMBIGUOUS).documented_in_code
