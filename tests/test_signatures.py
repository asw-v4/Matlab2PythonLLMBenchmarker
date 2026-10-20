from pathlib import Path

import numpy as np
import pytest

from bench.runs import ERROR, TIMEOUT, BAD_OUTPUT, CaseRun
from bench.signatures import classify_case, classify_sample, generic_signature
from bench.taskio import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())
TASKS = {p.name: load_task(p) for p in TASK_DIRS}
IDS = sorted(TASKS)


def run_from(outputs: dict, task) -> CaseRun:
    return CaseRun.ok([outputs[n] for n in task.output_names])


# ---- the whole pipeline on every real task

@pytest.mark.parametrize("tid", IDS)
def test_the_key_itself_passes_every_case(tid):
    task = TASKS[tid]
    runs = [run_from(c.expected, task) for c in task.cases]
    s = classify_sample(runs, task)
    assert s.passed and s.signature == "pass" and s.n_failed == 0


@pytest.mark.parametrize("tid", IDS)
def test_each_known_wrong_translation_gets_its_declared_label_on_the_cases_it_fails(tid):
    task = TASKS[tid]
    labels = task.meta["known_wrong"]
    for name, label in labels.items():
        runs = [run_from(c.wrong[name], task) for c in task.cases]
        res = classify_sample(runs, task)
        failing = {r.signature for r in res.cases if not r.passed}
        assert failing, f"{tid}/{name} never fails"
        assert failing == {label}, f"{tid}/{name}: got {failing}, declared {label}"
        assert res.signature == label and not res.passed
        for r in res.cases:                       # where the wrong translation happens to be right, it passes
            assert r.passed or r.signature == label


@pytest.mark.parametrize("tid", IDS)
def test_a_failing_case_names_the_matching_variants_in_the_detail(tid):
    task = TASKS[tid]
    for name in task.meta["known_wrong"]:
        for c in task.cases:
            r = classify_case(run_from(c.wrong[name], task), c, task)
            if not r.passed:
                assert name in r.detail
                return
    pytest.fail("no failing case found")


# ---- runs that did not produce an answer

@pytest.mark.parametrize("status,label", [(TIMEOUT, "timeout"), (ERROR, "runtime-error"), (BAD_OUTPUT, "bad-output")])
def test_runs_that_did_not_finish(status, label):
    task = TASKS["t01_movavg"]
    r = classify_case(CaseRun.failed(status, "boom"), task.cases[0], task)
    assert (r.passed, r.signature, r.detail) == (False, label, "boom")


def test_returning_one_array_for_a_two_output_task_is_a_wrong_output_count():
    task = TASKS["t02_colscale_diff"]
    c = task.cases[0]
    r = classify_case(CaseRun.ok([c.expected["B"]]), c, task)
    assert r.signature == "wrong-output-count"
    r = classify_case(CaseRun.ok([c.expected["B"], c.expected["D"], c.expected["D"]]), c, task)
    assert r.signature == "wrong-output-count"


def test_wrong_number_of_case_runs_is_rejected():
    task = TASKS["t01_movavg"]
    with pytest.raises(ValueError):
        classify_sample([CaseRun.failed(ERROR, "x")], task)


# ---- a failure that matches no known-wrong translation gets a generic label

def test_generic_labels_on_a_real_task():
    task = TASKS["t01_movavg"]
    c = task.cases[0]                                  # expected is a 1x6 row
    e = c.expected["result"]
    cases = {
        "nan-output": np.full_like(e, np.nan),
        "transposed-output": e.T.copy() * 1.0,
        "scale-mismatch": e * 2,
        "permuted-output": e[:, ::-1].copy(),
        "value-mismatch": e + 0.5,
        "shifted-output": e[:, 1:],
    }
    for label, bad in cases.items():
        r = classify_case(CaseRun.ok([bad]), c, task)
        assert (r.passed, r.signature) == (False, label), (label, r)


def test_a_failure_in_the_second_output_is_labelled_from_that_output():
    task = TASKS["t02_colscale_diff"]
    c = task.cases[0]
    bad = c.expected["D"] * 3
    r = classify_case(CaseRun.ok([c.expected["B"], bad]), c, task)
    assert r.signature == "scale-mismatch" and r.detail.startswith("D:")


# ---- generic_signature on its own

E = np.array([[1.0, 2.0, 3.0, 4.0]])


@pytest.mark.parametrize("actual,label", [
    (np.array([[2.0, 3.0, 4.0]]), "shifted-output"),            # first value missing
    (np.array([[1.0, 2.0, 3.0]]), "shifted-output"),            # last value missing
    (np.array([[3.0, 4.0]]), "shifted-output"),                 # two missing
    (np.array([[4.0]]), "shape-mismatch"),                      # three missing: not a small shift
    (np.array([[1.0, 2.0, 3.0, 4.0, 5.0]]), "shape-mismatch"),  # too long
    (np.array([1.0, 2.0, 3.0, 4.0]), "shape-mismatch"),         # right values, wrong shape
    (E.reshape(2, 2), "shape-mismatch"),
    (E.T.copy(), "transposed-output"),
    (np.array([[4.0, 3.0, 2.0, 1.0]]), "permuted-output"),
    (np.array([[2.0, 4.0, 6.0, 8.0]]), "scale-mismatch"),
    (np.array([[0.5, 1.0, 1.5, 2.0]]), "scale-mismatch"),
    (np.array([[1.0, 2.0, 3.0, 5.0]]), "value-mismatch"),
    (np.array([[np.nan, 2.0, 3.0, 4.0]]), "nan-output"),
    (np.array([["a", "b", "c", "d"]]), "bad-output"),
])
def test_generic_signature(actual, label):
    assert generic_signature(actual, E)[0] == label


def test_scale_needs_the_zeros_to_stay_zero():
    e = np.array([[0.0, 1.0, 2.0]])
    assert generic_signature(np.array([[0.0, 2.0, 4.0]]), e)[0] == "scale-mismatch"
    assert generic_signature(np.array([[1.0, 2.0, 4.0]]), e)[0] == "value-mismatch"


def test_a_scale_needs_at_least_two_nonzero_values():
    assert generic_signature(np.array([[0.0, 6.0]]), np.array([[0.0, 3.0]]))[0] == "value-mismatch"
    assert generic_signature(np.array([[0.0, 6.0, 2.0]]), np.array([[0.0, 3.0, 1.0]]))[0] == "scale-mismatch"


def test_square_matrix_transposed():
    e = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert generic_signature(e.T.copy(), e)[0] == "permuted-output"      # same shape, same values: not a shape problem
    assert generic_signature(np.array([[1.0, 2.0, 3.0]]).T, np.array([[1.0, 2.0, 3.0]]))[0] == "transposed-output"


# ---- sample-level label

def test_sample_label_is_the_commonest_failing_label_and_ties_go_alphabetical():
    task = TASKS["t04_robust_stats"]
    wrap, ddof, rnd = (task.meta["known_wrong"][k] for k in ("wraparound", "ddof0", "round_half_even"))

    def runs(variants):          # per case: the name of the wrong variant to return, or None for the key
        return [run_from(c.wrong[v] if v else c.expected, task) for c, v in zip(task.cases, variants)]

    # wraparound breaks cases 1, 4 and 5; rounding half to even breaks cases 1, 3 and 4 (case 5 has no ties)
    s = classify_sample(runs(["wraparound", None, "round_half_even", "wraparound", None]), task)
    assert not s.passed and s.signature == wrap and s.n_failed == 3
    assert [r.signature for r in s.cases] == [wrap, "pass", rnd, wrap, "pass"]
    s = classify_sample(runs(["ddof0", None, None, "wraparound", None]), task)    # a tie: scale vs wraparound
    assert s.signature == sorted([ddof, wrap])[0]
    assert rnd == "rounding-mismatch"


# ---- single numbers

def test_a_scalar_against_a_1x1_key_is_judged_on_its_value_not_its_shape():
    e = np.array([[2.0]])
    assert generic_signature(np.float64(3.0), e)[0] == "value-mismatch"
    assert generic_signature(np.array(4.0), e)[0] == "value-mismatch"        # twice the key, but one number proves nothing
    assert generic_signature(np.array([5.0]), e)[0] == "value-mismatch"
    assert generic_signature(np.array(np.nan), e)[0] == "nan-output"
    assert generic_signature(np.array([1.0, 2.0]), e)[0] == "shape-mismatch"  # two numbers are not one
