import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bench.scoring import compare
from bench.taskio import load_task

TASK_DIR = Path(__file__).parent.parent / "tasks" / "t01_movavg"
TASK = load_task(TASK_DIR)
TOL = dict(rtol=TASK.tolerance["rtol"], atol=TASK.tolerance["atol"])


def _golden():
    spec = importlib.util.spec_from_file_location("t01_solution", TASK_DIR / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.movavg


GOLDEN = _golden()
CASES = [pytest.param(c, id=f"case{c.index:02d}") for c in TASK.cases]


def test_task_loads_with_five_cases_and_octave_version():
    assert TASK.id == "t01_movavg"
    assert len(TASK.cases) == 5
    assert TASK.octave_version == (TASK_DIR.parent.parent / "octave" / "VERSION").read_text().strip()


def test_inputs_have_documented_types():
    for c in TASK.cases:
        assert isinstance(c.inputs["k"], int)
        assert c.inputs["x"].ndim == 2 and c.inputs["x"].shape[0] == 1


@pytest.mark.parametrize("case", CASES)
def test_golden_solution_passes(case):
    out = GOLDEN(case.inputs["x"], case.inputs["k"])
    v = compare(out, case.expected, vector_ok=TASK.vector_ok, **TOL)
    assert v.passed, v.reason


@pytest.mark.parametrize("case", CASES)
def test_octave_key_agrees_with_independent_numpy_formula(case):
    x = case.inputs["x"].ravel()
    k = case.inputs["k"]
    independent = np.convolve(x, np.ones(k) / k, mode="valid")
    np.testing.assert_allclose(case.expected.ravel(), independent, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_expected_has_n_minus_k_plus_1_entries(case):
    assert case.expected.shape == (1, case.inputs["x"].shape[1] - case.inputs["k"] + 1)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", sorted(TASK.meta["known_wrong"]))
def test_every_known_wrong_variant_fails(case, name):
    v = compare(case.wrong[name], case.expected, vector_ok=TASK.vector_ok, **TOL)
    assert not v.passed, f"{name} unexpectedly passes on case {case.index}"


def test_known_wrong_names_match_the_answer_key():
    for c in TASK.cases:
        assert set(c.wrong) == set(TASK.meta["known_wrong"])


def test_known_wrong_signatures_are_labelled():
    assert set(TASK.meta["known_wrong"].values()) <= {"shifted-output", "window-mismatch"}


@pytest.mark.parametrize("case", CASES)
def test_same_length_reading_differs_from_valid_reading_when_k_gt_1(case):
    reading = case.readings["same"]
    assert reading.shape == case.inputs["x"].shape
    if case.inputs["k"] > 1:
        assert not compare(reading, case.expected, **TOL).passed
    else:
        assert compare(reading, case.expected, **TOL).passed  # k=1: both readings coincide


def test_golden_matches_the_same_length_reading_for_no_case_with_k_gt_1():
    for c in TASK.cases:
        if c.inputs["k"] > 1:
            out = GOLDEN(c.inputs["x"], c.inputs["k"])
            assert not compare(out, c.readings["same"], vector_ok=True, **TOL).passed


def test_specs_hide_answers_and_ambiguous_one_omits_the_code():
    spec = (TASK_DIR / "spec.md").read_text()
    amb = (TASK_DIR / "spec_ambiguous.md").read_text()
    assert "for i = 1:(n-k+1)" in spec
    assert "for i" not in amb and "n-k+1" not in amb
    for text in (spec, amb):
        assert "expected" not in text.lower() and "[2 3 4" not in text
