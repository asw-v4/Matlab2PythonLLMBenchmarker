import importlib.util
from pathlib import Path

import numpy as np
import pytest

from bench.scoring import compare_outputs
from bench.taskio import load_task

TASK_DIR = Path(__file__).parent.parent / "tasks" / "t02_colscale_diff"
TASK = load_task(TASK_DIR)
NAMES = TASK.output_names
TOL = dict(rtol=TASK.tolerance["rtol"], atol=TASK.tolerance["atol"])

# case index -> (rows, cols): 1 tall matrix, 2 wide matrix, 3 row vector, 4 column vector, 5 2x2
SHAPES = {1: (4, 3), 2: (3, 5), 3: (1, 5), 4: (5, 1), 5: (2, 2)}
ROW_CASE = 3


def _golden():
    spec = importlib.util.spec_from_file_location("t02_solution", TASK_DIR / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.colscale_diff


GOLDEN = _golden()
CASES = [pytest.param(c, id=f"case{c.index:02d}") for c in TASK.cases]


def run_check(outputs, case, vector_ok=None):
    ok = TASK.vector_ok if vector_ok is None else vector_ok
    return compare_outputs(tuple(outputs[n] for n in NAMES), case.expected, names=NAMES, vector_ok=ok, **TOL)


def independent(A):
    """A different route to the same answer: explicit loops, no min/max/diff axis arguments."""
    r, c = A.shape
    if r > 1 and c > 1:
        B = np.column_stack([(A[:, j] - A[:, j].min()) / (A[:, j].max() - A[:, j].min()) for j in range(c)])
        D = B[1:, :] - B[:-1, :]
    else:
        v = A.ravel()
        b = (v - v.min()) / (v.max() - v.min())
        B = b.reshape(A.shape)
        D = (b[1:] - b[:-1]).reshape((1, c - 1) if r == 1 else (r - 1, 1))
    return B, D


def test_task_loads_with_two_outputs_and_octave_version():
    assert TASK.id == "t02_colscale_diff"
    assert NAMES == ["B", "D"]
    assert TASK.vector_ok is False
    assert len(TASK.cases) == 5
    assert TASK.octave_version == (TASK_DIR.parent.parent / "octave" / "VERSION").read_text().strip()


def test_inputs_are_the_documented_shapes_with_no_constant_column():
    for c in TASK.cases:
        A = c.inputs["A"]
        assert A.shape == SHAPES[c.index]
        cols = [A.ravel()] if 1 in A.shape else [A[:, j] for j in range(A.shape[1])]
        assert all(col.max() > col.min() for col in cols), f"case {c.index} has a constant column"


@pytest.mark.parametrize("case", CASES)
def test_golden_solution_passes(case):
    v = compare_outputs(GOLDEN(case.inputs["A"]), case.expected, names=NAMES, vector_ok=False, **TOL)
    assert v.passed, v.reason


@pytest.mark.parametrize("case", CASES)
def test_octave_key_agrees_with_independent_formula(case):
    B, D = independent(case.inputs["A"])
    np.testing.assert_allclose(case.expected["B"], B, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(case.expected["D"], D, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_expected_shapes_follow_matlab_default_dimension(case):
    r, c = SHAPES[case.index]
    assert case.expected["B"].shape == (r, c)
    assert case.expected["D"].shape == ((1, c - 1) if r == 1 else (r - 1, c))


@pytest.mark.parametrize("case", CASES)
def test_scaled_output_spans_zero_to_one(case):
    B = case.expected["B"]
    cols = [B.ravel()] if 1 in B.shape else [B[:, j] for j in range(B.shape[1])]
    for col in cols:
        assert col.min() == 0.0 and col.max() == 1.0


def test_known_wrong_names_match_the_answer_key():
    for c in TASK.cases:
        assert set(c.wrong) == set(TASK.meta["known_wrong"])


def test_known_wrong_signatures_are_labelled():
    assert set(TASK.meta["known_wrong"].values()) <= {"nan-output", "shape-mismatch", "transposed-output"}


# Which cases each wrong translation is expected to fail on (it is right where the conventions agree).
FAILS_ON = {
    "axis0": {ROW_CASE},
    "diff_last_axis": {1, 2, 4, 5},
    "transposed": {1, 2, 3, 4, 5},
}


@pytest.mark.parametrize("name", sorted(FAILS_ON))
def test_each_known_wrong_variant_fails_exactly_where_expected(name):
    failed = {c.index for c in TASK.cases if not run_check(c.wrong[name], c).passed}
    assert failed == FAILS_ON[name]


def test_every_known_wrong_variant_fails_somewhere():
    assert set(FAILS_ON) == set(TASK.meta["known_wrong"])
    assert all(FAILS_ON.values())


def test_axis0_on_a_row_vector_is_all_nan_and_has_no_differences():
    c = TASK.cases[ROW_CASE - 1]
    assert np.isnan(c.wrong["axis0"]["B"]).all()
    assert c.wrong["axis0"]["D"].shape == (0, 5)


@pytest.mark.parametrize("case", CASES)
def test_transposed_variant_is_exactly_the_transpose_of_the_key(case):
    for n in NAMES:
        np.testing.assert_array_equal(case.wrong["transposed"][n], case.expected[n].T)


def test_a_1d_result_does_not_pass_for_a_row_vector_here():
    c = TASK.cases[ROW_CASE - 1]
    flat = (c.expected["B"].ravel(), c.expected["D"].ravel())
    assert not compare_outputs(flat, c.expected, names=NAMES, vector_ok=False, **TOL).passed
    assert compare_outputs(flat, c.expected, names=NAMES, vector_ok=True, **TOL).passed  # would pass if allowed


@pytest.mark.parametrize("case", CASES)
def test_global_reading_differs_on_matrices_and_coincides_on_vectors(case):
    reading = case.readings["global"]
    same = run_check(reading, case).passed
    is_vector = 1 in case.inputs["A"].shape
    assert same == is_vector


def test_golden_does_not_match_the_global_reading_on_matrices():
    for c in TASK.cases:
        if 1 not in c.inputs["A"].shape:
            B, D = GOLDEN(c.inputs["A"])
            assert not compare_outputs((B, D), c.readings["global"], names=NAMES, **TOL).passed


def test_specs_hide_answers_and_ambiguous_one_omits_code_and_the_resolution():
    spec = (TASK_DIR / "spec.md").read_text()
    amb = (TASK_DIR / "spec_ambiguous.md").read_text()
    assert "min(A)" in spec and "diff(B)" in spec
    # shape remarks ("a single row or column") are fine; saying how to scale is not
    for banned in ("min(", "max(", "diff(B)", "per column", "each column", "column-wise", "columnwise",
                   "per row", "each row", "overall", "whole array"):
        assert banned not in amb.lower(), banned
    for text in (spec, amb):
        assert "expected" not in text.lower()
