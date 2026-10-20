import numpy as np

from bench.scoring import compare

KW = dict(rtol=1e-12, atol=1e-12)
ROW = np.array([[1.0, 2.0, 3.0]])


def test_exact_match_passes():
    assert compare(ROW.copy(), ROW, **KW).passed


def test_within_tolerance_passes():
    assert compare(ROW * (1 + 1e-14), ROW, **KW).passed


def test_outside_tolerance_fails():
    v = compare(ROW * (1 + 1e-6), ROW, **KW)
    assert not v.passed and v.reason == "value-mismatch"


def test_atol_handles_values_near_zero():
    assert compare(np.array([[1e-15]]), np.array([[0.0]]), **KW).passed


def test_shape_mismatch_fails_with_shapes_in_reason():
    v = compare(np.zeros((1, 4)), ROW, **KW)
    assert not v.passed and v.reason.startswith("shape-mismatch") and "(1, 4)" in v.reason


def test_1d_rejected_by_default():
    assert not compare(np.array([1.0, 2.0, 3.0]), ROW, **KW).passed


def test_1d_accepted_with_vector_ok():
    assert compare(np.array([1.0, 2.0, 3.0]), ROW, vector_ok=True, **KW).passed


def test_1d_wrong_length_rejected_even_with_vector_ok():
    assert not compare(np.array([1.0, 2.0]), ROW, vector_ok=True, **KW).passed


def test_column_never_accepted_for_row():
    assert not compare(ROW.T.copy(), ROW, vector_ok=True, **KW).passed


def test_matrix_expected_does_not_get_vector_leniency():
    m = np.arange(6.0).reshape(2, 3)
    assert not compare(m.ravel(), m, vector_ok=True, **KW).passed


def test_nan_is_a_mismatch():
    assert not compare(np.array([[np.nan]]), np.array([[np.nan]]), **KW).passed


def test_non_numeric_fails():
    assert compare("hello", ROW, **KW).reason == "not-numeric"
    assert compare(None, ROW, **KW).reason == "not-numeric"
    assert compare(np.array(["a"], dtype=object), ROW, **KW).reason == "not-numeric"


def test_empty_arrays_compare_by_shape():
    assert compare(np.zeros((1, 0)), np.zeros((1, 0)), **KW).passed
    assert not compare(np.zeros((1, 0)), ROW, **KW).passed
