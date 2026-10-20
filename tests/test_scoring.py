import numpy as np

from bench.scoring import compare, compare_outputs

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


# ---- compare_outputs: whole return values

B = np.array([[0.0, 1.0]])
D = np.array([[1.0]])
EXP2 = {"B": B, "D": D}


def test_multi_output_passes_for_tuple_or_list_in_order():
    assert compare_outputs((B, D), EXP2, names=["B", "D"], **KW).passed
    assert compare_outputs([B, D], EXP2, names=["B", "D"], **KW).passed


def test_multi_output_wrong_order_fails_on_the_first_output():
    v = compare_outputs((D, B), EXP2, names=["B", "D"], **KW)
    assert not v.passed and v.reason.startswith("B:")


def test_multi_output_names_the_failing_output():
    v = compare_outputs((B, D + 1), EXP2, names=["B", "D"], **KW)
    assert not v.passed and v.reason == "D: value-mismatch"


def test_multi_output_wrong_count_or_not_a_sequence():
    assert compare_outputs((B,), EXP2, names=["B", "D"], **KW).reason.startswith("wrong-number-of-outputs")
    assert compare_outputs((B, D, D), EXP2, names=["B", "D"], **KW).reason.startswith("wrong-number-of-outputs")
    assert compare_outputs(B, EXP2, names=["B", "D"], **KW).reason.startswith("wrong-number-of-outputs")


def test_single_output_accepts_bare_array_or_one_tuple_but_not_a_longer_tuple():
    exp = {"result": ROW}
    assert compare_outputs(ROW.copy(), exp, names=["result"], **KW).passed
    assert compare_outputs((ROW.copy(),), exp, names=["result"], **KW).passed
    assert not compare_outputs((ROW.copy(), ROW.copy()), exp, names=["result"], **KW).passed


def test_vector_ok_is_passed_through_per_output():
    exp = {"result": ROW}
    assert compare_outputs(np.array([1.0, 2.0, 3.0]), exp, names=["result"], vector_ok=True, **KW).passed
    assert not compare_outputs(np.array([1.0, 2.0, 3.0]), exp, names=["result"], vector_ok=False, **KW).passed


# ---- single numbers

ONE = np.array([[2.5]])


def test_a_single_number_matches_in_any_scalar_shape():
    for form in (2.5, np.float64(2.5), np.array(2.5), np.array([2.5]), np.array([[2.5]])):
        assert compare(form, ONE, **KW).passed, repr(form)


def test_a_single_number_still_has_to_be_the_right_value():
    assert not compare(2.6, ONE, **KW).passed


def test_scalar_leniency_does_not_apply_to_longer_arrays():
    assert not compare(np.array([2.5, 2.5]), ONE, **KW).passed
    assert not compare(2.5, ROW, **KW).passed
