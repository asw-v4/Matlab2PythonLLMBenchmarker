import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson as scipy_simpson
from scipy.io import loadmat

from bench.scoring import compare_outputs
from bench.taskio import load_task

TASK_DIR = Path(__file__).parent.parent / "tasks" / "t03_simpson"
TASK = load_task(TASK_DIR)
MAT = loadmat(TASK_DIR / "expected.mat")
NAMES = TASK.output_names
TOL = dict(rtol=TASK.tolerance["rtol"], atol=TASK.tolerance["atol"])


def _load(name):
    spec = importlib.util.spec_from_file_location(f"t03_{name}", TASK_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FUNCS = _load("functions").FUNCS
TRUE_VALUES = _load("functions").TRUE_VALUES
GOLDEN = _load("solution").simpson
CASES = [pytest.param(c, id=f"case{c.index:02d}") for c in TASK.cases]


def passes(value, expected):
    return compare_outputs(value, expected, names=NAMES, **TOL).passed


def args(case):
    i = case.inputs
    return FUNCS[i["f"]], i["a"], i["b"], i["n"]


def test_task_loads_with_five_cases_and_octave_version():
    assert TASK.id == "t03_simpson" and NAMES == ["result"] and len(TASK.cases) == 5
    assert TASK.octave_version == (TASK_DIR.parent.parent / "octave" / "VERSION").read_text().strip()


def test_inputs_have_documented_types_and_n_is_even():
    for c in TASK.cases:
        assert isinstance(c.inputs["f"], str) and c.inputs["f"] in FUNCS
        assert isinstance(c.inputs["a"], float) and isinstance(c.inputs["b"], float)
        assert isinstance(c.inputs["n"], int) and c.inputs["n"] % 2 == 0 and c.inputs["n"] >= 2


@pytest.mark.parametrize("case", CASES)
def test_python_function_definitions_agree_with_the_octave_ones(case):
    k = f"case{case.index:02d}_"
    x, fx = MAT[k + "check_x"].ravel(), MAT[k + "check_f"].ravel()
    np.testing.assert_allclose(FUNCS[case.inputs["f"]](x), fx, rtol=1e-14, atol=1e-16)


@pytest.mark.parametrize("case", CASES)
def test_golden_solution_passes(case):
    assert passes(GOLDEN(*args(case)), case.expected)


@pytest.mark.parametrize("case", CASES)
def test_octave_key_agrees_with_scipy_simpson_on_octaves_own_samples(case):
    k = f"case{case.index:02d}_"
    ref = scipy_simpson(MAT[k + "check_f"].ravel(), x=MAT[k + "check_x"].ravel())
    np.testing.assert_allclose(case.expected["result"].item(), ref, rtol=1e-12, atol=1e-14)


def test_simpson_is_exact_for_the_cubic():
    assert abs(TASK.cases[0].expected["result"].item() - 4.0) < 1e-14


@pytest.mark.parametrize("case", CASES)
def test_key_is_close_to_the_true_integral(case):
    i = case.inputs
    if i["f"] == "gauss":
        true = math.sqrt(math.pi) / 2 * math.erf(2.0)
    else:
        true = TRUE_VALUES[(i["f"], i["a"], i["b"])]
    assert abs(case.expected["result"].item() - true) / abs(true) < 1e-2


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", sorted(TASK.meta["known_wrong"]))
def test_every_known_wrong_variant_fails_on_every_case(case, name):
    assert not passes(case.wrong[name]["result"], case.expected), f"{name} passes on case {case.index}"


def test_known_wrong_names_and_labels():
    assert set(TASK.cases[0].wrong) == set(TASK.meta["known_wrong"])
    assert set(TASK.meta["known_wrong"].values()) <= {"endpoint-mismatch", "weights-mismatch"}


# ---- the wrong answers are what numpy code would actually produce

def _weights(y, w_odd=4.0, w_even=2.0):
    return y[0] + w_odd * y[1:-1:2].sum() + w_even * y[2:-2:2].sum() + y[-1]


@pytest.mark.parametrize("case", CASES)
def test_no_endpoint_matches_what_np_arange_gives(case):
    f, a, b, n = args(case)
    h = (b - a) / n
    x = np.arange(a, b, h)
    assert len(x) == n, "np.arange happens to give n points for these inputs"
    got = h / 3 * _weights(f(x))
    np.testing.assert_allclose(got, case.wrong["no_endpoint"]["result"].item(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", CASES)
def test_linspace_n_matches_np_linspace_with_n_points(case):
    f, a, b, n = args(case)
    got = (b - a) / n / 3 * _weights(f(np.linspace(a, b, n)))
    np.testing.assert_allclose(got, case.wrong["linspace_n"]["result"].item(), rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("case", CASES)
def test_weights_swapped_matches_swapped_numpy_slices(case):
    f, a, b, n = args(case)
    got = (b - a) / n / 3 * _weights(f(np.linspace(a, b, n + 1)), w_odd=2.0, w_even=4.0)
    np.testing.assert_allclose(got, case.wrong["weights_swapped"]["result"].item(), rtol=1e-12, atol=1e-14)


# ---- ambiguity variant

@pytest.mark.parametrize("case", CASES)
def test_panels_reading_is_simpson_with_twice_the_subintervals_and_differs_from_the_key(case):
    f, a, b, n = args(case)
    reading = case.readings["panels"]
    assert passes(GOLDEN(f, a, b, 2 * n), reading)
    if case.inputs["f"] == "cubic":
        # Simpson's rule is exact for a cubic, so both readings give the same number
        assert passes(reading["result"], case.expected)
    else:
        assert not passes(reading["result"], case.expected)


def test_only_the_cubic_case_cannot_tell_the_two_readings_apart():
    same = [c.index for c in TASK.cases if passes(c.readings["panels"]["result"], c.expected)]
    assert same == [1]


def test_specs_hide_answers_and_ambiguous_one_leaves_the_meaning_of_n_open():
    spec = (TASK_DIR / "spec.md").read_text()
    amb = (TASK_DIR / "spec_ambiguous.md").read_text().lower()
    assert "linspace(a, b, n+1)" in spec and "even" in spec
    assert "panels" in amb
    for banned in ("linspace", "even", "subinterval", "interval", "points", "n+1", "end-1", "h/3", "y("):
        assert banned not in amb, banned
    for text in (spec.lower(), amb):
        assert "expected" not in text
