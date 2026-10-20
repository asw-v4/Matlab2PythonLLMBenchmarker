import importlib.util
import math
import statistics
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
import pytest

from bench.scoring import compare, compare_outputs
from bench.taskio import load_task

TASK_DIR = Path(__file__).parent.parent / "tasks" / "t04_robust_stats"
TASK = load_task(TASK_DIR)
NAMES = TASK.output_names
EXACT = TASK.exact_outputs
TOL = dict(rtol=TASK.tolerance["rtol"], atol=TASK.tolerance["atol"])
ARGS = ["x", "a", "b", "u", "w"]


def _load():
    spec = importlib.util.spec_from_file_location("t04_solution", TASK_DIR / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOL = _load()
CASES = [pytest.param(c, id=f"case{c.index:02d}") for c in TASK.cases]


def call(fn, case):
    return fn(*(case.inputs[k] for k in ARGS))


def check(outputs, case):
    """Compare a dict of named outputs with the key."""
    return compare_outputs(tuple(outputs[n] for n in NAMES), case.expected, names=NAMES,
                           vector_ok=TASK.vector_ok, exact=EXACT, **TOL)


def differing_outputs(wrong, case):
    return {n for n in NAMES
            if not compare(wrong[n], case.expected[n], vector_ok=True,
                           rtol=0.0 if n in EXACT else TOL["rtol"], atol=0.0 if n in EXACT else TOL["atol"]).passed}


def test_task_loads_with_six_outputs_four_of_them_exact():
    assert TASK.id == "t04_robust_stats" and len(TASK.cases) == 5
    assert NAMES == ["s", "v", "r", "m", "rm", "sat"]
    assert EXACT == ["r", "m", "rm", "sat"] and TASK.vector_ok is True
    assert TASK.octave_version == (TASK_DIR.parent.parent / "octave" / "VERSION").read_text().strip()


def test_inputs_have_the_documented_types():
    for c in TASK.cases:
        i = c.inputs
        for k in ("x", "a", "b"):
            assert i[k].dtype == np.float64 and i[k].ndim == 2 and i[k].shape[0] == 1
        for k in ("u", "w"):
            assert i[k].dtype == np.uint8 and i[k].shape == i["u"].shape
        assert i["a"].shape == i["b"].shape and (i["b"] != 0).all()
        assert (i["a"] == np.round(i["a"])).all() and (i["b"] == np.round(i["b"])).all()


@pytest.mark.parametrize("case", CASES)
def test_golden_solution_passes(case):
    out = dict(zip(NAMES, call(SOL.robust_stats, case)))
    v = check(out, case)
    assert v.passed, v.reason


@pytest.mark.parametrize("case", CASES)
def test_octave_key_agrees_with_pure_python_standard_library(case):
    i, e = case.inputs, case.expected
    x = i["x"].ravel().tolist()
    np.testing.assert_allclose(e["s"].item(), statistics.stdev(x), rtol=1e-12)
    np.testing.assert_allclose(e["v"].item(), statistics.variance(x), rtol=1e-12)
    away = [float(Decimal(repr(t)).quantize(Decimal(1), rounding=ROUND_HALF_UP)) for t in x]  # ties away from zero
    assert e["r"].ravel().tolist() == away
    a, b = i["a"].ravel().astype(int).tolist(), i["b"].ravel().astype(int).tolist()
    assert e["m"].ravel().tolist() == [p % q for p, q in zip(a, b)]            # Python's % follows the divisor
    assert e["rm"].ravel().tolist() == [math.fmod(p, q) for p, q in zip(a, b)]  # fmod follows the dividend
    u, w = i["u"].ravel().astype(int).tolist(), i["w"].ravel().astype(int).tolist()
    assert e["sat"].ravel().tolist() == [min(p + q, 255) for p, q in zip(u, w)]


def test_octave_returns_uint8_for_the_saturating_sum():
    for c in TASK.cases:
        assert c.expected["sat"].dtype == np.uint8


def test_the_key_actually_exercises_each_trap():
    # ties in x, opposite signs in a and b, and sums over 255 all occur somewhere
    assert any((np.abs(c.inputs["x"] - np.fix(c.inputs["x"])) == 0.5).any() for c in TASK.cases)
    assert any((np.sign(c.inputs["a"]) != np.sign(c.inputs["b"])).any() for c in TASK.cases)
    assert any((c.inputs["u"].astype(int) + c.inputs["w"].astype(int) > 255).any() for c in TASK.cases)


# ---- known-wrong variants

def test_known_wrong_names_and_labels():
    assert set(TASK.cases[0].wrong) == set(TASK.meta["known_wrong"])
    assert set(TASK.meta["known_wrong"].values()) <= {
        "scale-mismatch", "rounding-mismatch", "sign-mismatch", "wraparound-mismatch"}


# Each wrong translation changes only the outputs named here, and only on the cases where the
# difference can show (e.g. no ties in x means rounding half to even gives the same answer).
BREAKS = {
    "ddof0": ({"s", "v"}, {1, 2, 3, 4, 5}),
    "round_half_even": ({"r"}, {1, 3, 4}),
    "rem_as_mod": ({"rm"}, {1, 4, 5}),
    "mod_as_rem": ({"m"}, {1, 4, 5}),
    "wraparound": ({"sat"}, {1, 4, 5}),
}


def test_breaks_table_covers_every_variant():
    assert set(BREAKS) == set(TASK.meta["known_wrong"])


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_each_wrong_variant_breaks_only_its_own_outputs(name):
    outputs, cases = BREAKS[name]
    for c in TASK.cases:
        diff = differing_outputs(c.wrong[name], c)
        assert diff <= outputs, f"{name} changed {diff - outputs} on case {c.index}"
        assert (diff == outputs) == (c.index in cases), f"{name} on case {c.index}: {diff}"


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_each_wrong_variant_fails_the_whole_check_exactly_where_it_should(name):
    failed = {c.index for c in TASK.cases if not check(c.wrong[name], c).passed}
    assert failed == BREAKS[name][1]


# ---- the wrong answers are what numpy code would actually produce

@pytest.mark.parametrize("case", CASES)
def test_numpy_defaults_reproduce_the_octave_wrong_answers(case):
    i, w = case.inputs, case.wrong
    np.testing.assert_allclose(np.std(i["x"]), w["ddof0"]["s"].item(), rtol=1e-12)      # ddof=0 is the default
    np.testing.assert_allclose(np.var(i["x"]), w["ddof0"]["v"].item(), rtol=1e-12)
    np.testing.assert_array_equal(np.round(i["x"]), w["round_half_even"]["r"])           # half to even
    np.testing.assert_array_equal(i["a"] % i["b"], w["rem_as_mod"]["rm"])                # % is mod, not rem
    np.testing.assert_array_equal(np.fmod(i["a"], i["b"]), w["mod_as_rem"]["m"])         # fmod is rem, not mod
    with np.errstate(all="ignore"):
        np.testing.assert_array_equal(i["u"] + i["w"], w["wraparound"]["sat"])           # uint8 + uint8 wraps


# ---- ambiguity variant: the same function, described in words

# reading -> (the known-wrong variant it is the same computation as, the outputs it concerns)
READINGS = {
    "population": ("ddof0", {"s", "v"}),
    "half_even": ("round_half_even", {"r"}),
    "truncated_modulo": ("mod_as_rem", {"m"}),
    "floored_remainder": ("rem_as_mod", {"rm"}),
    "wraparound": ("wraparound", {"sat"}),
}


def tol_for(name):
    return (dict(rtol=0.0, atol=0.0) if name in EXACT
            else dict(rtol=TOL["rtol"], atol=TOL["atol"]))


def test_ambiguity_is_the_same_function_with_all_six_outputs():
    assert TASK.meta["ambiguity"]["function"] == "robust_stats" and TASK.meta["ambiguity"]["outputs"] == NAMES
    assert TASK.meta["ambiguity"]["function"] == TASK.meta["task"]["function"]
    assert set(READINGS) == set(TASK.meta["readings"])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reading", sorted(READINGS))
def test_each_reading_concerns_only_its_outputs_and_equals_the_matching_wrong_variant(reading, case):
    variant, outs = READINGS[reading]
    got = case.readings[reading]
    assert set(got) == outs
    for n in outs:
        np.testing.assert_array_equal(got[n], case.wrong[variant][n])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("reading", sorted(READINGS))
def test_each_reading_differs_from_the_key_exactly_where_its_variant_does(reading, case):
    variant, outs = READINGS[reading]
    differs = {n for n in outs
               if not compare(case.readings[reading][n], case.expected[n], vector_ok=True, **tol_for(n)).passed}
    assert (differs == outs) == (case.index in BREAKS[variant][1])
    assert differs in (set(), outs)          # a reading either changes all its outputs or none, on a given case


@pytest.mark.parametrize("case", CASES)
def test_reading_a_is_the_key_and_golden_matches_it_not_the_readings_where_they_differ(case):
    out = dict(zip(NAMES, call(SOL.robust_stats, case)))
    assert check(out, case).passed
    for reading, (variant, outs) in READINGS.items():
        if case.index in BREAKS[variant][1]:
            for n in outs:
                assert not compare(out[n], case.readings[reading][n], vector_ok=True, **tol_for(n)).passed


def test_population_reading_agrees_with_the_standard_library():
    for c in TASK.cases:
        x = c.inputs["x"].ravel().tolist()
        np.testing.assert_allclose(c.readings["population"]["s"].item(), statistics.pstdev(x), rtol=1e-12)
        np.testing.assert_allclose(c.readings["population"]["v"].item(), statistics.pvariance(x), rtol=1e-12)


# ---- specs

def test_spec_has_no_hints_in_the_code_and_states_the_input_types():
    spec = (TASK_DIR / "spec.md").read_text()
    code = spec.split("```matlab")[1].split("```")[0]
    assert "%" not in code, "comments in the MATLAB would give the traps away"
    for line in ("std(x)", "var(x)", "round(x)", "mod(a, b)", "rem(a, b)", "u + w"):
        assert line in code
    assert "uint8" in spec


def test_ambiguous_spec_is_the_same_task_in_words_and_settles_nothing():
    spec = (TASK_DIR / "spec.md").read_text()
    raw = (TASK_DIR / "spec_ambiguous.md").read_text()
    amb = raw.lower()

    # same function, signature, input types and return order as the plain spec
    for text in (spec, raw):
        for must in ("robust_stats(x, a, b, u, w)", "(s, v, r, m, rm, sat)", "uint8", "no zeros",
                     "2-D numpy arrays"):
            assert must in text, must

    # described in words: one phrase per line of the MATLAB
    for phrase in ("standard deviation of `x`", "variance of `x`", "rounded to whole numbers", "modulo",
                   "remainder of `a` divided by `b`", "`u` plus `w`"):
        assert phrase in amb, phrase
    for banned in ("```", "std(", "var(", "round(", "mod(", "rem(", "u + w", "matlab's"):
        assert banned not in amb, banned

    # nothing that settles any of the ambiguities
    for banned in ("n-1", "n - 1", "sample", "population", "unbiased", "ddof", "bessel", "half", "tie",
                   "away from zero", "even", "floor", "truncat", "sign", "saturat", "wrap", "overflow",
                   "clip", "255", "fmod", "c-style"):
        assert banned not in amb, banned
    for text in (amb, spec.lower()):
        assert "expected" not in text
