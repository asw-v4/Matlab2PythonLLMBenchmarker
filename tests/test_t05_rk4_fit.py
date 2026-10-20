import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from bench.scoring import compare, compare_outputs
from bench.taskio import load_task

TASK_DIR = Path(__file__).parent.parent / "tasks" / "t05_rk4_fit"
TASK = load_task(TASK_DIR)
MAT = loadmat(TASK_DIR / "expected.mat")
NAMES = TASK.output_names
KW = TASK.compare_kwargs()
EPS = np.finfo(float).eps


def _load():
    spec = importlib.util.spec_from_file_location("t05_solution", TASK_DIR / "solution.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SOL = _load()
CASES = [pytest.param(c, id=f"case{c.index:02d}") for c in TASK.cases]


def check(outputs, case):
    return compare_outputs(tuple(outputs[n] for n in NAMES), case.expected, **KW)


def system_matrix(h, N):
    t = np.arange(N + 1) * h
    return np.column_stack([np.ones(N + 1), t, t ** 2])


def test_task_loads_with_three_outputs_and_separate_tolerance_for_c():
    assert TASK.id == "t05_rk4_fit" and len(TASK.cases) == 5 and NAMES == ["Y", "c", "yflat"]
    assert TASK.tolerance_overrides == {"c": {"rtol": 1e-8, "atol": 1e-10}}
    assert TASK.octave_version == (TASK_DIR.parent.parent / "octave" / "VERSION").read_text().strip()


def test_inputs_have_documented_types():
    for c in TASK.cases:
        assert isinstance(c.inputs["h"], float) and isinstance(c.inputs["N"], int) and c.inputs["N"] >= 3


@pytest.mark.parametrize("case", CASES)
def test_golden_solution_passes(case):
    out = dict(zip(NAMES, SOL.rk4_fit(case.inputs["h"], case.inputs["N"])))
    v = check(out, case)
    assert v.passed, v.reason


@pytest.mark.parametrize("case", CASES)
def test_expected_shapes(case):
    N = case.inputs["N"]
    assert case.expected["Y"].shape == (2, N + 1)
    assert case.expected["c"].shape == (3, 1)
    assert case.expected["yflat"].shape == (2 * (N + 1), 1)


# ---- the key against independent routes

@pytest.mark.parametrize("case", CASES)
def test_trajectory_agrees_with_the_rk4_step_written_as_a_matrix_power_series(case):
    # For y'' = -y one RK4 step is exactly (I + hM + (hM)^2/2 + (hM)^3/6 + (hM)^4/24) applied to the state.
    h, N = case.inputs["h"], case.inputs["N"]
    M = np.array([[0.0, 1.0], [-1.0, 0.0]])
    P = sum(np.linalg.matrix_power(h * M, k) / math.factorial(k) for k in range(5))
    s = np.array([1.0, 0.0])
    cols = [s]
    for _ in range(N):
        s = P @ s
        cols.append(s)
    np.testing.assert_allclose(case.expected["Y"], np.column_stack(cols), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_trajectory_is_close_to_the_exact_solution(case):
    h, N = case.inputs["h"], case.inputs["N"]
    t = np.arange(N + 1) * h
    np.testing.assert_allclose(case.expected["Y"][0], np.cos(t), atol=2e-3)
    np.testing.assert_allclose(case.expected["Y"][1], -np.sin(t), atol=2e-3)


@pytest.mark.parametrize("case", CASES)
def test_coefficients_solve_the_least_squares_problem(case):
    # normal equations: A' (A c - y) = 0, without using any solver
    h, N = case.inputs["h"], case.inputs["N"]
    A = system_matrix(h, N)
    resid = A.T @ (A @ case.expected["c"] - case.expected["Y"][0].reshape(-1, 1))
    assert np.abs(resid).max() < 1e-12


@pytest.mark.parametrize("case", CASES)
def test_coefficients_agree_with_numpy_lstsq(case):
    A = system_matrix(case.inputs["h"], case.inputs["N"])
    c = np.linalg.lstsq(A, case.expected["Y"][0], rcond=None)[0]
    np.testing.assert_allclose(case.expected["c"].ravel(), c, rtol=1e-10, atol=1e-12)  # tighter than the task's own


@pytest.mark.parametrize("case", CASES)
def test_yflat_is_the_columns_of_Y_one_after_another(case):
    Y = case.expected["Y"]
    np.testing.assert_array_equal(case.expected["yflat"].ravel(), np.concatenate([Y[:, j] for j in range(Y.shape[1])]))
    assert not np.array_equal(case.expected["yflat"].ravel(), Y.ravel())      # numpy's default order differs


# ---- the tolerance on c is justified by the conditioning

@pytest.mark.parametrize("case", CASES)
def test_octave_condition_number_matches_numpy_and_is_small(case):
    cond_octave = MAT[f"case{case.index:02d}_check_cond"].item()
    cond_numpy = np.linalg.cond(system_matrix(case.inputs["h"], case.inputs["N"]))
    np.testing.assert_allclose(cond_octave, cond_numpy, rtol=1e-6)
    assert 10 < cond_octave < 30


def test_tolerance_on_c_has_large_headroom_even_over_the_normal_equations_error():
    worst_cond = max(MAT[f"case{c.index:02d}_check_cond"].item() for c in TASK.cases)
    smallest_coefficient = min(np.abs(c.expected["c"]).min() for c in TASK.cases)
    budget = TASK.tolerance_overrides["c"]["atol"] + TASK.tolerance_overrides["c"]["rtol"] * smallest_coefficient
    assert budget >= 1000 * worst_cond * EPS                 # a good solver: error ~ cond * eps
    assert budget >= 1000 * worst_cond ** 2 * EPS            # normal equations: error ~ cond^2 * eps


def _c_alternatives(case):
    """Other legitimate ways to get the coefficients, each a different numerical route."""
    h, N = case.inputs["h"], case.inputs["N"]
    A, y = system_matrix(h, N), case.expected["Y"][0]
    t = np.arange(N + 1) * h
    Q, R = np.linalg.qr(A)
    return {
        "lstsq": np.linalg.lstsq(A, y, rcond=None)[0],
        "normal equations": np.linalg.solve(A.T @ A, A.T @ y),
        "qr": np.linalg.solve(R, Q.T @ y),
        "pinv": np.linalg.pinv(A) @ y,
        "polyfit reversed": np.polyfit(t, y, 2)[::-1],
    }


@pytest.mark.parametrize("case", CASES)
def test_every_legitimate_way_to_solve_the_fit_passes_the_check(case):
    Y = case.expected["Y"]
    for name, c in _c_alternatives(case).items():
        out = {"Y": Y, "c": c.reshape(3, 1), "yflat": case.expected["yflat"]}
        v = check(out, case)
        assert v.passed, f"{name}: {v.reason}"


@pytest.mark.parametrize("case", CASES)
def test_legitimate_solvers_use_under_a_tenth_of_the_allowed_tolerance(case):
    tol = TASK.tolerance_overrides["c"]
    exp = case.expected["c"].ravel()
    for name, c in _c_alternatives(case).items():
        used = np.max(np.abs(c - exp) / (tol["atol"] + tol["rtol"] * np.abs(exp)))
        assert used < 0.1, f"{name} uses {used:.3g} of the tolerance"


# ---- known-wrong variants

def test_known_wrong_names_and_labels():
    assert set(TASK.cases[0].wrong) == set(TASK.meta["known_wrong"])
    assert set(TASK.meta["known_wrong"].values()) == {"permuted-output"}


BREAKS = {"ravel_rowmajor": "yflat", "coef_order_reversed": "c"}


def test_breaks_table_covers_every_variant():
    assert set(BREAKS) == set(TASK.meta["known_wrong"])


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("name", sorted(BREAKS))
def test_each_wrong_variant_breaks_only_its_own_output_on_every_case(name, case):
    w = case.wrong[name]
    assert not check(w, case).passed
    broken = {n for n in NAMES
              if not compare(w[n], case.expected[n], vector_ok=True,
                             **{**{"rtol": KW["rtol"], "atol": KW["atol"]}, **KW["overrides"].get(n, {})}).passed}
    assert broken == {BREAKS[name]}


@pytest.mark.parametrize("case", CASES)
def test_wrong_variants_are_exact_permutations_of_the_key(case):
    wrong = case.wrong["ravel_rowmajor"]["yflat"]
    np.testing.assert_array_equal(wrong, case.expected["Y"].reshape(-1, 1))          # numpy's default: row by row
    assert sorted(wrong.ravel()) == sorted(case.expected["yflat"].ravel())           # same values, different order
    assert not np.array_equal(wrong, case.expected["yflat"])
    np.testing.assert_array_equal(case.wrong["coef_order_reversed"]["c"], case.expected["c"][::-1])


@pytest.mark.parametrize("case", CASES)
def test_numpy_defaults_reproduce_the_wrong_answers(case):
    h, N = case.inputs["h"], case.inputs["N"]
    Y, c, _ = SOL.rk4_fit(h, N)
    np.testing.assert_allclose(Y.ravel().reshape(-1, 1), case.wrong["ravel_rowmajor"]["yflat"], rtol=1e-12)   # ravel() is row-major
    t = np.arange(N + 1) * h
    poly = np.polyfit(t, Y[0], 2)                                       # highest power first
    np.testing.assert_allclose(poly.reshape(-1, 1), case.wrong["coef_order_reversed"]["c"], rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("case", CASES)
def test_np_linalg_solve_cannot_be_used_for_the_fit(case):
    A = system_matrix(case.inputs["h"], case.inputs["N"])
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.solve(A, case.expected["Y"][0])


# ---- ambiguity variant

@pytest.mark.parametrize("case", CASES)
def test_no_initial_reading_is_the_key_without_its_first_column(case):
    reading = case.readings["no_initial"]
    assert set(reading) == {"Y"}
    N = case.inputs["N"]
    assert reading["Y"].shape == (2, N)
    np.testing.assert_array_equal(reading["Y"], case.expected["Y"][:, 1:])
    assert not compare(reading["Y"], case.expected["Y"], rtol=1e-12, atol=1e-12).passed   # different shape
    assert compare(SOL.rk4_traj(case.inputs["h"], N), case.expected["Y"], rtol=1e-12, atol=1e-12).passed


def test_ambiguity_function_is_declared():
    assert TASK.meta["ambiguity"] == {"function": "rk4_traj", "outputs": ["Y"]}


# ---- specs

def test_spec_has_no_hints_in_the_code():
    spec = (TASK_DIR / "spec.md").read_text()
    code = spec.split("```matlab")[1].split("```")[0]
    assert "%" not in code, "comments in the MATLAB would give the traps away"
    for piece in ("A \\ Y(1,:)'", "yflat = Y(:);", "Y(:,i+1)", "t = (0:N)' * h;"):
        assert piece in code
    assert "least squares" not in spec.lower() and "column-major" not in spec.lower()


def test_ambiguous_spec_leaves_the_initial_state_open_and_gives_nothing_away():
    amb = (TASK_DIR / "spec_ambiguous.md").read_text().lower()
    assert "rk4_traj(h, n)" in amb and "for `n` steps" in amb
    for banned in ("initial", "include", "n+1", "n + 1", "excluding", "without the", "polyfit", "flatten",
                   "least", "coefficient", "```"):
        assert banned not in amb, banned
    for text in (amb, (TASK_DIR / "spec.md").read_text().lower()):
        assert "expected" not in text
