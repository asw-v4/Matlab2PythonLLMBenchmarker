"""Name a failure by what its output looks like.

These are output signatures, not diagnoses: an output equal to the key shifted by one is called
`shifted-output`, which is evidence, not proof, of an indexing error. Root causes would need the code read.

Order of checks for a failing case:
  1. it did not run: timeout, runtime-error, bad-output, wrong-output-count
  2. the whole return value equals one of the task's known-wrong translations -> that variant's label
  3. generic output patterns on the first output that fails: nan-output, transposed-output,
     shifted-output, shape-mismatch, permuted-output, scale-mismatch, value-mismatch
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from bench.runs import ERROR, OK, TIMEOUT, BAD_OUTPUT, CaseRun
from bench.scoring import compare_outputs

PASS = "pass"
STRUCTURAL = dict(rtol=1e-9, atol=1e-12)    # for recognising a pattern, looser than the pass tolerance


@dataclass(frozen=True)
class CaseResult:
    passed: bool
    signature: str          # "pass" or a label
    detail: str


@dataclass(frozen=True)
class SampleResult:
    passed: bool
    signature: str          # "pass", or the commonest failing label (ties: alphabetical)
    cases: tuple[CaseResult, ...]

    @property
    def n_failed(self) -> int:
        return sum(not c.passed for c in self.cases)


def _close(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and bool(np.allclose(a, b, equal_nan=False, **STRUCTURAL))


def generic_signature(actual, expected) -> tuple[str, str]:
    """Label one wrong output (an array that should have matched `expected`)."""
    a, e = np.asarray(actual), np.asarray(expected)
    if a.dtype == object or a.dtype.kind not in "biufc":
        return "bad-output", "not numeric"
    a, e = a.astype(float), e.astype(float)
    if np.isnan(a).any() and not np.isnan(e).any():
        return "nan-output", "contains NaN"

    if a.shape != e.shape:
        if a.ndim == 2 and e.ndim == 2 and _close(a.T, e):
            return "transposed-output", f"equals the key transposed {e.shape}"
        if a.size == e.size and np.allclose(a.ravel(), e.ravel(), **STRUCTURAL):
            return "shape-mismatch", f"right values, shape {a.shape} instead of {e.shape}"
        fa, fe = a.ravel(), e.ravel()
        if 0 < fa.size < fe.size and fe.size - fa.size <= 2:
            for start in range(fe.size - fa.size + 1):
                if np.allclose(fa, fe[start:start + fa.size], **STRUCTURAL):
                    return "shifted-output", f"equals the key shifted by {start}, {fe.size - fa.size} value(s) short"
        return "shape-mismatch", f"shape {a.shape} instead of {e.shape}"

    if np.allclose(np.sort(a.ravel()), np.sort(e.ravel()), **STRUCTURAL):
        return "permuted-output", "same values in a different order"
    nonzero = e != 0
    if nonzero.any() and np.isfinite(a).all() and np.allclose(a[~nonzero], 0.0, atol=STRUCTURAL["atol"]):
        ratios = a[nonzero] / e[nonzero]
        if np.allclose(ratios, ratios[0], rtol=1e-6) and not np.isclose(ratios[0], 1.0, rtol=1e-9):
            return "scale-mismatch", f"key times {ratios[0]:.6g}"
    return "value-mismatch", "different values"


def classify_case(run: CaseRun, case, task) -> CaseResult:
    if run.status == TIMEOUT:
        return CaseResult(False, "timeout", run.error or "")
    if run.status == ERROR:
        return CaseResult(False, "runtime-error", run.error or "")
    if run.status == BAD_OUTPUT:
        return CaseResult(False, "bad-output", run.error or "")
    assert run.status == OK, run.status

    kw = task.compare_kwargs()
    actual = run.as_result()
    verdict = compare_outputs(actual, case.expected, **kw)
    if verdict.passed:
        return CaseResult(True, PASS, "")
    if verdict.reason.startswith("wrong-number-of-outputs"):
        return CaseResult(False, "wrong-output-count", verdict.reason)

    labels = task.meta.get("known_wrong", {})
    # NaN equals NaN here: a model that reproduces a known-wrong translation reproduces its NaNs too
    matched = sorted(name for name in labels
                     if compare_outputs(actual, case.wrong[name], equal_nan=True, **kw).passed)
    if matched:
        label = "+".join(sorted({labels[n] for n in matched}))
        return CaseResult(False, label, f"matches known-wrong translation(s): {', '.join(matched)}")

    names = kw["names"]
    failing = verdict.reason.split(":")[0] if len(names) > 1 else names[0]
    idx = names.index(failing) if failing in names else 0
    actual_out = actual[idx] if len(names) > 1 else actual[0]
    label, detail = generic_signature(actual_out, case.expected[failing])
    return CaseResult(False, label, f"{failing}: {detail}")


def classify_sample(runs: list[CaseRun], task) -> SampleResult:
    """One result per test case, and a sample-level signature."""
    if len(runs) != len(task.cases):
        raise ValueError(f"expected {len(task.cases)} case runs, got {len(runs)}")
    results = tuple(classify_case(r, c, task) for r, c in zip(runs, task.cases))
    failing = [r.signature for r in results if not r.passed]
    if not failing:
        return SampleResult(True, PASS, results)
    counts = Counter(failing)
    top = max(counts.values())
    return SampleResult(False, sorted(s for s, n in counts.items() if n == top)[0], results)
