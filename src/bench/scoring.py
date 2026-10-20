"""Compare a model's output with the Octave answer key."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str


def compare(actual, expected, *, rtol: float, atol: float, vector_ok: bool = False,
            equal_nan: bool = False) -> Verdict:
    """Pass if `actual` matches `expected` within tolerance.

    With `vector_ok`, a 1-D `actual` is accepted where `expected` is a 1xM row or Mx1 column of
    the same length. A transposed 2-D result is never accepted.
    """
    try:
        actual = np.asarray(actual)
        expected = np.asarray(expected)
        if actual.dtype == object or actual.dtype.kind not in "biufc":
            return Verdict(False, "not-numeric")
    except Exception:
        return Verdict(False, "not-numeric")

    # A single number is a single number: float, 0-d, (1,) and (1, 1) are all accepted for a 1x1 key.
    if actual.shape != expected.shape and actual.size == 1 and expected.size == 1:
        actual = actual.reshape(expected.shape)

    if actual.shape != expected.shape:
        is_vector = expected.ndim == 2 and 1 in expected.shape
        if vector_ok and is_vector and actual.ndim == 1 and actual.size == expected.size:
            actual = actual.reshape(expected.shape)
        else:
            return Verdict(False, f"shape-mismatch: got {actual.shape}, expected {expected.shape}")

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan=equal_nan)
    except AssertionError:
        return Verdict(False, "value-mismatch")
    return Verdict(True, "pass")


def compare_outputs(actual, expected: dict, *, names: list[str], rtol: float, atol: float,
                    vector_ok: bool = False, exact: tuple | list | set = (),
                    overrides: dict | None = None, equal_nan: bool = False) -> Verdict:
    """Compare a function's whole return value with the key, one named output at a time.

    A single-output task takes the array itself (or a 1-tuple). A multi-output task takes a tuple or
    list with one entry per name, in order. The first mismatching output is reported, prefixed by name.
    Outputs named in `exact` are compared with zero tolerance instead of rtol and atol, and an output in
    `overrides` ({name: {"rtol": ..., "atol": ...}}) uses its own tolerance. NaN never equals NaN unless
    `equal_nan` is set, which is only for recognising a known-wrong answer that itself contains NaN.
    """
    overrides = overrides or {}
    if len(names) == 1:
        if isinstance(actual, tuple) and len(actual) == 1:
            actual = actual[0]
        parts = {names[0]: actual}
    else:
        if not isinstance(actual, (tuple, list)) or len(actual) != len(names):
            return Verdict(False, f"wrong-number-of-outputs: expected {len(names)}")
        parts = dict(zip(names, actual))
    for name in names:
        tol = dict(rtol=rtol, atol=atol)
        tol.update(overrides.get(name, {}))
        if name in exact:
            tol = dict(rtol=0.0, atol=0.0)
        v = compare(parts[name], expected[name], vector_ok=vector_ok, equal_nan=equal_nan, **tol)
        if not v.passed:
            return Verdict(False, f"{name}: {v.reason}")
    return Verdict(True, "pass")
