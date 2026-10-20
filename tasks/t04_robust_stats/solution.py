"""Golden solution: a correct translation the model never sees. Used by the tests."""
import numpy as np


def robust_stats(x, a, b, u, w):
    x = np.asarray(x, dtype=float)
    s = float(np.std(x, ddof=1))
    v = float(np.var(x, ddof=1))
    r = np.sign(x) * np.floor(np.abs(x) + 0.5)          # halves go away from zero
    m = np.mod(a, b)                                    # sign of the divisor
    rm = np.fmod(a, b)                                  # sign of the dividend
    sat = np.minimum(u.astype(np.int64) + w.astype(np.int64), 255).astype(np.uint8)
    return s, v, r, m, rm, sat


def spread(x):
    """Golden solution for the ambiguity variant, reading A (sample standard deviation)."""
    return float(np.std(np.asarray(x, dtype=float), ddof=1))
