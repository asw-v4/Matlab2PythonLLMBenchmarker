"""Golden solution: a correct translation the model never sees. Used by the tests."""
import numpy as np


def simpson(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    h = (b - a) / n
    y = np.asarray(f(x), dtype=float)
    return float(h / 3 * (y[0] + 4 * y[1:-1:2].sum() + 2 * y[2:-2:2].sum() + y[-1]))
