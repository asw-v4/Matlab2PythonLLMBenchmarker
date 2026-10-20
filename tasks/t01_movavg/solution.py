"""Golden solution: a correct translation the model never sees. Used by the tests."""
import numpy as np


def movavg(x, k):
    x = np.asarray(x, dtype=float)
    n = x.shape[1]
    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])
