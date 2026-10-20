"""Golden solution: a correct translation the model never sees. Used by the tests."""
import numpy as np


def colscale_diff(A):
    A = np.asarray(A, dtype=float)
    # MATLAB's min, max and diff work along the first dimension whose size is not 1.
    axis = 0 if A.shape[0] > 1 else 1
    lo = A.min(axis=axis, keepdims=True)
    hi = A.max(axis=axis, keepdims=True)
    B = (A - lo) / (hi - lo)
    D = np.diff(B, axis=axis)
    return B, D
