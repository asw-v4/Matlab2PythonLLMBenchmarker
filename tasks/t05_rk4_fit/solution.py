"""Golden solution: a correct translation the model never sees. Used by the tests."""
import numpy as np


def _trajectory(h, N):
    f = lambda s: np.array([s[1], -s[0]])
    Y = np.zeros((2, N + 1))
    Y[:, 0] = [1.0, 0.0]
    for i in range(N):
        s = Y[:, i]
        k1 = f(s)
        k2 = f(s + h / 2 * k1)
        k3 = f(s + h / 2 * k2)
        k4 = f(s + h * k3)
        Y[:, i + 1] = s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return Y


def rk4_fit(h, N):
    Y = _trajectory(h, N)
    t = np.arange(N + 1) * h
    A = np.column_stack([np.ones(N + 1), t, t ** 2])
    c = np.linalg.lstsq(A, Y[0], rcond=None)[0].reshape(3, 1)       # A \ y: least squares
    yflat = Y.ravel(order="F").reshape(-1, 1)                       # column by column, like Y(:)
    return Y, c, yflat


def rk4_traj(h, N):
    """Golden solution for the ambiguity variant, reading A (the initial state is included)."""
    return _trajectory(h, N)
