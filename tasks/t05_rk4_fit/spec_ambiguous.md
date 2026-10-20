I need a Python version of a MATLAB function I use called `rk4_traj(h, N)`. It solves y'' = -y, starting from y = 1 and y' = 0, with the classic fourth-order Runge-Kutta method using a fixed step size `h`, for `N` steps. It returns the states as a 2-row numpy array with one column per time step, where the first row is y and the second row is y'. I've lost the original code, sadly.

Could you write `rk4_traj(h, N)` in Python for me? `h` is a float, `N` is an int, and I'd like the array back.
