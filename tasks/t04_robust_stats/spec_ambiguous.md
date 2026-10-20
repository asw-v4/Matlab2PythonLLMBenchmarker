I need a Python version of a MATLAB function I use called `robust_stats(x, a, b, u, w)`. It returns six things: the standard deviation of `x`, the variance of `x`, `x` rounded to whole numbers, `a` modulo `b`, the remainder of `a` divided by `b`, and `u` plus `w`. I've lost the original code, sadly.

Could you write `robust_stats(x, a, b, u, w)` in Python for me? `x`, `a` and `b` are 2-D numpy arrays of floats (`a` and `b` hold whole numbers, and `b` has no zeros), and `u` and `w` are 2-D numpy arrays of dtype uint8. I'd like the six results back as a tuple `(s, v, r, m, rm, sat)`.
