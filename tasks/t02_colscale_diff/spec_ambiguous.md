I need a Python version of a MATLAB function I use called `colscale_diff(A)`. It rescales the values in `A` so they run from 0 to 1, and then applies MATLAB's `diff` to the rescaled array. It returns both the rescaled array `B` and the differences `D`. I've lost the original code, sadly.

Could you write `colscale_diff(A)` in Python for me? `A` comes in as a 2-D numpy array of floats, and I'd like the two results back as a tuple `(B, D)` of numpy arrays.
