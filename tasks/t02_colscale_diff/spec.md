Translate this MATLAB function to Python.

```matlab
function [B, D] = colscale_diff(A)
  B = (A - min(A)) ./ (max(A) - min(A));
  D = diff(B);
end
```

Write a Python function `colscale_diff(A)`. `A` is a 2-D numpy array of floats, which may have a single row or a single column. Return the two results as a tuple `(B, D)` of numpy arrays.
