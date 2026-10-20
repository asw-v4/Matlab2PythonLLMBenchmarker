Translate this MATLAB function to Python.

```matlab
function [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w)
  s   = std(x);
  v   = var(x);
  r   = round(x);
  m   = mod(a, b);
  rm  = rem(a, b);
  sat = u + w;
end
```

Write a Python function `robust_stats(x, a, b, u, w)`. `x`, `a` and `b` are 2-D numpy arrays of floats (`a` and `b` hold whole numbers, and `b` has no zeros), and `u` and `w` are 2-D numpy arrays of dtype uint8. Return the six results as a tuple `(s, v, r, m, rm, sat)`, in that order.
