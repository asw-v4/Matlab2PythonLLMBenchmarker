Translate this MATLAB function to Python.

```matlab
function y = movavg(x, k)
  n = length(x);
  y = zeros(1, n-k+1);
  for i = 1:(n-k+1)
    y(i) = mean(x(i:i+k-1));
  end
end
```

Write a Python function `movavg(x, k)`. `x` is a 2-D numpy array of shape `(1, n)` and `k` is an int with `1 <= k <= n`. Return the result as a numpy array.
