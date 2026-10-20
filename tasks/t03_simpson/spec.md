Translate this MATLAB function to Python.

```matlab
function I = simpson(f, a, b, n)
  x = linspace(a, b, n+1);
  h = (b - a) / n;
  y = f(x);
  I = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
```

Write a Python function `simpson(f, a, b, n)`. `f` is a Python function that takes a numpy array and returns an array of the same shape, `a` and `b` are floats, and `n` is an even positive int. Return the result as a float.
