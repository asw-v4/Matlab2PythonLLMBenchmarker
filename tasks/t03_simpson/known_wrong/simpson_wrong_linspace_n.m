% n points instead of n+1, but the step still taken from n intervals: np.linspace(a, b, n) with h = (b-a)/n.
function I = simpson_wrong_linspace_n(f, a, b, n)
  x = linspace(a, b, n);
  h = (b - a) / n;
  y = f(x);
  I = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
