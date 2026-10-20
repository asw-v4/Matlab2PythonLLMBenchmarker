% The last point is missing, as np.arange(a, b, h) gives (it stops before b).
function I = simpson_wrong_no_endpoint(f, a, b, n)
  h = (b - a) / n;
  x = a + (0:n-1) * h;
  y = f(x);
  I = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
