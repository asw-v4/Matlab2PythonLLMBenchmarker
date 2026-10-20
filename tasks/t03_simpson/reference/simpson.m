function I = simpson(f, a, b, n)
  x = linspace(a, b, n+1);
  h = (b - a) / n;
  y = f(x);
  I = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
