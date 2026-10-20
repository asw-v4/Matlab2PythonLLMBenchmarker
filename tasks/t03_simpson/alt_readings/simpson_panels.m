% Ambiguity reading B: n panels, each a parabola over two subintervals, so 2n subintervals.
function I = simpson_panels(f, a, b, n)
  I = simpson(f, a, b, 2 * n);
end
