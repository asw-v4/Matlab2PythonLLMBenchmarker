% std and var divide by n, as numpy does by default; everything else is correct.
function [s, v, r, m, rm, sat] = robust_stats_wrong_ddof0(x, a, b, u, w)
  [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w);
  s = std(x, 1);
  v = var(x, 1);
end
