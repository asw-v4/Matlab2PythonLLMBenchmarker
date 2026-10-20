% rem translated as Python's %, which follows the sign of the divisor; everything else is correct.
function [s, v, r, m, rm, sat] = robust_stats_wrong_rem_as_mod(x, a, b, u, w)
  [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w);
  rm = mod(a, b);
end
