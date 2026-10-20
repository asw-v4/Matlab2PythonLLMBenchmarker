% Halves round to the nearest even integer, as np.round and Python's round do; everything else is correct.
function [s, v, r, m, rm, sat] = robust_stats_wrong_round_half_even(x, a, b, u, w)
  [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w);
  r = round(x);
  tie = abs(x - fix(x)) == 0.5;
  r(tie) = 2 * round(x(tie) / 2);
end
