% mod translated as np.fmod, which follows the sign of the dividend; everything else is correct.
function [s, v, r, m, rm, sat] = robust_stats_wrong_mod_as_rem(x, a, b, u, w)
  [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w);
  m = rem(a, b);
end
