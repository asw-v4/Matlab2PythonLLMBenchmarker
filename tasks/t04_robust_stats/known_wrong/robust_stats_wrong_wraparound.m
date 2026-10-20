% uint8 addition wraps around at 256, as numpy does; everything else is correct.
function [s, v, r, m, rm, sat] = robust_stats_wrong_wraparound(x, a, b, u, w)
  [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w);
  sat = uint8(mod(double(u) + double(w), 256));
end
