function [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w)
  s   = std(x);
  v   = var(x);
  r   = round(x);
  m   = mod(a, b);
  rm  = rem(a, b);
  sat = u + w;
end
