% Ambiguity reading B for "the standard deviation of x": divide by n instead of n-1.
% (Reading A is the `s` output of robust_stats.)
function s = spread_population(x)
  s = std(x, 1);
end
