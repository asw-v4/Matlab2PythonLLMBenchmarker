% Ambiguity reading B: scale the whole array to [0, 1] using one overall min and max.
function [B, D] = colscale_diff_global(A)
  lo = min(A(:));
  hi = max(A(:));
  B = (A - lo) ./ (hi - lo);
  D = diff(B);
end
