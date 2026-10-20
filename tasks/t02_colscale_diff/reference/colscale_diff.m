function [B, D] = colscale_diff(A)
  B = (A - min(A)) ./ (max(A) - min(A));
  D = diff(B);
end
