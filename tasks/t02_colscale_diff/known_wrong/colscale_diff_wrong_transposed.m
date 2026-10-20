% Right numbers, both outputs transposed.
function [B, D] = colscale_diff_wrong_transposed(A)
  B = (A - min(A)) ./ (max(A) - min(A));
  D = diff(B);
  B = B.';
  D = D.';
end
