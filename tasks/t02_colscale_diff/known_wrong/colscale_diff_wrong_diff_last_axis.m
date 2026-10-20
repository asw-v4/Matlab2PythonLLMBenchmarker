% Scales correctly but differences along the last axis, as np.diff(B) does by default.
function [B, D] = colscale_diff_wrong_diff_last_axis(A)
  B = (A - min(A)) ./ (max(A) - min(A));
  D = diff(B, 1, 2);
end
