% Always works down axis 0, as a port using axis=0 for every input would.
% Right for matrices and columns; for a row vector each element is its own min and max (0/0).
function [B, D] = colscale_diff_wrong_axis0(A)
  B = (A - min(A, [], 1)) ./ (max(A, [], 1) - min(A, [], 1));
  D = diff(B, 1, 1);
end
