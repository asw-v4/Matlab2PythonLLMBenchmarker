% Flattens row by row, as numpy's default ravel() does; everything else is correct.
function [Y, c, yflat] = rk4_fit_wrong_ravel_rowmajor(h, N)
  [Y, c, yflat] = rk4_fit(h, N);
  yflat = reshape(Y.', [], 1);
end
