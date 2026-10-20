% Coefficients highest power first, as np.polyfit returns them; everything else is correct.
function [Y, c, yflat] = rk4_fit_wrong_coef_order_reversed(h, N)
  [Y, c, yflat] = rk4_fit(h, N);
  c = flipud(c);
end
