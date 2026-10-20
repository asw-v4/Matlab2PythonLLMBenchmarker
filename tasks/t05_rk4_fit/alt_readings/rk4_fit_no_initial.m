% Ambiguity reading "no_initial": the returned states are only the N after the start, without the
% initial condition, and the fit uses their own times h, 2h, ..., N*h. All three outputs follow from that.
function [Y, c, yflat] = rk4_fit_no_initial(h, N)
  Yfull = rk4_fit(h, N);
  Y = Yfull(:, 2:end);
  t = (1:N)' * h;
  A = [ones(N,1), t, t.^2];
  c = A \ Y(1,:)';
  yflat = Y(:);
end
