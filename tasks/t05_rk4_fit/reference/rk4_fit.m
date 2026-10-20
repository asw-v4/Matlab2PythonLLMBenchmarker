function [Y, c, yflat] = rk4_fit(h, N)
  f = @(s) [s(2); -s(1)];
  Y = zeros(2, N+1);
  Y(:,1) = [1; 0];
  for i = 1:N
    s  = Y(:,i);
    k1 = f(s);
    k2 = f(s + h/2*k1);
    k3 = f(s + h/2*k2);
    k4 = f(s + h*k3);
    Y(:,i+1) = s + h/6*(k1 + 2*k2 + 2*k3 + k4);
  end
  t = (0:N)' * h;
  A = [ones(N+1,1), t, t.^2];
  c = A \ Y(1,:)';
  yflat = Y(:);
end
