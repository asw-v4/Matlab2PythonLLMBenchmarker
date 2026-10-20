% Loop stops one too early, as a port with range(n-k) would.
function y = movavg_wrong_drop_last(x, k)
  n = length(x);
  y = zeros(1, n-k);
  for i = 1:(n-k)
    y(i) = mean(x(i:i+k-1));
  end
end
