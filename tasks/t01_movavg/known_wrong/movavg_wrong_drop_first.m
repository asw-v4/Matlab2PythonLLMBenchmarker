% Loop starts one too late, as a port with range(1, n-k+1) would.
function y = movavg_wrong_drop_first(x, k)
  n = length(x);
  y = zeros(1, n-k);
  for i = 2:(n-k+1)
    y(i-1) = mean(x(i:i+k-1));
  end
end
