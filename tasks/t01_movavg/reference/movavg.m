function y = movavg(x, k)
  n = length(x);
  y = zeros(1, n-k+1);
  for i = 1:(n-k+1)
    y(i) = mean(x(i:i+k-1));
  end
end
