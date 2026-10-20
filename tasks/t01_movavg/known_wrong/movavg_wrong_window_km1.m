% Window one element short, as x[i:i+k-1] would give when "i+k-1" is carried over literally.
function y = movavg_wrong_window_km1(x, k)
  n = length(x);
  y = zeros(1, n-k+1);
  for i = 1:(n-k+1)
    y(i) = mean(x(i:i+k-2));
  end
end
