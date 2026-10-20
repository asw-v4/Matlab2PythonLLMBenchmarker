% Ambiguity reading B: output keeps the input length, zero-padded at the edges.
function y = movavg_same(x, k)
  y = conv(x, ones(1, k) / k, 'same');
end
