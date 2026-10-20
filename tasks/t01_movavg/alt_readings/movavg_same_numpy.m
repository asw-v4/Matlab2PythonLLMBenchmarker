% Ambiguity reading B', numpy's alignment of the same-length reading: np.convolve(x, ones(k)/k, mode='same').
% It equals movavg_same for odd k. For even k it starts half a step earlier (floor((k-1)/2) instead of
% ceil((k-1)/2)), because "centred" has no single meaning for an even window.
function y = movavg_same_numpy(x, k)
  full = conv(x, ones(1, k) / k);
  s = floor((k - 1) / 2);
  y = full(s + 1 : s + numel(x));
end
