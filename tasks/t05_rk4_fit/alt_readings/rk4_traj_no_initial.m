% Ambiguity reading B for "solve for N steps and return the states": only the N states after the start,
% without the initial condition. (Reading A is the `Y` output of rk4_fit, which includes it.)
function Y = rk4_traj_no_initial(h, N)
  Y = rk4_fit(h, N);
  Y = Y(:, 2:end);
end
