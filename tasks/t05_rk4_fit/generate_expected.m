% Run in Octave:  octave --no-gui --no-window-system generate_expected.m
% Defines the fixed test inputs and writes every Octave result to expected.mat.
% Outputs: Y, c, yflat.  The ambiguity reading covers only `Y`.
% caseNN_check_cond is the condition number of the least-squares matrix, used to justify the tolerance on c.
1;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'reference'));
addpath(fullfile(here, 'known_wrong'));
addpath(fullfile(here, 'alt_readings'));

%        h     N
cases = {
  {0.1,   10},    % the usual one
  {0.05,  20},
  {0.2,   5},
  {0.25,  8},     % h exact in binary, so the time grid has no rounding
  {0.3,   3}      % fewest points: 4 rows for 3 unknowns
};

wrongs = {'ravel_rowmajor', 'coef_order_reversed'};
outs = {'Y', 'c', 'yflat'};

S = struct();
S.octave_version = version();
for k = 1:numel(cases)
  h = cases{k}{1}; N = cases{k}{2};
  p = sprintf('case%02d_', k);
  S.([p 'in_h']) = h;
  S.([p 'in_N']) = N;

  [r1{1:3}] = rk4_fit(h, N);
  for j = 1:3, S.([p 'expected__' outs{j}]) = r1{j}; end

  t = (0:N)' * h;
  S.([p 'check_cond']) = cond([ones(N+1,1), t, t.^2]);

  for q = 1:numel(wrongs)
    [r2{1:3}] = feval(['rk4_fit_wrong_' wrongs{q}], h, N);
    for j = 1:3, S.([p 'wrong_' wrongs{q} '__' outs{j}]) = r2{j}; end
  end

  S.([p 'reading_no_initial__Y']) = rk4_traj_no_initial(h, N);
end

save('-v7', fullfile(here, 'expected.mat'), '-struct', 'S');
printf('wrote %d cases to expected.mat with Octave %s\n', numel(cases), version());
for k = 1:numel(cases)
  printf('case %d: cond(A) = %.4g\n', k, S.(sprintf('case%02d_check_cond', k)));
end
