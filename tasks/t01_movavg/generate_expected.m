% Run in Octave:  octave --no-gui --no-window-system generate_expected.m
% Defines the fixed test inputs and writes every Octave result to expected.mat.
% Variable names:  caseNN_in_<arg>, caseNN_expected, caseNN_wrong_<name>, caseNN_reading_<name>
1;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'reference'));
addpath(fullfile(here, 'known_wrong'));
addpath(fullfile(here, 'alt_readings'));

inputs = {
  {[1 2 3 4 5 6 7 8],                     3},   % plain
  {[0.5 -1.25 3.75 2 -0.5 8 1.125],       2},   % non-integers, negatives
  {[10 20 30 40 50],                      5},   % window as long as the input
  {[4 -2 7],                              1},   % window of one
  {sin(1:12) * 10,                        4}    % even window on longer data
};

S = struct();
S.octave_version = version();
for c = 1:numel(inputs)
  x = inputs{c}{1};
  k = inputs{c}{2};
  p = sprintf('case%02d_', c);
  S.([p 'in_x'])                  = x;
  S.([p 'in_k'])                  = k;
  S.([p 'expected'])              = movavg(x, k);
  S.([p 'wrong_drop_first'])      = movavg_wrong_drop_first(x, k);
  S.([p 'wrong_drop_last'])       = movavg_wrong_drop_last(x, k);
  S.([p 'wrong_window_km1'])      = movavg_wrong_window_km1(x, k);
  S.([p 'reading_same'])          = movavg_same(x, k);
end

save('-v7', fullfile(here, 'expected.mat'), '-struct', 'S');
printf('wrote %d cases to expected.mat with Octave %s\n', numel(inputs), version());
