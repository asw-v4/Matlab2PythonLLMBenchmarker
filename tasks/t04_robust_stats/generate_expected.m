% Run in Octave:  octave --no-gui --no-window-system generate_expected.m
% Defines the fixed test inputs and writes every Octave result to expected.mat.
% Outputs: s, v, r, m, rm, sat.  The ambiguity reading covers only `s`.
1;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'reference'));
addpath(fullfile(here, 'known_wrong'));
addpath(fullfile(here, 'alt_readings'));

%        x                                   a                          b                       u                                w
inputs = {
  {[2.5 -2.5 0.5 1.5 -0.5 3.7 -1.2],  [-7 7 -7 7 0 5],        [3 -3 3 -3 4 -2],     uint8([250 100 0 128 255 200]),  uint8([10 200 5 127 1 55])},   % ties, mixed signs, saturating sums
  {[1.1 2.2 3.3 4.4],                 [7 9 10 12],            [2 4 3 5],            uint8([1 2 3 4]),                uint8([5 6 7 8])},             % nothing unusual: only n-1 vs n differs
  {[-0.5 -1.5 -2.5 -3.5],             [-9 -8 -7],             [-4 -3 -2],           uint8([10 20 30]),               uint8([1 2 3])},               % ties; a and b both negative
  {[10.5 -4.5],                       [100 -100],             [7 7],                uint8([255 255]),                uint8([255 0])},               % n = 2; full-scale uint8
  {[0.125 -0.375 0.875 2.0625 -1.9375 0.3], [-13 -6 -1 4 11 17], [5 -5 4 -4 3 -3],  uint8([1 2 3 4 5 6]),            uint8([250 251 252 253 254 255])}  % no ties; sums cross 255 at different points
};

wrongs = {'ddof0', 'round_half_even', 'rem_as_mod', 'mod_as_rem', 'wraparound'};
outs = {'s', 'v', 'r', 'm', 'rm', 'sat'};

S = struct();
S.octave_version = version();
for c = 1:numel(inputs)
  x = inputs{c}{1}; a = inputs{c}{2}; b = inputs{c}{3}; u = inputs{c}{4}; w = inputs{c}{5};
  p = sprintf('case%02d_', c);
  S.([p 'in_x']) = x;  S.([p 'in_a']) = a;  S.([p 'in_b']) = b;  S.([p 'in_u']) = u;  S.([p 'in_w']) = w;

  [r1{1:6}] = robust_stats(x, a, b, u, w);
  for k = 1:6, S.([p 'expected__' outs{k}]) = r1{k}; end

  for q = 1:numel(wrongs)
    [r2{1:6}] = feval(['robust_stats_wrong_' wrongs{q}], x, a, b, u, w);
    for k = 1:6, S.([p 'wrong_' wrongs{q} '__' outs{k}]) = r2{k}; end
  end

  S.([p 'reading_population__s']) = spread_population(x);
end

save('-v7', fullfile(here, 'expected.mat'), '-struct', 'S');
printf('wrote %d cases to expected.mat with Octave %s\n', numel(inputs), version());
