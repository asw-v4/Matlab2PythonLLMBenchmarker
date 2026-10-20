% Run in Octave:  octave --no-gui --no-window-system generate_expected.m
% Defines the fixed test inputs and writes every Octave result to expected.mat.
% Variable names:  caseNN_in_<arg>, caseNN_expected__<out>, caseNN_wrong_<name>__<out>,
%                  caseNN_reading_<name>__<out>      (outputs: B, D)
1;

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'reference'));
addpath(fullfile(here, 'known_wrong'));
addpath(fullfile(here, 'alt_readings'));

% No constant columns, so no 0/0 in the reference.
inputs = {
  [3 -1 7.5; 9 4 0.25; 1 2.5 3; 6 8 -2],           % 4x3 matrix (tall)
  [2 9 4 1 7; 5 3 8 6 0.5; 1 7 2 5 3],             % 3x5 matrix (wide)
  [3 1 4 1.5 5],                                    % 1x5 row vector
  [3; 1; 4; 1.5; 5],                                % 5x1 column vector
  [1 2; 4 6]                                        % 2x2 matrix
};

wrongs = {'axis0', 'diff_last_axis', 'transposed'};
readings = {'global'};

S = struct();
S.octave_version = version();
for c = 1:numel(inputs)
  A = inputs{c};
  p = sprintf('case%02d_', c);
  S.([p 'in_A']) = A;

  [B, D] = colscale_diff(A);
  S.([p 'expected__B']) = B;
  S.([p 'expected__D']) = D;

  for w = 1:numel(wrongs)
    [B, D] = feval(['colscale_diff_wrong_' wrongs{w}], A);
    S.([p 'wrong_' wrongs{w} '__B']) = B;
    S.([p 'wrong_' wrongs{w} '__D']) = D;
  end

  for r = 1:numel(readings)
    [B, D] = feval(['colscale_diff_' readings{r}], A);
    S.([p 'reading_' readings{r} '__B']) = B;
    S.([p 'reading_' readings{r} '__D']) = D;
  end
end

save('-v7', fullfile(here, 'expected.mat'), '-struct', 'S');
printf('wrote %d cases to expected.mat with Octave %s\n', numel(inputs), version());
