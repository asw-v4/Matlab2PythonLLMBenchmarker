% Run in Octave:  octave --no-gui --no-window-system generate_expected.m
% Defines the fixed test inputs and writes every Octave result to expected.mat.
% The function to integrate is stored by name (caseNN_in_f); functions.py has the Python versions.
% caseNN_check_x / caseNN_check_f hold f evaluated on the grid, so the tests can confirm the Python
% and Octave definitions agree.
1;

function f = test_function(name)
  switch name
    case 'cubic',   f = @(x) x.^3;
    case 'gauss',   f = @(x) exp(-x.^2);
    case 'sine',    f = @(x) sin(x);
    case 'quartic', f = @(x) x.^4;
    case 'lorentz', f = @(x) 1 ./ (1 + x.^2);
    otherwise,      error('unknown test function: %s', name);
  end
end

here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'reference'));
addpath(fullfile(here, 'known_wrong'));
addpath(fullfile(here, 'alt_readings'));

%        function   a    b     n
cases = {
  {'cubic',   0,   2,      2},    % one parabola; the interior sums are empty or a single point
  {'gauss',   0,   2,     10},
  {'sine',    0,   pi,      8},   % non-terminating step, so floating-point endpoints matter
  {'quartic', -1,  3,      6},    % asymmetric interval, not exact for Simpson
  {'lorentz', 0,   1,      4}
};

wrongs = {'no_endpoint', 'linspace_n', 'weights_swapped'};
readings = {'panels'};

S = struct();
S.octave_version = version();
for c = 1:numel(cases)
  name = cases{c}{1}; a = cases{c}{2}; b = cases{c}{3}; n = cases{c}{4};
  f = test_function(name);
  p = sprintf('case%02d_', c);
  S.([p 'in_f']) = name;
  S.([p 'in_a']) = a;
  S.([p 'in_b']) = b;
  S.([p 'in_n']) = n;
  xg = linspace(a, b, n+1);
  S.([p 'check_x']) = xg;
  S.([p 'check_f']) = f(xg);
  S.([p 'expected']) = simpson(f, a, b, n);
  for w = 1:numel(wrongs)
    S.([p 'wrong_' wrongs{w}]) = feval(['simpson_wrong_' wrongs{w}], f, a, b, n);
  end
  for r = 1:numel(readings)
    S.([p 'reading_' readings{r}]) = feval(['simpson_' readings{r}], f, a, b, n);
  end
end

save('-v7', fullfile(here, 'expected.mat'), '-struct', 'S');
printf('wrote %d cases to expected.mat with Octave %s\n', numel(cases), version());
