# The five pilot tasks

Each task is a small MATLAB function. The model translates it to Python, and I check the result against what Octave gives for the original. Each one is built around a place where MATLAB and Python quietly disagree, because that is where a translation can look right and still be wrong.

**Status:** draft. The MATLAB below is a sketch of each reference function. The real ones will be written and run in Octave in Stage 6, and anything that changes there gets updated here.

| # | Function | What it's testing | Main trap | Failure signature |
|---|----------|-------------------|-----------|-------------------|
| 1 | `movavg` | Indexing | 1-based loop, end-inclusive slices | `shifted-output`, `window-mismatch` |
| 2 | `colscale_diff` | Array orientation | Default dimension of `min` and `diff` | `nan-output`, `shape-mismatch`, `transposed-output` |
| 3 | `simpson` | Integration bounds | Whether the endpoint is included | `endpoint-mismatch`, `weights-mismatch` |
| 4 | `robust_stats` | Silent numerical drift | Defaults that differ between the two languages | `scale-mismatch`, `rounding-mismatch`, `sign-mismatch`, `wraparound-mismatch` |
| 5 | `rk4_fit` | Everything at once | Column-major flatten, `\` as least squares | `permuted-output`, `runtime-error` |

## How a task is checked
- **Inputs.** The model gets the MATLAB source and a short description, but not the test inputs. MATLAB has no 1-D arrays, so array inputs are always passed as 2-D numpy arrays. That keeps row vs column unambiguous.
- **Pass.** The output matches Octave's within a stated tolerance (`numpy.testing.assert_allclose`). Tolerances are tight where only floating-point summation order can differ, and exact for integer and rounding outputs.
- **Failure signature.** For each trap I also have Octave compute what the common wrong translation would give. If the model's output matches that wrong answer, the failure gets that signature. A signature is evidence of what the output looks like, not a diagnosis of the cause.
- **Golden solution.** Each task has a correct numpy translation that the model never sees. Tests check that it passes and that every known-wrong variant fails, which shows the tolerances are neither too strict nor too loose.
- **Ambiguity variant.** Each task has a second version of the spec with one thing deliberately left open. Because MATLAB source usually settles the question, these versions describe the function in words instead of showing code. They are scored on what the model does about the gap, not just whether it passes (see the end of this file).

---

## Task 1: `movavg` (moving average)

**What it does.** Takes a row vector `x` and a window size `k`, and returns the average of every run of `k` consecutive values, so the output has `n-k+1` entries.

```matlab
function y = movavg(x, k)
  n = length(x);
  y = zeros(1, n-k+1);
  for i = 1:(n-k+1)
    y(i) = mean(x(i:i+k-1));
  end
end
```

**Test inputs.** Three row vectors with different lengths and window sizes.

**What could go wrong**
- MATLAB's `x(i:i+k-1)` includes both ends. The Python equivalent is `x[i:i+k]`, where the end is excluded. Translating the arithmetic literally gives a window of `k-1`.
- The loop `1:(n-k+1)` becomes `range(n-k+1)` in Python. A model that keeps the MATLAB bounds, or starts at 1, drops or shifts values.
- A loop that starts a step late or stops a step early drops the first or last average, so the output is one value short. That is the `shifted-output` signature. A window of `k-1` keeps the length but changes every value, which is `window-mismatch`.

**Ambiguity variant.** The spec describes the function in words ("the moving average of a row vector using a window of `k` consecutive values") and doesn't say whether the output keeps length `n` (zero-padded at the edges) or shortens to `n-k+1`. The MATLAB source would settle the question, so this version leaves it out. The two readings are checked against Octave's `loop` result and `conv(x, ones(1,k)/k, 'same')`. They coincide when `k = 1`, so the variant is only scored on cases with `k > 1`.

---

## Task 2: `colscale_diff` (column scaling and differencing)

**What it does.** Rescales each column of `A` to the range 0 to 1, then takes differences between consecutive rows.

```matlab
function [B, D] = colscale_diff(A)
  B = (A - min(A)) ./ (max(A) - min(A));
  D = diff(B);
end
```

**Test inputs.** Five cases: a 4x3 matrix, a wide 3x5 matrix, a 1x5 row vector, a 5x1 column vector, and a 2x2 matrix. None has a constant column, so there's no 0/0. The model returns a tuple `(B, D)`, and a 1-D result is not accepted in place of a row or column here, because orientation is the point of the task.

**What could go wrong**
- MATLAB's `min`, `max` and `diff` work along the first dimension that isn't size 1. For a matrix that's down the columns. For a 1x5 row it's along the row, so `min(A)` is a single number and the whole vector is scaled together.
- numpy's `np.diff` defaults to the last axis, not the first. On the 4x3 matrix that differences across each row instead of down it.
- Forcing `axis=0` everywhere fixes the matrix but breaks the row vector: each element becomes its own minimum and maximum, and the result is `0/0 = NaN`.
- Where each wrong translation actually fails (checked against Octave): `axis=0` everywhere is right for matrices and columns and fails only on the row vector, where it gives all NaN (`nan-output`); `np.diff`'s default axis is right only for the row vector and gives the wrong shape on the other four (`shape-mismatch`); returning the transposed results fails on all five (`transposed-output`).

**Ambiguity variant.** The spec describes the function in words: it rescales the values so they run from 0 to 1 and then applies MATLAB's `diff`, without saying whether the scaling is per column or uses one minimum and maximum for the whole array. The two readings are Octave's `(A - min(A)) ./ (max(A) - min(A))` and the same with `min(A(:))` and `max(A(:))`. On a row or column vector they coincide, so this variant is only scored on the three matrix cases.

---

## Task 3: `simpson` (composite Simpson's rule)

**What it does.** Estimates the integral of a function `f` from `a` to `b` by fitting parabolas through groups of three points. `n` is the number of subintervals and must be even.

```matlab
function I = simpson(f, a, b, n)
  x = linspace(a, b, n+1);
  h = (b - a) / n;
  y = f(x);
  I = h/3 * (y(1) + 4*sum(y(2:2:end-1)) + 2*sum(y(3:2:end-2)) + y(end));
end
```

**Test inputs.** Five cases: a cubic on [0, 2] with n = 2, `exp(-x^2)` on [0, 2] with n = 10, `sin` on [0, pi] with n = 8, `x^4` on [-1, 3] with n = 6, and `1/(1+x^2)` on [0, 1] with n = 4. The functions are passed to the model as Python callables and stored in the answer key by name. Simpson's rule is exact for cubics, so that case can also be checked against the analytic value (4), and the other keys are checked against their true integrals.

**What could go wrong**
- `linspace(a, b, n+1)` includes both ends. The tempting Python equivalent `np.arange(a, b, h)` excludes `b`, and with floating-point steps it can also come out one point long or short. A missing endpoint gives the `endpoint-mismatch` signature.
- The weights alternate 4, 2, 4, ... between the end points. In MATLAB these are `2:2:end-1` and `3:2:end-2`. A 1-based/0-based slip swaps the 4s and 2s.
- `np.linspace(a, b, n)` gives `n` points where the MATLAB code has `n+1`, while the step is still `(b-a)/n`. This is the other way of ending up with the wrong endpoint, and it also gets `endpoint-mismatch`.
- I checked each of these against numpy: the no-endpoint and `linspace(a, b, n)` answers Octave computed are what `np.arange` and `np.linspace` code actually produces, and the swapped-weights answer matches swapped numpy slices.

**Ambiguity variant.** The spec describes the function in words, with "`n` panels" and no code. A panel is one parabola, covering two subintervals, but many people use `n` for the number of subintervals, so both readings are plausible. The two Octave answers are `n` subintervals (the key) and `2n` subintervals. I originally planned "points vs subintervals", but that resolves itself: Simpson's rule needs an even number of subintervals, so only one reading is ever valid for a given `n`. For the cubic, Simpson is exact, so both readings give 4 and that case can't tell them apart. The variant is scored on the other four cases.

---

## Task 4: `robust_stats` (silent numerical drift)

**What it does.** Computes six small things that look trivial and all have language-specific defaults: standard deviation, variance, rounding, `mod`, `rem`, and unsigned 8-bit addition.

```matlab
function [s, v, r, m, rm, sat] = robust_stats(x, a, b, u, w)
  s   = std(x);        % divides by n-1
  v   = var(x);        % divides by n-1
  r   = round(x);      % halves round away from zero
  m   = mod(a, b);     % result takes the sign of b
  rm  = rem(a, b);     % result takes the sign of a
  sat = u + w;         % uint8: stops at 255
end
```

The comments above are for you, the reader. The version the model sees has none, because they would give the traps away.

**Test inputs.** Five cases, each a set of five row vectors. Case 1 has `.5` ties in `x`, mixed signs in `a` and `b`, and `uint8` sums that cross 255. Case 2 is deliberately plain, so only `n-1` vs `n` can show. Case 3 has ties but `a` and `b` both negative, where `mod` and `rem` agree. Case 4 has `n = 2` and full-scale `uint8` values. Case 5 has no ties but mixed signs and sums crossing 255 at different points. `std` and `var` are compared with a tight tolerance. Rounding, `mod`, `rem` and the `uint8` sum are whole numbers and must match exactly.

**What could go wrong.** None of these raise an error; they just return a different number. Each known-wrong translation changes exactly one of these and leaves the rest correct, so the failure points at one cause. I checked each against numpy: `np.std` and `np.var` with their defaults, `np.round`, `%`, `np.fmod` and `uint8 + uint8` reproduce the wrong answers Octave computed.
- **`std` and `var`:** MATLAB divides by `n-1`; numpy's default divides by `n`. The result is off by a constant factor, `sqrt((n-1)/n)` for `std`. Signature: `scale-mismatch`.
- **`round`:** MATLAB rounds 2.5 to 3 and -2.5 to -3. numpy (and Python's own `round`) rounds halves to the nearest even number, giving 2 and -2. Signature: `rounding-mismatch`.
- **`mod` and `rem`:** `mod` follows the sign of the divisor and `rem` follows the sign of the dividend. Python's `%` matches `mod`, so translating `rem` to `%` is wrong for negative inputs (`np.fmod` is the correct one), and the reverse mistake, `mod` as `np.fmod`, is wrong in the same way. Signature: `sign-mismatch`.
- **`uint8` addition:** MATLAB saturates, so 250 + 10 is 255. numpy wraps around, so it gives 4. Signature: `wraparound-mismatch`.
- A wrong translation only shows on the cases where its difference can: with no ties in `x`, rounding halves to even gives the same answer, and with all-positive `a` and `b`, `mod` and `rem` agree.

**Ambiguity variant.** The same function, `robust_stats` with the same six outputs, described in words with no code. Each line of the MATLAB becomes a natural phrase: "the standard deviation of `x`", "the variance of `x`", "`x` rounded to whole numbers", "`a` modulo `b`", "the remainder of `a` divided by `b`", and "`u` plus `w`". Each phrase is open in exactly the way its trap is, so every known-wrong translation above is also a valid second reading of one phrase:

| Phrase | Reading A (the key) | Reading B |
|---|---|---|
| standard deviation, variance | divide by `n-1` | divide by `n` |
| rounded to whole numbers | halves round away from zero | halves round to the nearest even number |
| modulo | sign of the divisor | sign of the dividend (like C's `%`) |
| remainder | sign of the dividend | sign of the divisor (like Python's `%`) |
| `u` plus `w` | `uint8` stops at 255 | wraps around at 256 |

The readings reuse the Octave results of the known-wrong translations, and each covers only the outputs it concerns. On a given case a reading only differs from the key where its trap can show (no ties means no rounding difference, and so on). This was a smaller function, `spread(x)`, in my first version, but then the two specs weren't the same task, so the results couldn't be compared.

---

## Task 5: `rk4_fit` (ODE solver and least-squares fit)

**What it does.** Solves a harmonic oscillator (`y'' = -y`, starting at `y = 1`, `y' = 0`) with a fixed-step fourth-order Runge-Kutta method, then fits a quadratic to the first state with the backslash operator. It returns the full trajectory, the three fitted coefficients, and the trajectory flattened into one column.

```matlab
function [Y, c, yflat] = rk4_fit(h, N)
  f = @(s) [s(2); -s(1)];
  Y = zeros(2, N+1);
  Y(:,1) = [1; 0];
  for i = 1:N
    s  = Y(:,i);
    k1 = f(s);
    k2 = f(s + h/2*k1);
    k3 = f(s + h/2*k2);
    k4 = f(s + h*k3);
    Y(:,i+1) = s + h/6*(k1 + 2*k2 + 2*k3 + k4);
  end
  t = (0:N)' * h;
  A = [ones(N+1,1), t, t.^2];
  c = A \ Y(1,:)';     % least-squares fit
  yflat = Y(:);        % flatten column by column
end
```

The comments above are for you, the reader. The version the model sees has none, because they would give the traps away.

I use a fixed step and not `ode45` because an adaptive solver picks its own steps, so there would be no single right answer to check against.

**Test inputs.** Five `(h, N)` pairs: (0.1, 10), (0.05, 20), (0.2, 5), (0.25, 8) and (0.3, 3). `N` is always at least 3, so the 3-column matrix has more rows than columns and the fit really is least squares. The trajectory and the flattened column are compared with a tight tolerance (1e-12), and the coefficients with a looser one (1e-8).

**What could go wrong**
- The state is stored one column per time step, indexed `Y(:,i)` with `i` from 1. A Python port gets the indexing wrong easily.
- `A \ y` with more rows than columns means least squares. `np.linalg.solve` needs a square matrix and raises `LinAlgError` (checked on all five cases); the right call is `np.linalg.lstsq`. Signature: `runtime-error`.
- `Y(:)` reads down each column in turn, so the two states are interleaved at each time step. numpy's default `ravel()` reads along rows, so all of one state comes first. The values are the same but in a different order. Signature: `permuted-output`.
- `np.polyfit` returns the coefficients highest power first, the reverse of `A \ y`. Same signature, but only the coefficients are wrong.
- Each known-wrong translation changes exactly one output, and I checked each against numpy: `ravel()` and `polyfit` reproduce the wrong answers Octave computed.

**Why the coefficients get a looser tolerance.** The fit is a solve, and different solvers round differently. The conditioning of the 3-column matrix is small (condition number 15.8 to 21.0, recorded in the answer key), so a good solver's error is around 5e-15. Solving the normal equations instead squares that, to around 1e-13. The tolerance has to accept every legitimate route and still reject the wrong answers, which differ at order 1. A test runs five routes (`lstsq`, normal equations, QR, `pinv`, reversed `polyfit`) and checks that each passes using under a tenth of the allowed tolerance.

**Ambiguity variant.** Like Task 4, this is about a smaller function, `rk4_traj(h, N)`, described in words with no code: it solves the same equation "for `N` steps" and returns a 2-row array with one column per time step. It doesn't say whether the starting state is one of the columns. The two readings are the full trajectory with the starting state (2 by N+1, the key's `Y`) and only the N states after it (2 by N). The variant is scored on that single output.

---

## How the ambiguity variants are scored
These are scored separately from pass rate, because Octave's answer is one valid reading and not the only one. Each variant has two Octave answers, one per reading. The outcome for a sample is one of: **asked** a clarifying question, **stated an assumption** and went ahead, **silently chose** reading A, or **silently chose** reading B. Detecting these is done by rule from the response, not by a judge model.

## Settled while building
- The Octave results are `.mat` files (version 7), read with scipy.
- Whether a 1-D result counts as a vector is set per task (`vector_ok` in `task.toml`): yes for Tasks 1, 4 and 5, no for Task 2.
- A single number matches whether it comes back as a float, a 0-d array, or a 1x1 array.
- Where several outputs need different strictness, `task.toml` marks some outputs exact (Task 4) or gives one its own tolerance (Task 5).

## Still to settle
- The exact rules for detecting "asked" vs "stated an assumption".
- Task 5's ambiguity variant is still a smaller function, `rk4_traj`, scored on a single output. It should probably become the same task as the plain spec, as Task 4's now is.
- How the per-prompt confirmation (repeat until two consecutive replies match) is applied to scored samples.
