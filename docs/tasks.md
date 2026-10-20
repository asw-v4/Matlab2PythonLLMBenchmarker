# The five pilot tasks

Each task is a small MATLAB function. The model translates it to Python, and I check the result against what Octave gives for the original. Each one is built around a place where MATLAB and Python quietly disagree, because that is where a translation can look right and still be wrong.

**Status:** draft. The MATLAB below is a sketch of each reference function. The real ones will be written and run in Octave in Stage 6, and anything that changes there gets updated here.

| # | Function | What it's testing | Main trap | Failure signature |
|---|----------|-------------------|-----------|-------------------|
| 1 | `movavg` | Indexing | 1-based loop, end-inclusive slices | `shifted-output`, `window-mismatch` |
| 2 | `colscale_diff` | Array orientation | Default dimension of `min` and `diff` | `shape-mismatch`, `transposed-output` |
| 3 | `simpson` | Integration bounds | Whether the endpoint is included | `endpoint-mismatch` |
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

**Test inputs.** A 4x3 matrix, a 1x5 row vector, and a 5x1 column vector. None has a constant column, so there's no 0/0.

**What could go wrong**
- MATLAB's `min`, `max` and `diff` work along the first dimension that isn't size 1. For a matrix that's down the columns. For a 1x5 row it's along the row, so `min(A)` is a single number and the whole vector is scaled together.
- numpy's `np.diff` defaults to the last axis, not the first. On the 4x3 matrix that differences across each row instead of down it.
- Forcing `axis=0` everywhere fixes the matrix but breaks the row vector: each element becomes its own minimum and maximum, and the result is `0/0 = NaN`.
- These show up as `shape-mismatch`, `transposed-output` or NaN output.

**Ambiguity variant.** The spec says "scale the data to the range 0 to 1" without saying per column or across the whole array.

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

**Test inputs.** A few smooth functions (a cubic, `exp(-x^2)`, `sin`) with different `a`, `b` and `n`. Simpson's rule is exact for cubics, so for that case Octave's answer can also be checked against the analytic value.

**What could go wrong**
- `linspace(a, b, n+1)` includes both ends. The tempting Python equivalent `np.arange(a, b, h)` excludes `b`, and with floating-point steps it can also come out one point long or short. A missing endpoint gives the `endpoint-mismatch` signature.
- The weights alternate 4, 2, 4, ... between the end points. In MATLAB these are `2:2:end-1` and `3:2:end-2`. A 1-based/0-based slip swaps the 4s and 2s.
- Using `n` as the number of points instead of intervals gives a different, and possibly invalid (odd number of intervals), rule.

**Ambiguity variant.** The spec says "use n = 10" without saying whether that means 10 points or 10 subintervals.

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

**Test inputs.** A vector containing exact `.5` values and negatives, integer vectors with mixed signs (no zero divisors), and `uint8` values close to 255.

**What could go wrong.** None of these raise an error; they just return a different number.
- **`std` and `var`:** MATLAB divides by `n-1`; numpy's default divides by `n`. The result is off by a constant factor, `sqrt((n-1)/n)` for `std`. Signature: `scale-mismatch`.
- **`round`:** MATLAB rounds 2.5 to 3 and -2.5 to -3. numpy (and Python's own `round`) rounds halves to the nearest even number, giving 2 and -2. Signature: `rounding-mismatch`.
- **`mod` and `rem`:** `mod` follows the sign of the divisor and `rem` follows the sign of the dividend. Python's `%` matches `mod`, so translating `rem` to `%` is wrong for negative inputs (`np.fmod` is the correct one). Signature: `sign-mismatch`.
- **`uint8` addition:** MATLAB saturates, so 250 + 10 is 255. numpy wraps around, so it gives 4. Signature: `wraparound-mismatch`.

**Ambiguity variant.** The spec says "standard deviation" without saying sample or population.

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

I use a fixed step and not `ode45` because an adaptive solver picks its own steps, so there would be no single right answer to check against.

**Test inputs.** Two or three `(h, N)` pairs, for example `h = 0.1, N = 10`.

**What could go wrong**
- The state is stored one column per time step, indexed `Y(:,i)` with `i` from 1. A Python port gets the indexing wrong easily.
- `A \ y` with more rows than columns means least squares. `np.linalg.solve` needs a square matrix and raises an error; the right call is `np.linalg.lstsq`. Signature: `runtime-error`.
- `Y(:)` reads down each column in turn, so the two states are interleaved at each time step. numpy's default `ravel()` reads along rows, so all of one state comes first. The values are the same but in a different order. Signature: `permuted-output`.
- The coefficients are slightly sensitive to rounding, so their tolerance is looser than the trajectory's. I'll record the matrix's condition number to justify it.

**Ambiguity variant.** The spec says "integrate to t = 1" without saying whether the returned trajectory includes the starting state (N+1 columns) or only the N later ones.

---

## How the ambiguity variants are scored
These are scored separately from pass rate, because Octave's answer is one valid reading and not the only one. Each variant has two Octave answers, one per reading. The outcome for a sample is one of: **asked** a clarifying question, **stated an assumption** and went ahead, **silently chose** reading A, or **silently chose** reading B. Detecting these is done by rule from the response, not by a judge model.

## Still to settle
- Exact output file format for the Octave results (`.mat` read with scipy is the current plan).
- Whether a 1-D numpy result counts as matching a vector output (leaning yes for Task 1, no where orientation is the point, as in Task 2).
- The exact rules for detecting "asked" vs "stated an assumption".
