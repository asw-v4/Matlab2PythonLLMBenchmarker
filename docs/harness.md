# The harness

How a model's reply becomes a result. Everything here works without Inspect; Inspect will later drive the same
functions. Nothing has been run against the model on the real tasks yet. The dry run replays the golden
solutions instead.

## One sample, start to finish

```
spec  ->  fresh conversation  ->  confirm  ->  parse  ->  run in sandbox  ->  score  ->  log
```

1. **Fresh conversation** (`prompts.py`). One system message and one user message holding the spec. Never anything
   from another sample, and a test checks every request against every spec to make sure of it.
2. **Confirm** (`confirm.py`). The prompt is sent until two replies in a row are identical, and that reply is the one
   scored (at most 6 attempts). A fresh server answers some prompts differently the first time
   ([notes](warmup_study_notes.md)), so a reply that has repeated is the safer one to judge. If the first reply
   differs from the confirmed one, it is scored too and both are reported. Every attempt is logged.
3. **Parse** (`parsing.py`). The first Python block that defines the task's function, the last `CONFIDENCE: n` line,
   and two text judgements (below).
4. **Run** (`sandbox.py`). The code runs on every test case in a throwaway container (below).
5. **Score** (`scoring.py`, `signatures.py`). The output has to match Octave's within the task's tolerances. A failure is
   named by what its output looks like.
6. **For a vague spec** (`ambiguity.py`). Which reading did the code's output match, and did the reply ask or
   say what it had assumed?
7. **Log** (`runlog.py`, `pilot.py`). One JSON line per sample, written as it finishes.

## The sandbox

One container per sample, from an image with numpy 2.2.6 and scipy 1.18.1 (the versions in `uv.lock`), built by
`scripts/build_sandbox.ps1`. It has:

- no network, not even DNS
- a read-only root filesystem and a read-only mount for the code, plus a 16 MB `/tmp` that can't run programs
- a non-root user, no Linux capabilities, `no-new-privileges`
- 512 MB of memory (no swap), one CPU, 128 processes, 60 seconds
- one BLAS thread, so sums are added in the same order every run

Each restriction has a test that tries to break it from inside the container. For five of them I also weakened the
restriction on purpose and checked that its test then failed, which caught one test that was passing for the wrong
reason.

**Limit.** This protects against careless code, not an adversary. The result comes back on stdout, so code could
print a fake result line. These are replies from a small local model, not hostile input.

## Failure signatures

A failing case gets a label from what its output looks like, not from a diagnosis:

1. it didn't run: `timeout`, `runtime-error`, `bad-output`, `wrong-output-count`, or `no-code` for a reply without usable code
2. it equals one of the task's known-wrong translations: that variant's label (NaN equals NaN here, because some of those answers contain NaN)
3. a generic pattern on the first wrong output: `nan-output`, `transposed-output`, `shifted-output` (a value or two short), `shape-mismatch`, `permuted-output`, `scale-mismatch` (needs at least two non-zero values to share the ratio), `value-mismatch`

A sample's label is its commonest failing label (ties go alphabetically). An output equal to the key shifted by one
is called `shifted-output`; that is evidence of an indexing error, not proof. Finding the cause would need the
code read.

## Vague specs

Judged mainly from the code. For each output that has alternative readings, the function's results on the cases that
tell the readings apart are compared with reading A (the key) and with each alternative. The sample is then described by
what it did with the vagueness:

| Outcome | Meaning |
|---|---|
| `asked` | no code, a clarifying question |
| `asked-and-proceeded` | a clarifying question, and code |
| `stated-assumption` | code, and the prose says what was assumed |
| `silent` | code with neither; any choice was made without saying so |
| `no-answer` | no code and no question |

The choice itself (A, a named reading, `other`, `error`, or `mixed` when different outputs picked different readings) is
recorded separately, so "silent, chose A" and "silent, chose the other one" can be told apart.

## Calibration

`CONFIDENCE: 0-100` against whether the sample passed, as a Brier score next to a baseline that always says the
overall pass rate, and the skill score against that baseline. The mean confidence on failing samples is reported too,
since "does reasoning just make wrong code more confident?" is one of the questions. Plain specs only. A reply with
no usable confidence is counted and left out of the scores.

## Draft decisions to review

These are my choices, not yours, and they affect results:

- **The system prompts** (`prompts.py`). The free-text one asks for a code block and a confidence line and says
  nothing about questions; the format-constrained one offers `QUESTION:` or `CODE:`. Their hashes are in every log.
- **The two text judgements** (`parsing.py`). "Asked a clarifying question" needs a clarifying phrase ("do you mean",
  "which ...", "should it ...", "is the output meant to ...") and no courtesy phrase ("would you like me to ..."). "Stated an
  assumption" needs words like assume, ambiguous, interpret, convention or "I went with". These are keyword rules. The
  log records the sentences and words that matched so they can be checked by eye, and the outcome rests mainly on the
  code, but expect some misjudgements.
- **Confirmation.** Scoring the repeated reply, with at most 6 attempts. It costs 2 to 3 requests per prompt.
- **Sampling.** Temperature 0, seed 0, 1500 new tokens, passed explicitly on every request because this server's own
  defaults are not temperature 0.
- **The libraries in the sandbox.** numpy and scipy. Code importing anything else fails with a runtime error.
- **Alternative readings are a finite list.** A reply that makes some other choice is `other`, which will hide some
  behaviour. Task 1's same-length reading has two alignments for exactly this reason, found when the first test of the
  harness classed ordinary numpy code as matching neither.

## Not done

- The Inspect integration (needs its documentation).
- Think. The pilot is Instruct only.
- A run against the model on the real tasks.

## Running it

```
uv run pytest                         # all tests, about 90 s (the Docker ones are slow)
uv run pytest -m "not docker"         # the fast ones, about 20 s
uv run python scripts/run_pilot.py --dry-run          # whole pipeline with golden answers, no model
uv run python scripts/run_pilot.py --tasks t01_movavg # the real model (start scripts/serve.ps1 first)
```
