# The harness

How a model's reply becomes a result. The harness works on its own (`scripts/run_pilot.py`), and
`scripts/run_inspect.py` drives the same functions through Inspect (see "Inspect" below). Nothing has been run
against the model on the real tasks yet. The dry runs replay the golden solutions instead.

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
| `stated-assumption` | code, and the prose flags the vague point as a choice or assumption |
| `described-choice` | code, and the prose says what was done about the vague point without flagging it |
| `silent` | code with none of those; any choice was made without saying so |
| `no-answer` | no code and no question |

The choice itself (A, a named reading, `other`, `error`, or `mixed` when different outputs picked different readings) is
recorded separately, so "silent, chose A" and "silent, chose the other one" can be told apart.

**How the prose is judged** (`parsing.py`, decided 2026-10-20). These are keyword rules, applied to the prose only (code and
the confidence line removed):

- *Asked a clarifying question*: a sentence with a "?" that is addressed to the user. It needs a clarifying phrase ("do you
  mean", "could you provide ...", "which ...", "should it ...", "is the output meant to ...") and no courtesy phrase ("would
  you like me to add tests?"). A rhetorical "What does this do? It averages each window." is deliberately not counted, which
  means a genuine "What does the function return for X?" with no "you" or "original" in it is missed too.
- *Mentions the vague point*: the reply uses a word from the task's own vocabulary (`[ambiguity] vocabulary` in `task.toml`),
  such as `full window*`, `padded` or `n-k+1` for Task 1, and `n-1`, `ties` or `saturat*` for Task 4. A test checks that no
  vocabulary word already appears in the vague spec, so a reply echoing the spec doesn't count, and that every alternative
  reading can be named with it.
- *Flags it as a choice*: uncertainty language ("assumed", "ambiguous", "interpreted", "I went with", "if you want ...")
  in or next to a sentence that mentions the vague point. "I assumed x is a numpy array" is not about the vague point, so it
  doesn't count.

- *Documented in code* (decided 2026-10-20, recorded apart from the outcome): the same vocabulary applied to the code's own
  comments and docstrings, for example a docstring "Returns: array of length n - k + 1". A reply that is only a code block
  can be `silent` in the prose and still documented, which the first real run showed straight away. Among the replies with
  code, the report gives three numbers: *documented* (the comments or docstrings mention the vague point), *flagged* (they
  also use uncertainty language) and *quiet* (silent in the prose **and** nothing in the code, the strictest reading of
  "quietly picked an answer"). The outcome labels themselves stay prose-only, so "said it to the user" and "wrote it in the
  code" can be reported side by side.

The rules are crude, so the log keeps the sentences and words that matched. After a run every reply to a vague spec is
checked by hand with `scripts/review_replies.py`, which writes a review sheet (the prose, and separately the code's comments
and docstrings) and a labels file with a hand column for the outcome and one for "documented in code". It then reports
how often the rules agreed with the hand labels. That agreement goes in the write-up. Because the text judgements depend only on
the stored reply and the task's vocabulary, `--rescore` recomputes them with the current rules, so a log written before a
rule change can be reviewed without running the model again; the code's results (pass, signature, which reading it chose)
are kept as they were.

## Calibration

`CONFIDENCE: 0-100` against whether the sample passed, as a Brier score next to a baseline that always says the
overall pass rate, and the skill score against that baseline. The mean confidence on failing samples is reported too,
since "does reasoning just make wrong code more confident?" is one of the questions. Plain specs only. A reply with
no usable confidence is counted and left out of the scores.

## Decisions

Reviewed and settled on 2026-10-20. They affect results, so a change to any of them is a new experiment.

- **The system prompts** (`prompts.py`; hashes are in every log). The free-text one reads: *"You translate MATLAB code to
  Python. Put your Python function in a single ```python code block. Finish with a last line of the form `CONFIDENCE: <number
  from 0 to 100>`, your percent confidence that your function is correct."* It says nothing about questions. The
  format-constrained one offers `QUESTION:` or `CODE:` as the first word. Two wordings were changed in review: a "careful
  programmer" persona (a careful person is more likely to ask, which is one of the things measured) and "on every test input"
  (which hints that inputs vary, the very thing some tasks test).
- **The text judgements**: as above, task-aware, with a hand check of the pilot's replies.
- **Confirmation.** Scoring the repeated reply, with at most 6 attempts. It costs 2 to 3 requests per prompt.
- **Sampling.** Temperature 0, seed 0, 1500 new tokens, passed explicitly on every request because this server's own
  defaults are not temperature 0.
- **The libraries in the sandbox.** numpy and scipy. Code importing anything else fails with a runtime error.
- **Calibration** on plain specs only.
- **Alternative readings are a finite list.** A reply that makes some other choice is `other`, which will hide some
  behaviour. Task 1's same-length reading has two alignments for exactly this reason, found when the first test of the
  harness classed ordinary numpy code as matching neither.

## Inspect

Inspect 0.3.282, with no OpenAI SDK. What is in `src/bench/`:

- `inspect_llamaserver.py`, a model provider for `llama-server` (`llamaserver/<label>`). It sends only the messages,
  temperature, seed and max tokens, and raises an error if any of the three is unset. It serialises the HTTP calls with
  a lock, because Inspect's default adaptive concurrency (it starts at 20 connections) overrides a provider's own
  connection limit. The server's `timings` and build fingerprint go into each output's metadata.
- `inspect_pilot.py`: `build_samples` (each sample's input is exactly our system + user messages), the solver
  `confirmed_generate`, the scorer `translation_scorer`, the metrics `pilot_summary` and `ambiguity_summary`, and
  `records_from_log`, which turns an Inspect log back into the records `report.py` reads.
- `scripts/run_inspect.py` puts the run header (hashes, sampling, sandbox image, git state) in the log's metadata.

How I checked it, against a fake server that records every request:

- the request body is exactly `messages`, `temperature`, `seed`, `max_tokens`, and the messages are exactly the two we built;
- identical consecutive calls are not cached (the confirmation step needs real repeats);
- requests never overlap, and arrive in dataset order only with `max_samples=1`, which the script sets;
- the `openai` package is never imported.

Two things found on the way: Inspect's default epoch reducer turns the text in a dict-valued score into numbers
before a metric sees it (so my metrics are declared `scores="unreduced"`), and a pass rate taken from the
calibration summary silently dropped samples with no confidence (so it is now counted over all plain samples).

## Not done

- Think. The pilot is Instruct only.
- A run against the model on the real tasks.

## Running it

```
uv run pytest                         # all tests, about 2 min (the Docker ones are slow)
uv run pytest -m "not docker"         # the fast ones, under a minute
uv run python scripts/run_pilot.py --dry-run          # whole pipeline with golden answers, no model
uv run python scripts/run_inspect.py --dry-run        # the same through Inspect, with a mock model
uv run python scripts/run_inspect.py --tasks t01_movavg   # the real model (start scripts/serve.ps1 first)
```
