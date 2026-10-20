# First warm-up observations (2026-10-20)

Raw evidence behind [docs/warmup_study_notes.md](../../../docs/warmup_study_notes.md). Kept as it was
collected, not tidied.

- `restart_study/restart_N.json`: five fresh server starts (runs 1, 2, 3, 5, 6; run 4 was interrupted).
  Each file is the same seven requests in order: amb, plain, amb, plain, plain, plain, amb, with full
  reply text. "amb" is `tasks/t01_movavg/spec_ambiguous.md`, "plain" is `spec.md`.
- `canary_first_run.jsonl`: the first canary run on a fresh server (`scripts/warmup.py`), full log.
- `scripts/`: the throwaway scripts that produced the above. They hard-code temp-folder paths from the
  day and are archival. They are not part of the benchmark.
  - `restart_driver.ps1` restarts the server and runs `restart_runs.py` each time.
  - `analyse_restarts2.py` prints which distinct reply each request position got.
  - `transfer_check.py` checks whether the warm-up from the canary carried over to the task prompts.

Not saved: the first isolation run, where the plain spec's first two replies differed. Only its
True/False comparison lines were recorded.
