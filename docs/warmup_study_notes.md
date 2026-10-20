# Warm-up behaviour in llama-server: notes for a possible follow-on project

While checking that two prompts couldn't leak into each other, I found that a freshly started
llama-server answers some prompts differently the first time it sees them than on later repeats, with
identical model, prompt, seed and temperature 0. It's repeatable, so it isn't random noise. I don't know
why. This file records what I saw, so I can pick it up as its own small project later.

## How I describe it
Not as random, and not as "deterministic non-determinism". In these tests the output was **repeatable
for a given server history** (same restart, same request sequence, same replies) but it **depended on that
history**, so the reply to a prompt isn't a function of the prompt alone. That is a narrower claim than
"deterministic": it holds for one model, one GPU, one build and five restarts.

The canary doesn't cure this. After a canary had warmed the server, one task prompt still differed on its
first appearance. The canary is a way to see where replies settle and to log it. Protection for the
benchmark comes from confirming each prompt individually (repeat until two consecutive replies match).

## What I saw
Model: OLMo 3 7B Instruct Q8_0, llama.cpp build b11734 (Vulkan), RX 7900 XT, one slot, prompt cache off,
temperature 0, seed 0. Every request reported `cache_n = 0`, so nothing was served from a prompt cache.
Raw data is in [data/warmup/2026-10-20_first_observations](../data/warmup/2026-10-20_first_observations).

"differs" means the reply differed from the one given by later repeats of the same prompt.

| Situation | What happened |
|---|---|
| 5 fresh restarts, same 7 requests each (amb, plain, amb, plain, plain, plain, amb) | requests 1 and 2 differ; requests 3 to 7 are steady. Identical in all 5 restarts, diverging at the same character. |
| Canary prompt on a fresh server, 6 repeats | request 1 differs (140 tokens against 99); requests 2 to 6 are steady. |
| Task prompts straight after that canary | `amb` already steady on its first appearance; `plain` differs, then steady. |
| First isolation run (fresh, after one short "Say hi") | `amb` steady from the start; `plain`'s first reply differs from its second. |

The differences are late and small: in one case the two replies were identical for the first 1,597 of
1,788 characters. That looks like a near-tie between two tokens flipping, not a different "mode" of
answering.

## What I don't know
- Whether the trigger is the very first request after a start, the first time a prompt is seen, its
  content, its token length (the three prompts were 129, 145 and 157 tokens, so length alone doesn't fit),
  or what the previous request left behind in the GPU.
- Why it differs from what [Thinking Machines](references.md) describe. Their cause is batch size
  changing with server load. Mine is one user, one request at a time.
- Whether it happens on other backends, models or GPUs.

## An experiment that would separate the explanations
On many fresh starts, send controlled prompt sets and record which replies differ:
1. Different prompts with the **same** token count and different content.
2. The **same** content padded to different token counts.
3. One prompt repeated four times, started at different positions in the sequence.
4. The same sequences with the server's own warm-up switched off (`--no-warmup`; the help text says it
   runs an empty warm-up pass by default), and on the other backend or flags (HIP build, flash attention
   on/off, a different micro-batch size).

Analyse with the first differing character and the fingerprint from the canary (`AABBBB`-style patterns),
and count how many restarts reproduce each pattern. With 5 restarts the pattern was identical every time,
so a handful of restarts per condition would be enough.

## What already exists
- `scripts/warmup.py` and `src/bench/warmup.py`: the canary and its JSONL log (header, one record per
  request with the full reply and its hash, timings and `cache_n`, and a summary with the fingerprint).
- `scripts/serve.ps1 -Metrics`: exposes the server's cumulative counters, which the canary stores.
- `scripts/check_isolation.py`: confirms that no request reuses the prompt cache.
- The raw first observations, linked above.

Still to write: a driver that restarts the server and runs a prompt set each time (the throwaway
`restart_driver.ps1` is a starting point), a prompt generator for points 1 and 2 above, and the analysis.

## Caveats
Small numbers (5 restarts), one model, one GPU, one build. The first isolation run's replies were not saved.
Everything above is an observation. None of the causes is established.

## Related
- Thinking Machines, [Defeating Nondeterminism in LLM Inference](references.md), and the nondeterminism
  paper arXiv 2506.09501 (both in the references file).
