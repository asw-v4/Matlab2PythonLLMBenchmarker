# Matlab2PythonLLMBenchmarker

> **LLM use:** This project was planned and written with LLM assistance. This is a personal project of mine :)

I wanted to know how well a small local LLM can translate MATLAB code to Python. Understanding whether it knows when it has got it wrong would be nice. So this is a benchmark of MATLAB-to-Python translation tasks, built on [Inspect](https://inspect.aisi.org.uk/) (`inspect-ai`) and run entirely on my own machine.

**Status:** the five pilot tasks, the harness and the Inspect integration are built and tested ([tasks](docs/tasks.md), [harness](docs/harness.md)). I haven't run the model on them yet.

## What I'm measuring
1. **Pass rate.** Does the Python give the same numbers as the original, within a stated tolerance?
2. **How it fails.** Off-by-one shifts, transposed arrays, wrong integration endpoints, rounding quirks. I label these by what the output looks like, not by diagnosing the cause. That would need reading the code.
3. **Calibration.** The model ends each answer with `CONFIDENCE: 0-100`. Does that match whether the tests passed? Scored with a Brier score against a base-rate baseline.
4. **Ambiguity.** Some task specs are deliberately vague. Does the model ask what I meant, say what it assumed, just describe what it did, or quietly pick an answer? Which answer it picked comes from the code. Whether and how it said so comes from keyword rules on its text, which I then check by hand. A reply that is only code can still explain itself in a docstring, so I record that separately.

## Ground rules
- Everything runs locally. Models are served by llama.cpp's `llama-server` at `localhost:8080/v1`, with no hosted APIs.
- The answer key comes from GNU Octave running the original MATLAB-style code. Octave isn't MATLAB, so strictly the key is "what Octave does", and I pin its version.
- Scoring is plain code (`numpy.testing.assert_allclose`). No LLM marks another LLM's homework.
- Open numerical routines only.
- Generated code runs in a throwaway Linux container (Docker Engine in WSL2), not directly on my machine.

## The models
OLMo 3 7B Instruct is the main model, with OLMo 3 7B Think as the comparison (both Q8_0 GGUF). What I really want to see is whether extra reasoning helps with ambiguity or just makes wrong code more confident. Instruct and Think are trained separately, so any difference can't be pinned on reasoning alone.

## How I use Inspect
Inspect runs the evaluation: each (task, spec) is a sample, and a scorer does the checking. I tested it against a fake server that records every request, and that changed a few things:
- **No OpenAI SDK.** Inspect's usual route to a local server needs the `openai` package, and I don't want it, so there's a small provider for `llama-server` ([src/bench/inspect_llamaserver.py](src/bench/inspect_llamaserver.py)). It sends exactly the messages, temperature, seed and max tokens and nothing else, and refuses to run if any of those three is unset, because the server's own defaults aren't temperature 0. The server's timing numbers, including whether anything came from its prompt cache, are kept in the log.
- **One request at a time, in order.** Inspect runs many requests at once by default, and that overrode the provider's own connection limit. So the provider locks around each call, and the run uses one sample at a time to fix the order.
- **The repeat-until-two-match step is a custom solver** that records every attempt. **Scoring is my own harness** (Docker sandbox, failure signatures, ambiguity rules) and Inspect stores the results. Inspect's default epoch reducer turns text in a dict score into numbers, so my metrics ask for the unreduced scores.

Run it with `uv run python scripts/run_inspect.py` (add `--dry-run` to use a mock model instead of the server).

## One change to the chat template
`llama-server` wouldn't start with OLMo 3's own chat template. The template uses `tools | tojson`, and llama.cpp checks it with `tools` undefined, which errors. I added one line at the top of the template and serve that patched copy ([configs/olmo3_chat_template_patched.jinja](configs/olmo3_chat_template_patched.jinja)):

```
{%- set tools = tools if tools is defined else none -%}
```

- **What it changes:** nothing in the prompt when no tools are passed. I rendered three test conversations (user only, system + user, multi-turn) through `llama-server` and through the official template in `transformers`, and the output matched exactly. I haven't tested requests that pass tools, and the benchmark doesn't use any.
- **A default worth knowing about:** with no system message, the template inserts "You are a helpful function-calling AI assistant. You do not currently have access to any functions." That default isn't mine and it could affect results, so the benchmark prompts always set their own system message.
- **Hashes (SHA-256):** official `f5186d42d99c8a0445d37fd8a6c7ccf07fe3e24a29ce622d8bd245da9507b12b`, patched `dfe20ca3e054b0aef4474a1e5d4a4dae9980b7d1ec157efaf562dbb082c55df0`.

## Reproducibility
- Python and dependencies are pinned (`.python-version`, committed `uv.lock`).
- Every run logs the model repo and commit, GGUF SHA-256, chat template SHA-256, llama.cpp commit and backend, server flags, temperature, seed and Octave version.
- Main runs use temperature 0. On a GPU that doesn't guarantee identical output every time, so I measure how often repeats match instead of assuming.
- No memory carries over between samples. Each sample is its own fresh conversation (a system message plus one spec, never anything from another sample), so the model can't answer the vague spec using what it saw in the clear one. The server runs one slot with prompt caching off, and [scripts/check_isolation.py](scripts/check_isolation.py) confirms it: it interleaves the two specs and checks that no request reuses anything from the cache. Generated code runs in a new container each time.
- A fresh server answers some prompts differently the first time it sees them than on later repeats, even at temperature 0. The same request sequence reproduces the same replies across restarts, so it isn't random, but I don't know the cause yet ([notes](docs/warmup_study_notes.md)). `scripts/warmup.py` sends a fixed canary prompt until four replies in a row are identical and logs every request. The harness also repeats each prompt until two consecutive replies match, scores that reply, and logs every attempt.
- Think's decoding settings come from its model card and are reported as a separate condition.
- Sources are in [docs/references.md](docs/references.md).

## Hypotheses (draft, final before the pilot)
I'm writing these down before I have any results, so I can't quietly change my mind afterwards.
- Think doesn't ask clarifying questions any more often than Instruct.
- Think is at least as confident as Instruct on the samples it gets wrong.

## Limitations
My AMD GPU (RX 7900 XT, 20 GB), a 7B model, five pilot tasks. Results are descriptive, and verbalised confidence from a 7B model will be noisy.
