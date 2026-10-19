# References for Matlab2PythonLLMBenchmarker

Compiled 2026-10-19 from web search results. **Status key:**
- **Summary only** = I saw a search-result summary, not the full text. Read before citing.
- **Not found** = searched for it; the search did not return it.

Nothing here has been read in full. Treat every summary as a lead, not a verified claim.

---

## 1. Temperature for reasoning models (Think variant)

### DeepSeek-R1 usage guidance
- Link (mirror): https://docs.together.ai/docs/deepseek-r1
- Original paper to check: arXiv 2501.12948 (**not returned by search; verify exact wording there**)
- Status: Summary only, via mirrors of the DeepSeek-R1 README.
- Claim: recommends temperature 0.5-0.7 (0.6 suggested) and top-p 0.95, "to prevent endless repetitions or incoherent outputs". Also says R1-series models sometimes skip their thinking block and recommends forcing responses to start with `<think>\n`.
- Use here: supports running Think at a nonzero temperature. It is the R1 authors' advice, not evidence for OLMo 3.
- A community guide says temperature 0 is a poor choice for reasoning models. That is a third-party opinion, not official.

### Repetition reduced by temperature sampling
- Link: https://arxiv.org/pdf/2602.01198 ("A State-Transition Framework for Efficient LLM Reasoning")
- Status: Summary only (I saw one quoted sentence).
- Claim: a temperature-based sampling strategy reduces the share of repeated generations and improves reasoning performance.
- Use here: possible supporting citation. Check that the study covers models like ours.

### OLMo 3 Think model card (check for its own recommended settings)
- Link: https://huggingface.co/allenai/Olmo-3-7B-Think
- Status: Not read. **Action: read it for recommended temperature/top-p before choosing Think's decoding settings.**

---

## 2. Why temperature 0 is not guaranteed reproducible

### Understanding and Mitigating Numerical Sources of Nondeterminism in LLM Inference
- Link: https://arxiv.org/pdf/2506.09501
- Status: Summary only.
- Claim: even greedy decoding can differ across runs because of numerical precision issues; PyTorch's deterministic-algorithm flags do not fully remove this.
- Use here: justification for recording build/backend/flags and for measuring repeat-identity rather than assuming it.

### Separating sampling randomness from systems nondeterminism
- Link: https://arxiv.org/html/2601.07239v1
- Status: Summary only.
- Claim: distinguishes intentional sampling randomness from numerical and systems nondeterminism (batching, floating point) at temperature 0.

### Defeating Nondeterminism in LLM Inference (Thinking Machines Lab, Horace He)
- Link: not obtained. Only secondary write-ups were seen; look up the original post by title.
- Status: Summary only, via secondary sites.
- Claim (secondary): the cause is batch-dependent kernels, with fixes needed for RMSNorm, matrix multiplication and attention.
- Use here: background for why server batching/prompt caching matters.

### Eliminating Nondeterminism in LLM Inference via Integer-Only Arithmetic
- Link: https://neurips.cc/virtual/2026/170003
- Status: Summary only (abstract; results not verified).
- Claim: inference is nondeterministic even at temperature 0; proposes integer arithmetic so reduction order does not matter.
- Use here: optional background only.

---

## 3. OLMo 3

### Olmo 3 technical report
- Link: https://arxiv.org/pdf/2512.13961 (arXiv 2512.13961, 15 Dec 2025)
- Status: Summary only.
- Content: describes Olmo 3 at 7B and 32B, with Base, Think, Instruct and RL-Zero variants, released with the full model flow (stages, checkpoints, data, dependencies).
- Use here: primary citation for the model. The Hugging Face card showed its paper link as "TBD", so cite this report.
- Release dates differ between third-party listings. Take dates from the paper.

### Model cards
- Think 7B: https://huggingface.co/allenai/Olmo-3-7B-Think
- Instruct 7B: https://huggingface.co/allenai/Olmo-3-7B-Instruct/blob/main/README.md
- Third-party listings give the licence as Apache 2.0. **Verify on the model card.**

---

## 4. Calibration (Brier score)

### Calibration and Correctness of Language Models for Code
- Link: https://arxiv.org/pdf/2402.02047
- Status: Summary only.
- Claim: uses Brier score and ECE for code models. Defines an "unskilled" baseline that always predicts the base rate of correctness, and a Brier skill score against it. Reports poor calibration for GPT-3.5 with verbalized confidence on line-level completion.
- Use here: borrow the base-rate baseline. A bare Brier score on a small benchmark is hard to interpret without it.

### Other calibration papers that came up (titles only; not assessed)
- ConfTuner (tokenized Brier score for verbal confidence): https://arxiv.org/pdf/2508.18847
- Calibrating Expressions of Certainty: https://arxiv.org/pdf/2410.04315
- Localized Calibrated Uncertainty in Code Language Models: https://arxiv.org/pdf/2512.24560

### Kadavath et al., "Language Models (Mostly) Know What They Know"
- Status: **Not found** by search. Search again by title if wanted.

---

## 5. Clarifying questions on ambiguous specifications

### ClarifyGPT (FSE 2024)
- Link: https://arxiv.org/pdf/2310.10996
- Status: Summary only.
- Method: checks whether a requirement is ambiguous by generating test inputs and comparing outputs across several code samples. If outputs disagree, the LLM writes clarifying questions, and the answers are folded into the requirement before final code generation.
- Reported: GPT-4 Pass@1 rose from 70.96% to 80.80% on MBPP-sanitized in a 10-participant human study.
- Use here: closest precedent for measuring ask-vs-silently-resolve.

### ClarifyCodeBench
- Link: https://arxiv.org/pdf/2607.00711
- Status: Title only. Do not cite before reading what it does.

---

## 6. Prior work on code translation

I found **no dedicated MATLAB-to-Python benchmark** in a small search. That does not prove none exists. Word it as "I found none" in the README.

- Diehl et al., Llama 2-70B translating MATLAB to Python and R (small hand-run study, not a benchmark): https://arxiv.org/pdf/2503.19217
- Pietrini (2025), MATLAB-to-Python for scientific scripts, oil-and-gas case study reviewed by domain experts: https://www.unimercatorum.iris.cineca.it/handle/20.500.12606/31646
- PolyHumanEval (HumanEval extended to 14 languages): https://arxiv.org/abs/2410.09812v1
- RepoTransBench (repository-level translation, executable tests): https://arxiv.org/abs/2412.17744v1
- CodeTransOcean (large multilingual translation benchmark; MATLAB inclusion not confirmed): https://alphaxiv.org/benchmarks/university-of-illinois-at-urbana-champaign/codetransocean

---

## Still to look up
- DeepSeek-R1 paper, arXiv 2501.12948 (exact temperature wording)
- OLMo 3 Think model card (recommended decoding settings)
- Thinking Machines "Defeating Nondeterminism in LLM Inference" (original post)
- Kadavath et al. (calibration)
- Inspect (inspect-ai) documentation, to be pasted by the user before any Inspect code is written
