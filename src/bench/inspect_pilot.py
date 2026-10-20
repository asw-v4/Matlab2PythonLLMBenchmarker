"""The pilot as an Inspect task: the same harness, driven by Inspect.

    samples   one per (task, spec variant, prompt condition), each holding its own system + user message
    solver    confirmed_generate: send the prompt until two replies in a row match, keep that reply
    scorer    translation_scorer: parse, run in the Docker sandbox, score, name the failure, and for vague
              specs say what the model did
    metrics   pilot_summary (calibration, plain specs) and ambiguity_summary (outcome rates)

Run it with one sample at a time (eval(max_samples=1)) so requests reach the server in a fixed order; the
llamaserver provider already stops them overlapping. See scripts/run_inspect.py.
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Callable

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessageSystem, ChatMessageUser, GenerateConfig, ModelOutput, get_model
from inspect_ai.scorer import Metric, SampleScore, Score, Target, Value, metric, scorer
from inspect_ai.solver import Generate, TaskState, solver

from bench import calibration, prompts, sandbox
from bench import inspect_llamaserver  # noqa: F401  (importing registers the llamaserver provider)
from bench.confirm import confirm_reply_async
from bench.evaluate import AMBIGUOUS, PLAIN, evaluate_reply
from bench.pilot import spec_text
from bench.taskio import load_task

TASKS_DIR = Path(__file__).resolve().parents[2] / "tasks"
PILOT_CONFIG = GenerateConfig(temperature=0.0, seed=0, max_tokens=1500)
OUTCOMES = ("asked", "asked-and-proceeded", "stated-assumption", "silent", "no-answer")
SCORER_NAME = "translation_scorer"


@lru_cache(maxsize=None)
def _task(task_id: str):
    return load_task(TASKS_DIR / task_id)


def all_task_ids() -> list[str]:
    return sorted(p.name for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())


def sample_id(task_id: str, variant: str, condition: str) -> str:
    return f"{task_id}:{variant}:{condition}"


def build_samples(tasks, variants=(PLAIN, AMBIGUOUS), conditions=("free",)) -> list[Sample]:
    """One sample per task, variant and condition, in a fixed order. The input is the whole conversation."""
    samples = []
    for t in sorted(tasks, key=lambda t: t.id):
        for variant in variants:
            for condition in conditions:
                system, user = prompts.build_messages(condition, spec_text(t, variant))
                samples.append(Sample(
                    input=[ChatMessageSystem(content=system["content"]), ChatMessageUser(content=user["content"])],
                    target="", id=sample_id(t.id, variant, condition),
                    metadata={"task_id": t.id, "variant": variant, "condition": condition,
                              "system_sha256": prompts.sha256(system["content"]),
                              "user_sha256": prompts.sha256(user["content"])}))
    return samples


# ---- solver

@solver
def confirmed_generate(max_attempts: int = 6, config: GenerateConfig = PILOT_CONFIG):
    """Ask the model with the sample's messages until two replies in a row are identical; keep that reply.

    Calls the model directly (not Inspect's generate) so that each attempt starts from the same two messages and
    only the confirmed reply is added to the conversation. Every attempt is recorded in the sample's store.
    """
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model = get_model()
        messages = list(state.messages)
        outputs: list[ModelOutput] = []

        async def next_reply() -> dict:
            out = await model.generate(messages, config=config)
            outputs.append(out)
            meta = out.metadata or {}
            timings = meta.get("timings") or {}
            return {"text": out.completion, "finish_reason": meta.get("finish_reason", out.stop_reason),
                    "completion_tokens": out.usage.output_tokens if out.usage else None,
                    "prompt_tokens": out.usage.input_tokens if out.usage else None,
                    "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n")}

        confirmed = await confirm_reply_async(next_reply, max_attempts=max_attempts)
        final = outputs[-1]                      # the reply that repeated (or, if none did, the last one)
        state.output = final
        state.messages.append(final.message)
        replies = {}
        for a in confirmed.attempts:
            replies.setdefault(a.reply_sha256, a.text)
        state.store.set("confirmation", {
            "confirmed": confirmed.confirmed, "n_attempts": confirmed.n_attempts,
            "first_reply_differs": confirmed.first_reply_differs, "truncated": confirmed.truncated,
            "attempts": [a.as_dict(with_text=False) for a in confirmed.attempts],
            "replies": replies, "first_reply": confirmed.first_reply if confirmed.first_reply_differs else None,
        })
        return state

    return solve


# ---- scorer and metrics

def _variant(s: SampleScore) -> str | None:
    return (s.sample_metadata or {}).get("variant")


def _value(s: SampleScore) -> dict:
    v = s.score.value
    return v if isinstance(v, dict) else {}


# scores="unreduced": Inspect's default epoch reducer turns the text values in a dict score (outcome, choice,
# signature) into numbers before a metric sees them. With one epoch there is nothing to reduce anyway.
@metric(scores="unreduced")
def pilot_summary() -> Metric:
    """Plain specs: pass rate and calibration (Brier score against the pass-rate baseline, and skill)."""
    def compute(scores: list[SampleScore]) -> Value:
        plain = [_value(s) for s in scores if _variant(s) == PLAIN]
        c = calibration.summarise([(v.get("confidence"), int(v.get("passed", 0))) for v in plain])
        # the pass rate covers every plain sample; the calibration numbers only those that stated a confidence
        pass_rate = (sum(int(v.get("passed", 0)) for v in plain) / len(plain)) if plain else None
        return {"plain_n": len(plain), "plain_pass_rate": pass_rate, "confidence_scored": c.n,
                "confidence_missing": c.n_missing, "brier": c.brier, "brier_baseline": c.baseline,
                "brier_skill": c.skill, "mean_confidence_passing": c.mean_confidence_when_passing,
                "mean_confidence_failing": c.mean_confidence_when_failing}
    return compute


@metric(scores="unreduced")
def ambiguity_summary() -> Metric:
    """Vague specs: how often the model asked, stated an assumption, or answered silently, and which reading it took."""
    def compute(scores: list[SampleScore]) -> Value:
        amb = [_value(s) for s in scores if _variant(s) == AMBIGUOUS]
        n = len(amb)
        out: dict = {"ambiguous_n": n}
        for label in OUTCOMES:
            out["outcome_" + label.replace("-", "_")] = (sum(v.get("outcome") == label for v in amb) / n) if n else None
        chosen = [v.get("choice") for v in amb if v.get("choice")]
        out["chose_reading_a"] = (sum(c == "A" for c in chosen) / len(chosen)) if chosen else None
        return out
    return compute


@scorer(metrics=[pilot_summary(), ambiguity_summary()], name=SCORER_NAME)
def translation_scorer(runner: Callable | None = None, runner_cfg=None):
    """Evaluate the confirmed reply with the whole harness. `runner(code, task)` runs the model's code on a task's
    cases (default: the Docker sandbox). If the first reply differed from the confirmed one, it is scored too."""
    run = runner or (lambda code, t: sandbox.run_code(code, t, runner_cfg))

    async def score(state: TaskState, target: Target) -> Score:
        meta = state.metadata
        t = _task(meta["task_id"])

        async def evaluate(reply: str):
            return await asyncio.to_thread(evaluate_reply, reply, t, variant=meta["variant"],
                                           condition=meta["condition"], runner=run)

        ev = await evaluate(state.output.completion)
        conf = state.store.get("confirmation")
        first = await evaluate(conf["first_reply"]) if conf and conf.get("first_reply") else None
        return Score(
            value={"passed": int(ev.passed), "signature": ev.signature, "confidence": ev.confidence,
                   "outcome": ev.outcome, "choice": ev.choice_summary},
            answer=ev.outcome or ev.signature,
            explanation=f"{meta['task_id']} {meta['variant']}: {ev.outcome or ('pass' if ev.passed else ev.signature)}",
            metadata={"evaluation": ev.as_dict(), "first_reply_evaluation": first.as_dict() if first else None,
                      "confirmation": conf})

    return score


# ---- the task

def pilot_task(task_ids: list[str] | None = None, variants=(PLAIN, AMBIGUOUS), conditions=("free",),
               max_attempts: int = 6, runner: Callable | None = None, runner_cfg=None,
               config: GenerateConfig = PILOT_CONFIG) -> Task:
    ids = task_ids or all_task_ids()
    unknown = sorted(set(ids) - set(all_task_ids()))
    if unknown:
        raise ValueError(f"no such task(s): {unknown}")
    tasks = [_task(i) for i in ids]
    return Task(dataset=MemoryDataset(build_samples(tasks, variants, conditions), name="matlab2py-pilot"),
                solver=confirmed_generate(max_attempts=max_attempts, config=config),
                scorer=translation_scorer(runner=runner, runner_cfg=runner_cfg),
                config=config, epochs=1, name="matlab2py_pilot")


@task
def pilot(task_ids: str = "", variants: str = "plain,ambiguous", conditions: str = "free", max_attempts: int = 6):
    """For the command line:  inspect eval src/bench/inspect_pilot.py@pilot -T task_ids=t01_movavg"""
    return pilot_task([t for t in task_ids.split(",") if t] or None, tuple(variants.split(",")),
                      tuple(conditions.split(",")), max_attempts)


# ---- from an Inspect log back to our sample records (so report.py works on it)

def records_from_log(log) -> list[dict]:
    """The "sample" records that bench.report expects, rebuilt from an Inspect EvalLog."""
    records = []
    for s in log.samples or []:
        sc = (s.scores or {}).get(SCORER_NAME)
        if sc is None or not sc.metadata:
            continue
        records.append({"type": "sample", "task_id": s.metadata["task_id"], "variant": s.metadata["variant"],
                        "condition": s.metadata["condition"], "evaluation": sc.metadata["evaluation"],
                        "first_reply_evaluation": sc.metadata.get("first_reply_evaluation"),
                        "confirmation": sc.metadata["confirmation"]})
    return records
