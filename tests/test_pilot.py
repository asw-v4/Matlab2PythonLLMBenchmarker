import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import reply_with, run_local  # noqa: E402

from bench import pilot, prompts  # noqa: E402
from bench.runlog import RunLog, read_log  # noqa: E402
from bench.taskio import load_task  # noqa: E402

TASKS_DIR = Path(__file__).parent.parent / "tasks"
ALL = [load_task(p) for p in sorted(TASKS_DIR.iterdir()) if (p / "task.toml").exists()]
T1 = next(t for t in ALL if t.id == "t01_movavg")
PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": 100}
GOLDEN_T1 = (T1.dir / "solution.py").read_text()
WRONG_T1 = ("import numpy as np\n"
            "def movavg(x, k):\n"
            "    n = x.shape[1]\n"
            "    return np.array([[x[0, i:i + k].mean() for i in range(n - k)]])\n")


class Recorder:
    """An `ask` that records every conversation it is sent and answers from a script per user message."""

    def __init__(self, answers):
        self.answers, self.calls, self.counts = answers, [], {}

    def __call__(self, url, messages, **params):
        user = next(m["content"] for m in messages if m["role"] == "user")
        n = self.counts[user] = self.counts.get(user, 0) + 1
        self.calls.append((messages, params))
        script = self.answers[user] if isinstance(self.answers, dict) else self.answers
        text = script[min(n, len(script)) - 1] if isinstance(script, list) else script
        return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"completion_tokens": 5, "prompt_tokens": 5}, "timings": {"prompt_n": 5, "cache_n": 0}}


def go(tasks, ask, **kw):
    args = dict(variants=["plain", "ambiguous"], conditions=["free"], url="u", ask=ask, runner=run_local,
                params=PARAMS, max_attempts=6)
    args.update(kw)
    return pilot.run_pilot(tasks, **args)


# ---- the whole thing, dry-run style, on the real tasks

def test_dry_run_of_every_task_passes_plain_and_is_silent_reading_a_on_ambiguous():
    samples = go(ALL, pilot.dry_run_ask(ALL))
    assert len(samples) == len(ALL) * 2
    for s in samples:
        e = s["evaluation"]
        assert e["passed"] and e["signature"] == "pass" and e["confidence"] == 100
        assert s["confirmation"]["confirmed"] and s["confirmation"]["n_attempts"] == 2
        assert not s["confirmation"]["first_reply_differs"] and s["first_reply_evaluation"] is None
        if s["variant"] == "ambiguous":
            assert e["outcome"] == "silent" and e["choice_summary"] == "A"
            assert set(e["choices"].values()) <= {"A", "not-informative"}


def test_samples_come_in_a_fixed_order():
    samples = go(list(reversed(ALL)), pilot.dry_run_ask(ALL), conditions=["free", "format"])
    keys = [(s["task_id"], s["variant"], s["condition"]) for s in samples]
    assert keys == sorted(keys, key=lambda k: (k[0], ["plain", "ambiguous"].index(k[1]), ["free", "format"].index(k[2])))
    assert len(keys) == len(ALL) * 4


def test_each_sample_is_logged_as_it_finishes_with_a_header_first_and_a_summary_last(tmp_path):
    ask = pilot.dry_run_ask([T1])
    with RunLog(tmp_path / "run.jsonl") as log:
        log.write(pilot.build_run_header(tasks=[T1], conditions=["free"], variants=["plain", "ambiguous"],
                                        params=PARAMS, max_attempts=6, url="u", dry_run=True))
        samples = go([T1], ask, log=log)
        log.write(pilot.summary_record(samples))
    recs = read_log(tmp_path / "run.jsonl")
    assert [r["type"] for r in recs] == ["header", "sample", "sample", "summary"]
    assert recs[-1]["n_samples"] == 2 and {g["variant"] for g in recs[-1]["groups"]} == {"plain", "ambiguous"}
    assert recs[-1]["calibration_plain"]["n"] == 1


def test_the_progress_callback_sees_every_sample():
    seen = []
    go([T1], pilot.dry_run_ask([T1]), on_sample=seen.append)
    assert [s["variant"] for s in seen] == ["plain", "ambiguous"]


# ---- every sample is a fresh conversation

def test_every_request_is_exactly_one_system_and_one_user_message_and_never_another_samples_spec():
    ask = Recorder(pilot.golden_replies(ALL) and {u: r for u, r in pilot.golden_replies(ALL).items()})
    go(ALL, ask)
    specs = {pilot.spec_text(t, v).strip() for t in ALL for v in ("plain", "ambiguous")}
    assert ask.calls
    for messages, params in ask.calls:
        assert [m["role"] for m in messages] == ["system", "user"]
        assert messages[0]["content"] == prompts.SYSTEM_FREE
        mine = [s for s in specs if s in messages[1]["content"]]
        assert len(mine) == 1, "a request must contain its own spec and no other"
        assert params == PARAMS


def test_the_format_condition_changes_only_the_system_message():
    ask = Recorder(pilot.golden_replies([T1]))
    go([T1], ask, variants=["plain"], conditions=["free", "format"])
    free, fmt = ask.calls[0][0], ask.calls[-1][0]
    assert free[0]["content"] == prompts.SYSTEM_FREE and fmt[0]["content"] == prompts.SYSTEM_FORMAT
    assert free[1] == fmt[1]


# ---- the confirmation protocol, in the loop

def test_a_first_reply_that_differs_is_evaluated_separately_and_both_are_kept():
    spec = pilot.spec_text(T1, "plain")
    user = prompts.build_messages("free", spec)[1]["content"]
    ask = Recorder({user: [reply_with(WRONG_T1), reply_with(GOLDEN_T1), reply_with(GOLDEN_T1)]})
    s = go([T1], ask, variants=["plain"])[0]
    assert s["confirmation"]["n_attempts"] == 3 and s["confirmation"]["first_reply_differs"]
    assert s["evaluation"]["passed"]                                               # the confirmed reply is scored
    assert not s["first_reply_evaluation"]["passed"] and s["first_reply_evaluation"]["signature"] == "shifted-output"
    assert len(s["replies"]) == 2 and len(s["confirmation"]["attempts"]) == 3
    assert all("text" not in a for a in s["confirmation"]["attempts"])             # texts live in `replies`


def test_a_prompt_that_never_settles_is_marked_unconfirmed_and_the_last_reply_is_scored():
    user = prompts.build_messages("free", pilot.spec_text(T1, "plain"))[1]["content"]
    ask = Recorder({user: [reply_with(WRONG_T1), reply_with(GOLDEN_T1)] * 10})
    s = go([T1], ask, variants=["plain"], max_attempts=4)[0]
    assert not s["confirmation"]["confirmed"] and s["confirmation"]["n_attempts"] == 4
    assert s["evaluation"]["passed"]                                                # attempt 4 was the golden one


# ---- the header

def test_header_ties_the_run_to_its_exact_inputs():
    h = pilot.build_run_header(tasks=ALL, conditions=["free", "format"], variants=["plain"], params=PARAMS,
                               max_attempts=6, url="u", manifest={"name": "m"}, template_file_sha="abc",
                               lock_file=TASKS_DIR.parent / "uv.lock", note="n", context={"k": "v"}, dry_run=True)
    json.dumps(h)
    assert h["type"] == "header" and h["dry_run"] and h["sampling"] == PARAMS and h["max_attempts"] == 6
    assert set(h["tasks"]) == {t.id for t in ALL} and all(len(t["sha256"]) == 64 for t in h["tasks"].values())
    assert h["prompts"] == {"free": prompts.sha256(prompts.SYSTEM_FREE), "format": prompts.sha256(prompts.SYSTEM_FORMAT)}
    assert h["tasks"]["t01_movavg"]["octave_version"] == T1.octave_version
    assert len(h["lock_file_sha256"]) == 64 and h["model"] == {"name": "m"} and h["chat_template_file_sha256"] == "abc"
    assert h["context"] == {"k": "v"} and h["note"] == "n" and "python" in h["host"]


def test_header_leaves_out_the_local_work_folder_but_keeps_the_sandbox_limits():
    from bench.sandbox import SandboxConfig
    h = pilot.build_run_header(tasks=[T1], conditions=["free"], variants=["plain"], params=PARAMS, max_attempts=2,
                               url="u", sandbox_cfg=SandboxConfig(), image_id="sha256:x")
    cfg = h["sandbox"]["config"]
    assert "workroot" not in cfg and cfg["memory"] == "512m" and cfg["image"] and h["sandbox"]["image_id"] == "sha256:x"


# ---- the dry-run helper

def test_golden_replies_cover_every_spec_once():
    table = pilot.golden_replies(ALL)
    assert len(table) == len(ALL) * 2
    ask = pilot.dry_run_ask(ALL)
    for t in ALL:
        for v in ("plain", "ambiguous"):
            msgs = prompts.build_messages("free", pilot.spec_text(t, v))
            assert "def " + t.function in ask("u", msgs)["choices"][0]["message"]["content"]
