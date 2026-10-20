import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_openai import FakeServer  # noqa: E402

from inspect_ai import Task, eval as inspect_eval  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import (ChatMessageAssistant, ChatMessageSystem, ChatMessageTool, ChatMessageUser,  # noqa: E402
                              GenerateConfig)
from inspect_ai.scorer import Score, Target, mean, scorer  # noqa: E402
from inspect_ai.solver import TaskState, generate  # noqa: E402

from bench import inspect_llamaserver as ls  # noqa: E402

CONFIG = GenerateConfig(temperature=0.0, seed=0, max_tokens=123)
TIMINGS = {"prompt_n": 7, "cache_n": 0, "predicted_n": 5}


@scorer(metrics=[mean()])
def one():
    async def score(state: TaskState, target: Target):
        return Score(value=1.0)
    return score


def samples(n=3):
    return [Sample(input=[ChatMessageSystem(content="SYS"), ChatMessageUser(content=f"spec {i}\n")], id=f"s{i}")
            for i in range(1, n + 1)]


def run(srv, tmp_path, *, config=CONFIG, n=3, **kw):
    task = Task(dataset=MemoryDataset(samples(n)), solver=generate(), scorer=one(), config=config)
    logs = inspect_eval(task, model="llamaserver/olmo3-test", model_base_url=srv.base_url, log_dir=str(tmp_path),
                        display="none", **kw)
    return read_eval_log(logs[0].location)


# ---- small pure functions

@pytest.mark.parametrize("given,root", [
    ("http://127.0.0.1:8080/v1", "http://127.0.0.1:8080"),
    ("http://127.0.0.1:8080/v1/", "http://127.0.0.1:8080"),
    ("http://127.0.0.1:8080", "http://127.0.0.1:8080"),
    ("http://host:9/", "http://host:9"),
])
def test_server_root(given, root):
    assert ls.server_root(given) == root


def test_server_root_falls_back_to_the_environment_then_the_default(monkeypatch):
    monkeypatch.delenv(ls.ENV_BASE_URL, raising=False)
    assert ls.server_root(None) == ls.DEFAULT_URL
    monkeypatch.setenv(ls.ENV_BASE_URL, "http://elsewhere:1/v1")
    assert ls.server_root(None) == "http://elsewhere:1"
    assert ls.server_root("http://explicit:2") == "http://explicit:2"          # an explicit URL wins


def test_messages_become_plain_role_and_text_dicts():
    msgs = [ChatMessageSystem(content="s"), ChatMessageUser(content="u"), ChatMessageAssistant(content="a")]
    assert ls.to_messages(msgs) == [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
                                    {"role": "assistant", "content": "a"}]


def test_tool_messages_are_refused():
    with pytest.raises(ValueError, match="tool"):
        ls.to_messages([ChatMessageTool(content="x")])


@pytest.mark.parametrize("config,missing", [
    (GenerateConfig(), ["temperature", "seed", "max_tokens"]),
    (GenerateConfig(temperature=0.0, max_tokens=5), ["seed"]),
    (GenerateConfig(seed=0, max_tokens=5), ["temperature"]),
    (GenerateConfig(temperature=0.0, seed=0), ["max_tokens"]),
])
def test_all_three_settings_are_required(config, missing):
    with pytest.raises(ValueError) as e:
        ls.required_settings(config)
    for name in missing:
        assert name in str(e.value)


def test_zero_is_a_valid_setting_not_a_missing_one():
    assert ls.required_settings(GenerateConfig(temperature=0.0, seed=0, max_tokens=1)) == (0.0, 0, 1)


def test_output_from_a_response_keeps_text_usage_and_the_servers_own_numbers():
    resp = {"choices": [{"message": {"content": "hi"}, "finish_reason": "length"}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            "timings": TIMINGS, "system_fingerprint": "b1-abc"}
    out = ls.output_from_response("m", resp)
    assert out.completion == "hi" and out.stop_reason == "max_tokens"
    assert (out.usage.input_tokens, out.usage.output_tokens, out.usage.total_tokens) == (3, 4, 7)
    assert out.metadata == {"finish_reason": "length", "timings": TIMINGS, "system_fingerprint": "b1-abc"}


def test_an_empty_or_missing_reply_is_an_empty_string_and_odd_finish_reasons_are_unknown():
    out = ls.output_from_response("m", {"choices": [{"message": {"content": None}, "finish_reason": "weird"}]})
    assert out.completion == "" and out.stop_reason == "unknown" and out.usage is None


# ---- through a real Inspect run, against a fake server

def test_inspect_sends_exactly_the_messages_and_the_three_settings_and_nothing_else(tmp_path):
    with FakeServer(lambda b, n: f"reply {n}", extra={"timings": TIMINGS, "system_fingerprint": "b1-abc"}) as srv:
        log = run(srv, tmp_path, max_samples=1)         # one sample at a time keeps the requests in dataset order
        assert log.status == "success"
        assert len(srv.requests) == 3
        for i, body in enumerate(srv.requests, 1):
            assert set(body) == {"messages", "temperature", "seed", "max_tokens"}
            assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": f"spec {i}\n"}]
            assert (body["temperature"], body["seed"], body["max_tokens"]) == (0.0, 0, 123)
        assert set(srv.paths) == {"/v1/chat/completions"}


def test_the_servers_numbers_reach_the_inspect_log(tmp_path):
    with FakeServer(lambda b, n: "ok", extra={"timings": TIMINGS, "system_fingerprint": "b1-abc"}) as srv:
        log = run(srv, tmp_path, n=1)
    out = log.samples[0].output
    assert out.completion == "ok" and out.metadata["timings"] == TIMINGS
    assert out.metadata["system_fingerprint"] == "b1-abc" and out.usage.total_tokens == 15


def test_each_sample_is_a_fresh_conversation_in_the_log_too(tmp_path):
    with FakeServer(lambda b, n: f"reply {n}") as srv:
        log = run(srv, tmp_path, max_samples=1)
    for i, s in enumerate(log.samples, 1):
        assert [(m.role, m.text) for m in s.messages] == [("system", "SYS"), ("user", f"spec {i}\n"),
                                                          ("assistant", f"reply {i}")]


def test_requests_never_overlap_even_if_more_samples_are_allowed_to_run_at_once(tmp_path):
    # the provider allows one connection, so requests are serialised; their ORDER is only guaranteed with
    # max_samples=1, which is why the runner sets it
    with FakeServer(lambda b, n: "ok", delay=0.05) as srv:
        run(srv, tmp_path, n=6, max_samples=6)
        assert len(srv.requests) == 6 and srv.max_in_flight == 1


def test_with_max_samples_1_the_requests_arrive_in_dataset_order(tmp_path):
    with FakeServer(lambda b, n: "ok", delay=0.02) as srv:
        run(srv, tmp_path, n=8, max_samples=1)
        assert [b["messages"][1]["content"] for b in srv.requests] == [f"spec {i}\n" for i in range(1, 9)]


def test_a_missing_setting_fails_the_sample_instead_of_using_a_default(tmp_path):
    with FakeServer(lambda b, n: "ok") as srv:
        log = run(srv, tmp_path, config=GenerateConfig(temperature=0.0, max_tokens=5), n=1)    # no seed
        assert not srv.requests, "nothing may be sent without an explicit seed"
    assert log.status == "error" or any(s.error for s in (log.samples or []))
    text = str(log.error) + "".join(str(s.error) for s in (log.samples or []))
    assert "seed" in text


def test_a_server_that_is_down_fails_the_sample_loudly(tmp_path):
    class Dead:
        base_url = "http://127.0.0.1:9/v1"
    log = run(Dead(), tmp_path, n=1)
    assert log.status == "error" or any(s.error for s in (log.samples or []))


def test_the_openai_sdk_is_never_loaded(tmp_path):
    code = (
        "import sys; sys.path[:0] = [r'%s', r'%s']\n"
        "from fake_openai import FakeServer\n"
        "from inspect_ai import Task, eval\n"
        "from inspect_ai.dataset import MemoryDataset, Sample\n"
        "from inspect_ai.model import GenerateConfig\n"
        "from inspect_ai.solver import generate\n"
        "import bench.inspect_llamaserver\n"
        "with FakeServer(lambda b, n: 'ok') as srv:\n"
        "    t = Task(dataset=MemoryDataset([Sample(input='hi', id='a')]), solver=generate(),\n"
        "             config=GenerateConfig(temperature=0.0, seed=0, max_tokens=5))\n"
        "    eval(t, model='llamaserver/x', model_base_url=srv.base_url, log_dir=r'%s', display='none')\n"
        "print('OPENAI_LOADED' if any(m == 'openai' or m.startswith('openai.') for m in sys.modules) else 'clean')\n"
    ) % (Path(__file__).parent, Path(__file__).parent.parent / "src", tmp_path)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=180)
    assert r.stdout.strip().endswith("clean"), r.stdout + r.stderr
