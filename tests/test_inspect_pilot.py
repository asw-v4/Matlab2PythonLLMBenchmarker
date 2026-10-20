import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_openai import FakeServer  # noqa: E402
from helpers import reply_with, run_local  # noqa: E402

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.scorer import SampleScore, Score  # noqa: E402

from bench import inspect_pilot as ip  # noqa: E402
from bench import pilot, prompts, report  # noqa: E402

T1 = ip._task("t01_movavg")
ALL = [ip._task(i) for i in ip.all_task_ids()]
GOLDEN_T1 = (T1.dir / "solution.py").read_text()
WRONG_T1 = ("import numpy as np\n"
            "def movavg(x, k):\n"
            "    n = x.shape[1]\n"
            "    return np.array([[x[0, i:i + k].mean() for i in range(n - k)]])\n")
TIMINGS = {"prompt_n": 7, "cache_n": 0}
LOCAL = lambda code, task: run_local(code, task)  # noqa: E731  (trusted test code only)


def user_of(body):
    return next(m["content"] for m in body["messages"] if m["role"] == "user")


def run(srv, tmp_path, task_ids=("t01_movavg",), **kw):
    t = ip.pilot_task(list(task_ids), runner=LOCAL, **{k: v for k, v in kw.items() if k in ("variants", "conditions", "max_attempts")})
    logs = inspect_eval(t, model="llamaserver/olmo3-test", model_base_url=srv.base_url, log_dir=str(tmp_path),
                        max_samples=1, display="none")
    return read_eval_log(logs[0].location)


def metrics(log):
    return {k: m.value for k, m in log.results.scores[0].metrics.items()}


# ---- samples

def test_samples_are_one_per_task_variant_and_condition_in_a_fixed_order():
    samples = ip.build_samples(ALL, variants=("plain", "ambiguous"), conditions=("free", "format"))
    assert len(samples) == len(ALL) * 4
    ids = [s.id for s in samples]
    assert ids == sorted(ids, key=lambda i: (i.split(":")[0], ["plain", "ambiguous"].index(i.split(":")[1]),
                                              ["free", "format"].index(i.split(":")[2])))
    assert len(set(ids)) == len(ids)


def test_each_samples_input_is_exactly_our_conversation():
    for s in ip.build_samples(ALL):
        want = prompts.build_messages(s.metadata["condition"], pilot.spec_text(ip._task(s.metadata["task_id"]), s.metadata["variant"]))
        assert [(m.role, m.text) for m in s.input] == [(w["role"], w["content"]) for w in want]
        assert s.metadata["system_sha256"] == prompts.sha256(want[0]["content"])
        assert s.metadata["user_sha256"] == prompts.sha256(want[1]["content"])
        assert s.id == f"{s.metadata['task_id']}:{s.metadata['variant']}:{s.metadata['condition']}"


def test_the_pilot_covers_all_five_tasks_by_default_and_rejects_unknown_ones():
    assert len(ip.all_task_ids()) == 5
    assert len(ip.pilot_task(runner=LOCAL).dataset) == 10
    with pytest.raises(ValueError, match="nope"):
        ip.pilot_task(["nope"])


def test_the_command_line_entry_point_parses_its_arguments():
    t = ip.pilot(task_ids="t01_movavg,t02_colscale_diff", variants="plain", conditions="free,format", max_attempts=3)
    assert [s.id for s in t.dataset] == ["t01_movavg:plain:free", "t01_movavg:plain:format",
                                         "t02_colscale_diff:plain:free", "t02_colscale_diff:plain:format"]


# ---- the whole path through Inspect, against a fake server

def test_a_correct_model_passes_and_two_requests_are_made_per_sample(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1), extra={"timings": TIMINGS}) as srv:
        log = run(srv, tmp_path)
        assert log.status == "success" and len(srv.requests) == 4 and srv.max_in_flight == 1
        for body in srv.requests:
            assert set(body) == {"messages", "temperature", "seed", "max_tokens"}
            assert (body["temperature"], body["seed"], body["max_tokens"]) == (0.0, 0, 1500)
            assert [m["role"] for m in body["messages"]] == ["system", "user"]
    by_id = {s.id: s for s in log.samples}
    plain, amb = by_id["t01_movavg:plain:free"], by_id["t01_movavg:ambiguous:free"]
    assert plain.scores[ip.SCORER_NAME].value == {"passed": 1, "signature": "pass", "confidence": 90.0,
                                                  "outcome": None, "choice": None}
    assert amb.scores[ip.SCORER_NAME].value["outcome"] == "silent" and amb.scores[ip.SCORER_NAME].value["choice"] == "A"


def test_every_request_holds_only_its_own_spec(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1)) as srv:
        run(srv, tmp_path, task_ids=("t01_movavg", "t02_colscale_diff"))
        specs = {s.input[1].text for s in ip.build_samples([T1, ip._task("t02_colscale_diff")])}
        assert len(srv.requests) == 8
        for body in srv.requests:
            assert user_of(body) in specs and len(body["messages"]) == 2
        users = [user_of(b) for b in srv.requests]
        assert users == [u for u in users[::2] for _ in (0, 1)], "each prompt is sent twice in a row, then the next prompt"


def test_the_log_conversation_is_system_user_and_the_confirmed_reply(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1)) as srv:
        log = run(srv, tmp_path)
    for s in log.samples:
        assert [m.role for m in s.messages] == ["system", "user", "assistant"]
        assert s.output.completion == reply_with(GOLDEN_T1)


def test_a_first_reply_that_differs_is_recorded_and_scored_separately(tmp_path):
    plain_user = ip.build_samples([T1], variants=("plain",))[0].input[1].text
    seen = {}

    def answer(body, n):
        u = user_of(body)
        seen[u] = seen.get(u, 0) + 1
        return reply_with(WRONG_T1 if (u == plain_user and seen[u] == 1) else GOLDEN_T1)

    with FakeServer(answer, extra={"timings": TIMINGS}) as srv:
        log = run(srv, tmp_path, variants=("plain",))
        assert len(srv.requests) == 3
    s = log.samples[0]
    meta = s.scores[ip.SCORER_NAME].metadata
    assert meta["confirmation"]["n_attempts"] == 3 and meta["confirmation"]["first_reply_differs"]
    assert meta["evaluation"]["passed"] and not meta["first_reply_evaluation"]["passed"]
    assert meta["first_reply_evaluation"]["signature"] == "shifted-output"
    assert s.scores[ip.SCORER_NAME].value["passed"] == 1                        # the confirmed reply is the one scored
    assert [a["cache_n"] for a in meta["confirmation"]["attempts"]] == [0, 0, 0]  # the server's own number is kept
    assert len(meta["confirmation"]["replies"]) == 2 and "text" not in meta["confirmation"]["attempts"][0]


def test_a_prompt_that_never_settles_is_marked_unconfirmed(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1 if n % 2 else WRONG_T1)) as srv:
        log = run(srv, tmp_path, variants=("plain",), max_attempts=4)
        assert len(srv.requests) == 4
    conf = log.samples[0].scores[ip.SCORER_NAME].metadata["confirmation"]
    assert not conf["confirmed"] and conf["n_attempts"] == 4


def test_a_question_instead_of_code_is_an_ask_on_the_vague_spec_and_no_code_on_the_plain_one(tmp_path):
    q = "Do you mean the output keeps the input length, or only the full windows?"
    with FakeServer(lambda b, n: q) as srv:
        log = run(srv, tmp_path)
    v = {s.id: s.scores[ip.SCORER_NAME].value for s in log.samples}
    assert v["t01_movavg:ambiguous:free"]["outcome"] == "asked" and v["t01_movavg:ambiguous:free"]["choice"] is None
    assert v["t01_movavg:plain:free"]["passed"] == 0 and v["t01_movavg:plain:free"]["signature"] == "no-code"
    m = metrics(log)
    assert m["outcome_asked"] == 1.0 and m["outcome_silent"] == 0.0 and m["plain_pass_rate"] == 0.0


def test_the_metrics_in_the_log_are_the_ones_we_report(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1, confidence="CONFIDENCE: 80")) as srv:
        log = run(srv, tmp_path)
    m = metrics(log)
    assert m["plain_n"] == 1 and m["plain_pass_rate"] == 1.0 and m["confidence_scored"] == 1
    assert m["brier"] == pytest.approx(0.04) and m["mean_confidence_passing"] == 80.0
    assert m["ambiguous_n"] == 1 and m["outcome_silent"] == 1.0 and m["chose_reading_a"] == 1.0


def test_the_inspect_log_can_be_turned_back_into_our_report(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN_T1)) as srv:
        log = run(srv, tmp_path)
    recs = ip.records_from_log(log)
    assert [(r["task_id"], r["variant"]) for r in recs] == [("t01_movavg", "ambiguous"), ("t01_movavg", "plain")] \
        or {(r["task_id"], r["variant"]) for r in recs} == {("t01_movavg", "plain"), ("t01_movavg", "ambiguous")}
    summary = report.summarise(recs)
    assert summary["n_samples"] == 2 and summary["calibration_plain"].n == 1
    assert "t01_movavg" in report.format_report(summary)


def test_all_five_tasks_with_golden_answers_through_inspect(tmp_path):
    table = pilot.golden_replies(ALL)
    with FakeServer(lambda b, n: table[user_of(b)]) as srv:
        log = run(srv, tmp_path, task_ids=ip.all_task_ids())
        assert len(srv.requests) == 20 and srv.max_in_flight == 1
    assert log.status == "success" and len(log.samples) == 10
    for s in log.samples:
        v = s.scores[ip.SCORER_NAME].value
        assert v["passed"] == 1 and v["confidence"] == 100.0
        if s.metadata["variant"] == "ambiguous":
            assert v["outcome"] == "silent" and v["choice"] == "A"
    m = metrics(log)
    assert m["plain_pass_rate"] == 1.0 and m["plain_n"] == 5 and m["ambiguous_n"] == 5


# ---- metrics on their own

def ss(value, variant):
    return SampleScore(score=Score(value=value), sample_metadata={"variant": variant})


def test_pilot_summary_only_looks_at_plain_samples():
    scores = [ss({"passed": 1, "confidence": 100.0}, "plain"), ss({"passed": 0, "confidence": 100.0}, "plain"),
              ss({"passed": 0, "confidence": 0.0, "outcome": "silent"}, "ambiguous")]
    out = ip.pilot_summary()(scores)
    assert out["plain_n"] == 2 and out["plain_pass_rate"] == 0.5 and out["brier"] == pytest.approx(0.5)
    assert out["mean_confidence_failing"] == 100.0 and out["confidence_scored"] == 2


def test_the_pass_rate_covers_samples_with_no_confidence_too():
    out = ip.pilot_summary()([ss({"passed": 1, "confidence": None}, "plain"), ss({"passed": 0, "confidence": None}, "plain"),
                              ss({"passed": 1, "confidence": None}, "plain"), ss({"passed": 0, "confidence": None}, "plain")])
    assert out["plain_pass_rate"] == 0.5 and out["confidence_scored"] == 0 and out["confidence_missing"] == 4


def test_pilot_summary_counts_missing_confidences_without_scoring_them():
    out = ip.pilot_summary()([ss({"passed": 1, "confidence": None}, "plain"), ss({"passed": 0, "confidence": 50.0}, "plain")])
    assert out["confidence_scored"] == 1 and out["confidence_missing"] == 1 and out["brier"] == pytest.approx(0.25)


def test_ambiguity_summary_rates():
    scores = [ss({"outcome": "silent", "choice": "A"}, "ambiguous"), ss({"outcome": "silent", "choice": "same"}, "ambiguous"),
              ss({"outcome": "asked", "choice": None}, "ambiguous"), ss({"outcome": "stated-assumption", "choice": "A"}, "ambiguous"),
              ss({"passed": 1}, "plain")]
    out = ip.ambiguity_summary()(scores)
    assert out["ambiguous_n"] == 4 and out["outcome_silent"] == 0.5 and out["outcome_asked"] == 0.25
    assert out["outcome_stated_assumption"] == 0.25 and out["outcome_no_answer"] == 0.0
    assert out["chose_reading_a"] == pytest.approx(2 / 3)


def test_metrics_cope_with_nothing_to_summarise():
    assert ip.pilot_summary()([])["plain_n"] == 0
    out = ip.ambiguity_summary()([])
    assert out["ambiguous_n"] == 0 and out["outcome_silent"] is None and out["chose_reading_a"] is None


# ---- with the real sandbox

@pytest.mark.docker
def test_the_default_runner_uses_the_real_sandbox_from_inside_inspect(tmp_path):
    from bench.sandbox import SandboxConfig, docker_available
    if not docker_available(SandboxConfig()):
        pytest.skip("sandbox image not available")
    t3 = ip._task("t03_simpson")
    table = pilot.golden_replies([t3])
    with FakeServer(lambda b, n: table[user_of(b)]) as srv:
        task = ip.pilot_task(["t03_simpson"], variants=("plain",))              # no runner given: Docker
        logs = inspect_eval(task, model="llamaserver/olmo3-test", model_base_url=srv.base_url, log_dir=str(tmp_path),
                            max_samples=1, display="none")
    log = read_eval_log(logs[0].location)
    assert log.status == "success" and log.samples[0].scores[ip.SCORER_NAME].value["passed"] == 1
