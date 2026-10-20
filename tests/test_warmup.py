import json
from pathlib import Path

import pytest

from bench import warmup as w

TASKS = Path(__file__).parent.parent / "tasks"


def fake_ask(replies):
    """An `ask` that returns the given reply texts in order."""
    it = iter(replies)

    def ask(url, messages, **params):
        text = next(it)
        return {
            "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"completion_tokens": len(text), "prompt_tokens": 100},
            "timings": {"prompt_n": 100, "cache_n": 0, "prompt_per_second": 50.0, "predicted_per_second": 70.0},
        }

    return ask


# ---- small pure functions

def test_sha256_text_known_value():
    assert w.sha256_text("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_first_diff_index():
    assert w.first_diff_index("abc", "abc") is None
    assert w.first_diff_index("abcd", "abxd") == 2
    assert w.first_diff_index("abc", "abcdef") == 3  # one is a prefix of the other
    assert w.first_diff_index("", "a") == 0


def test_trailing_run():
    assert w.trailing_run([]) == (0, 0)
    assert w.trailing_run(["a"]) == (1, 1)
    assert w.trailing_run(["b", "b", "a", "a", "a"]) == (3, 3)
    assert w.trailing_run(["a", "a", "a"]) == (1, 3)
    assert w.trailing_run(["a", "b", "a"]) == (3, 1)  # equal values must be contiguous


def test_fingerprint_letters_by_first_appearance():
    assert w.fingerprint(["x", "x", "y", "y", "y"]) == "AABBB"
    assert w.fingerprint(["y", "x", "y"]) == "ABA"
    assert w.fingerprint([]) == ""


# ---- the canary itself

def test_canary_is_stable_and_hashed():
    assert w.canary_sha() == w.canary_sha()
    assert len(w.canary_sha()) == 64
    msgs = w.canary_messages()
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_canary_sha_changes_if_prompt_changes(monkeypatch):
    before = w.canary_sha()
    monkeypatch.setattr(w, "CANARY_USER", w.CANARY_USER + " ")
    assert w.canary_sha() != before


def test_canary_is_not_a_benchmark_task():
    text = (w.CANARY_SYSTEM + w.CANARY_USER).lower()
    for name in ("movavg", "colscale", "simpson", "robust_stats", "rk4_fit"):
        assert name not in text
    for spec in TASKS.glob("*/spec*.md"):
        assert spec.read_text(encoding="utf-8") not in w.CANARY_USER


def test_canary_uses_fixed_greedy_sampling():
    assert w.CANARY_PARAMS == {"temperature": 0.0, "seed": 0, "max_tokens": 400}


# ---- the warm-up loop

def test_stops_when_replies_stabilise_and_reports_transient():
    # B B then A forever (the pattern seen on a fresh server)
    summary, recs = w.run_warmup("u", ask=fake_ask(["B", "B"] + ["A"] * 20), min_requests=6, stable_run=4)
    assert summary["ok"] is True
    assert summary["n_requests"] == 6
    assert summary["fingerprint"] == "AABBBB"
    assert summary["stable_from"] == 3 and summary["transient_requests"] == 2
    assert summary["distinct_replies"] == 2
    assert summary["first_vs_final_diff"] == 0
    assert [r["i"] for r in recs] == [1, 2, 3, 4, 5, 6]


def test_does_not_stop_before_min_requests_even_if_stable():
    summary, _ = w.run_warmup("u", ask=fake_ask(["A"] * 20), min_requests=6, stable_run=4)
    assert summary["n_requests"] == 6 and summary["ok"] and summary["transient_requests"] == 0


def test_gives_up_at_n_max_when_replies_never_stabilise():
    summary, recs = w.run_warmup("u", ask=fake_ask(["A", "B"] * 10), n_max=8, min_requests=6, stable_run=4)
    assert summary["ok"] is False and summary["n_requests"] == 8 == len(recs)
    assert summary["stable_run_length"] == 1


def test_late_change_is_not_called_stable():
    # three identical, then a change: the early run must not count
    summary, _ = w.run_warmup("u", ask=fake_ask(["B"] * 5 + ["A"] * 6), n_max=12, min_requests=6, stable_run=4)
    assert summary["ok"] and summary["stable_from"] == 6 and summary["n_requests"] == 9


@pytest.mark.parametrize("kwargs", [dict(stable_run=0), dict(stable_run=7, min_requests=6), dict(min_requests=13)])
def test_rejects_inconsistent_settings(kwargs):
    with pytest.raises(ValueError):
        w.run_warmup("u", ask=fake_ask([]), **kwargs)


def test_records_hold_full_reply_hash_and_timings():
    _, recs = w.run_warmup("u", ask=fake_ask(["hello"] * 6), min_requests=6, stable_run=4)
    r = recs[0]
    assert r["reply"] == "hello" and r["reply_sha256"] == w.sha256_text("hello")
    assert r["cache_n"] == 0 and r["prompt_n"] == 100 and r["finish_reason"] == "stop"
    assert r["first_diff_vs_previous"] is None


def test_first_diff_vs_previous_is_recorded():
    _, recs = w.run_warmup("u", ask=fake_ask(["abcd", "abxd"] + ["abxd"] * 8), min_requests=6, stable_run=4)
    assert recs[1]["first_diff_vs_previous"] == 2 and recs[2]["first_diff_vs_previous"] is None


def test_sink_receives_every_record_then_summary_in_order():
    got = []
    summary, recs = w.run_warmup("u", ask=fake_ask(["A"] * 10), sink=got.append, min_requests=6, stable_run=4)
    assert [g["type"] for g in got] == ["request"] * 6 + ["summary"]
    assert got[-1] is summary


def test_metrics_are_stored_when_provided_and_none_otherwise():
    _, recs = w.run_warmup("u", ask=fake_ask(["A"] * 6), metrics=lambda: "m 1", min_requests=6, stable_run=4)
    assert recs[0]["metrics_raw"] == "m 1"
    _, recs = w.run_warmup("u", ask=fake_ask(["A"] * 6), metrics=lambda: None, min_requests=6, stable_run=4)
    assert recs[0]["metrics_raw"] is None


# ---- header

def test_extract_props_keeps_useful_fields_and_hashes_the_template():
    props = {"build_info": "b1-abc", "model_path": "m.gguf", "total_slots": 1, "chat_template": "TPL",
             "default_generation_settings": {"n_ctx": 8192, "params": {"temperature": 0.6}}}
    p = w.extract_props(props)
    assert p["build_info"] == "b1-abc" and p["n_ctx"] == 8192 and p["sampler_defaults"] == {"temperature": 0.6}
    assert p["chat_template_sha256"] == w.sha256_text("TPL") and "TPL" not in json.dumps(p)


def test_extract_props_tolerates_missing_fields():
    p = w.extract_props({})
    assert p["n_ctx"] is None and p["chat_template_sha256"] is None


def test_header_is_json_serialisable_and_labelled():
    h = w.build_header(url="u", props={"build_info": "b"}, git={"commit": "c", "dirty": False},
                       context={"gguf_sha256": "x"}, note="fresh", settings={"n_max": 12})
    json.dumps(h)
    assert h["type"] == "header" and h["schema_version"] == w.SCHEMA_VERSION
    assert h["canary_sha256"] == w.canary_sha() and h["context"] == {"gguf_sha256": "x"} and h["note"] == "fresh"


def test_header_without_props_or_context():
    h = w.build_header(url="u", props=None, git=None, context=None, note=None, settings={})
    assert h["server"] is None and h["context"] == {}


def test_collect_git_info_on_this_repo_and_on_a_non_repo(tmp_path):
    info = w.collect_git_info(Path(__file__).parent.parent)
    assert info is None or (len(info["commit"]) == 40 and isinstance(info["dirty"], bool))
    assert w.collect_git_info(tmp_path / "nope") is None
