import pytest

from bench.confirm import confirm_reply
from bench.warmup import sha256_text

PARAMS = {"temperature": 0.0, "seed": 0, "max_tokens": 100}
MSGS = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def fake_ask(replies, finish="stop"):
    it = iter(replies)
    seen = []

    def ask(url, messages, **params):
        seen.append((url, messages, params))
        text = next(it)
        return {"choices": [{"message": {"content": text}, "finish_reason": finish}],
                "usage": {"completion_tokens": 10, "prompt_tokens": 50},
                "timings": {"prompt_n": 50, "cache_n": 0}}

    ask.seen = seen
    return ask


def test_a_steady_prompt_is_confirmed_by_its_first_two_replies():
    ask = fake_ask(["A", "A", "A"])
    c = confirm_reply("u", MSGS, params=PARAMS, ask=ask)
    assert (c.reply, c.confirmed, c.n_attempts, c.first_reply_differs) == ("A", True, 2, False)
    assert len(ask.seen) == 2


def test_a_first_mode_prompt_needs_a_third_reply_and_remembers_the_first():
    c = confirm_reply("u", MSGS, params=PARAMS, ask=fake_ask(["B", "A", "A", "A"]))
    assert (c.reply, c.confirmed, c.n_attempts) == ("A", True, 3)
    assert c.first_reply == "B" and c.first_reply_differs


def test_two_matching_first_replies_stop_even_if_later_ones_would_differ():
    c = confirm_reply("u", MSGS, params=PARAMS, ask=fake_ask(["B", "B", "A"]))
    assert c.reply == "B" and c.n_attempts == 2 and not c.first_reply_differs


def test_never_confirmed_returns_the_last_reply_and_says_so():
    c = confirm_reply("u", MSGS, params=PARAMS, max_attempts=5, ask=fake_ask(list("ABABAB")))
    assert (c.reply, c.confirmed, c.n_attempts) == ("A", False, 5) and c.first_reply == "A"


def test_the_same_messages_and_params_are_sent_every_time():
    ask = fake_ask(["B", "A", "A"])
    confirm_reply("http://x", MSGS, params=PARAMS, ask=ask)
    assert len(ask.seen) == 3
    for url, messages, params in ask.seen:
        assert url == "http://x" and messages == MSGS and params == PARAMS


def test_every_attempt_is_recorded_with_its_details():
    c = confirm_reply("u", MSGS, params=PARAMS, ask=fake_ask(["abcd", "abxd", "abxd"]))
    a1, a2, a3 = c.attempts
    assert [a.i for a in c.attempts] == [1, 2, 3]
    assert a1.reply_sha256 == sha256_text("abcd") and a1.first_diff_vs_previous is None
    assert a2.first_diff_vs_previous == 2 and a3.first_diff_vs_previous is None
    assert a1.prompt_n == 50 and a1.cache_n == 0 and a1.completion_tokens == 10 and a1.finish_reason == "stop"
    assert a1.latency_s >= 0 and a1.t_utc.endswith("+00:00")


def test_attempt_as_dict_can_leave_out_the_text():
    c = confirm_reply("u", MSGS, params=PARAMS, ask=fake_ask(["A", "A"]))
    assert "text" in c.attempts[0].as_dict() and "text" not in c.attempts[0].as_dict(with_text=False)


def test_truncation_is_visible():
    c = confirm_reply("u", MSGS, params=PARAMS, ask=fake_ask(["cut", "cut"], finish="length"))
    assert c.truncated and c.confirmed


@pytest.mark.parametrize("n", [0, 1, -3])
def test_needs_at_least_two_attempts(n):
    with pytest.raises(ValueError):
        confirm_reply("u", MSGS, params=PARAMS, max_attempts=n, ask=fake_ask(["A"]))


def test_exactly_two_attempts_allowed_is_enough_to_confirm():
    assert confirm_reply("u", MSGS, params=PARAMS, max_attempts=2, ask=fake_ask(["A", "A"])).confirmed
    assert not confirm_reply("u", MSGS, params=PARAMS, max_attempts=2, ask=fake_ask(["B", "A"])).confirmed
