"""Per-prompt confirmation: ask until two replies in a row are identical, and score that reply.

A fresh llama-server answered some prompts differently the first time than on later repeats, even at
temperature 0 (docs/warmup_study_notes.md). So every prompt is sent until two consecutive replies match,
the reply that repeats is the one scored, and every attempt is logged. If the first reply differs from the
confirmed one, the caller can score it too and report both.

Two entry points share one tally: confirm_reply (plain HTTP, used by scripts/run_pilot.py) and
confirm_reply_async (a coroutine that fetches replies however it likes, used inside Inspect).
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Awaitable, Callable

from bench.llama_client import chat
from bench.warmup import first_diff_index, sha256_text, utc_now


@dataclass(frozen=True)
class Attempt:
    i: int
    t_utc: str
    latency_s: float
    text: str
    reply_sha256: str
    finish_reason: str | None
    completion_tokens: int | None
    prompt_tokens: int | None
    prompt_n: int | None
    cache_n: int | None
    first_diff_vs_previous: int | None

    def as_dict(self, *, with_text: bool = True) -> dict:
        d = asdict(self)
        if not with_text:
            d.pop("text")
        return d


@dataclass(frozen=True)
class Confirmed:
    reply: str                          # the reply to score
    confirmed: bool                     # two replies in a row matched
    attempts: tuple[Attempt, ...]
    first_reply: str

    @property
    def n_attempts(self) -> int:
        return len(self.attempts)

    @property
    def first_reply_differs(self) -> bool:
        return self.first_reply != self.reply

    @property
    def truncated(self) -> bool:
        return self.attempts[-1].finish_reason == "length"


def reply_from_response(resp: dict) -> dict:
    """The fields we keep from a raw llama-server chat-completions response."""
    usage, timings = resp.get("usage", {}), resp.get("timings", {})
    return {"text": resp["choices"][0]["message"]["content"], "finish_reason": resp["choices"][0].get("finish_reason"),
            "completion_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_n": timings.get("prompt_n"), "cache_n": timings.get("cache_n")}


class _Tally:
    def __init__(self, max_attempts: int):
        if max_attempts < 2:
            raise ValueError("need at least 2 attempts to confirm anything")
        self.max_attempts = max_attempts
        self.attempts: list[Attempt] = []

    def add(self, t_utc: str, latency: float, reply: dict) -> bool:
        """Record an attempt. True once two in a row match."""
        text = reply["text"]
        self.attempts.append(Attempt(
            i=len(self.attempts) + 1, t_utc=t_utc, latency_s=round(latency, 4), text=text,
            reply_sha256=sha256_text(text), finish_reason=reply.get("finish_reason"),
            completion_tokens=reply.get("completion_tokens"), prompt_tokens=reply.get("prompt_tokens"),
            prompt_n=reply.get("prompt_n"), cache_n=reply.get("cache_n"),
            first_diff_vs_previous=first_diff_index(self.attempts[-1].text, text) if self.attempts else None))
        return len(self.attempts) >= 2 and self.attempts[-1].text == self.attempts[-2].text

    def result(self, confirmed: bool) -> Confirmed:
        a = self.attempts
        return Confirmed(a[-1].text, confirmed, tuple(a), a[0].text)


def confirm_reply(url: str, messages: list[dict], *, params: dict, max_attempts: int = 6,
                  ask: Callable = chat) -> Confirmed:
    """Send `messages` until two consecutive replies are identical (at most `max_attempts` times).

    If that never happens, the last reply is returned with confirmed=False.
    """
    tally = _Tally(max_attempts)
    for _ in range(max_attempts):
        t_utc, t0 = utc_now(), time.perf_counter()
        resp = ask(url, messages, **params)
        if tally.add(t_utc, time.perf_counter() - t0, reply_from_response(resp)):
            return tally.result(True)
    return tally.result(False)


async def confirm_reply_async(next_reply: Callable[[], Awaitable[dict]], *, max_attempts: int = 6) -> Confirmed:
    """Like confirm_reply, but `next_reply()` fetches each reply. It returns a dict with a "text" key and,
    optionally, finish_reason, completion_tokens, prompt_tokens, prompt_n and cache_n."""
    tally = _Tally(max_attempts)
    for _ in range(max_attempts):
        t_utc, t0 = utc_now(), time.perf_counter()
        reply = await next_reply()
        if tally.add(t_utc, time.perf_counter() - t0, reply):
            return tally.result(True)
    return tally.result(False)
