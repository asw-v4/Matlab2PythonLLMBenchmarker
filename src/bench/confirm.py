"""Per-prompt confirmation: ask until two replies in a row are identical, and score that reply.

A fresh llama-server answered some prompts differently the first time than on later repeats, even at
temperature 0 (docs/warmup_study_notes.md). So every prompt is sent until two consecutive replies match,
the reply that repeats is the one scored, and every attempt is logged. If the first reply differs from the
confirmed one, the caller can score it too and report both.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Callable

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


def confirm_reply(url: str, messages: list[dict], *, params: dict, max_attempts: int = 6,
                  ask: Callable = chat) -> Confirmed:
    """Send `messages` until two consecutive replies are identical (at most `max_attempts` times).

    If that never happens, the last reply is returned with confirmed=False.
    """
    if max_attempts < 2:
        raise ValueError("need at least 2 attempts to confirm anything")
    attempts: list[Attempt] = []
    for i in range(1, max_attempts + 1):
        t_utc, t0 = utc_now(), time.perf_counter()
        resp = ask(url, messages, **params)
        latency = time.perf_counter() - t0
        text = resp["choices"][0]["message"]["content"]
        usage, timings = resp.get("usage", {}), resp.get("timings", {})
        attempts.append(Attempt(
            i=i, t_utc=t_utc, latency_s=round(latency, 4), text=text, reply_sha256=sha256_text(text),
            finish_reason=resp["choices"][0].get("finish_reason"),
            completion_tokens=usage.get("completion_tokens"), prompt_tokens=usage.get("prompt_tokens"),
            prompt_n=timings.get("prompt_n"), cache_n=timings.get("cache_n"),
            first_diff_vs_previous=first_diff_index(attempts[-1].text, text) if attempts else None,
        ))
        if i >= 2 and attempts[-1].text == attempts[-2].text:
            return Confirmed(text, True, tuple(attempts), attempts[0].text)
    return Confirmed(attempts[-1].text, False, tuple(attempts), attempts[0].text)
