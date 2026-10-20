"""Minimal HTTP client for a local llama-server (OpenAI-compatible endpoint)."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


def get_json(url: str, path: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def get_text(url: str, path: str, timeout: float = 30) -> str | None:
    """GET `path` as text, or None if the endpoint is disabled or unreachable."""
    try:
        with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError):
        return None


def chat(url: str, messages: list[dict], *, temperature: float = 0.0, seed: int = 0,
         max_tokens: int = 400, timeout: float = 300) -> dict:
    """One chat completion with every sampling choice passed explicitly."""
    body = {"messages": messages, "temperature": temperature, "seed": seed, "max_tokens": max_tokens}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())
