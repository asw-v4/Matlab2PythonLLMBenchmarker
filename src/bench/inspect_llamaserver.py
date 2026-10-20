"""An Inspect model provider for a local llama-server, with no OpenAI SDK.

    model="llamaserver/olmo3-7b-instruct"      the part after the slash is only a label for the logs
    model_base_url="http://127.0.0.1:8080/v1"  or set LLAMASERVER_BASE_URL (default http://127.0.0.1:8080)

It sends one plain chat-completions request through bench.llama_client, with exactly the messages, temperature,
seed and max_tokens it is given and nothing else. It refuses to run unless all three of those are set, so a
forgotten setting can't quietly fall back to the server's own defaults (which are not temperature 0).
The server's `timings` (including cache_n, which shows whether anything came from its prompt cache) and its
build fingerprint are kept in the output's metadata.

Importing this module registers the provider with Inspect.
"""
from __future__ import annotations

import asyncio
import os
import threading

from inspect_ai.model import GenerateConfig, ModelAPI, ModelOutput, ModelUsage, modelapi

from bench.llama_client import chat

# Inspect runs many requests at once by default (adaptive concurrency starting at 20, which overrides a
# provider's own max_connections), so the HTTP call itself is serialised here. Order is still only
# guaranteed by running one sample at a time (eval(max_samples=1)).
_ONE_AT_A_TIME = threading.Lock()


def send(url: str, messages: list[dict], **kwargs) -> dict:
    with _ONE_AT_A_TIME:
        return chat(url, messages, **kwargs)


ENV_BASE_URL = "LLAMASERVER_BASE_URL"
DEFAULT_URL = "http://127.0.0.1:8080"
PROVIDER = "llamaserver"
STOP_REASONS = {"stop": "stop", "length": "max_tokens"}


def server_root(base_url: str | None) -> str:
    """'http://host:8080/v1/' -> 'http://host:8080' (llama_client adds the path itself)."""
    url = (base_url or os.environ.get(ENV_BASE_URL) or DEFAULT_URL).rstrip("/")
    return url[:-3].rstrip("/") if url.endswith("/v1") else url


def to_messages(input) -> list[dict]:
    """Inspect chat messages -> plain {"role", "content"} dicts. Text only; no tools or tool results."""
    out = []
    for m in input:
        if m.role not in ("system", "user", "assistant"):
            raise ValueError(f"llamaserver cannot send a {m.role!r} message")
        out.append({"role": m.role, "content": m.text})
    return out


def required_settings(config: GenerateConfig) -> tuple[float, int, int]:
    missing = [name for name in ("temperature", "seed", "max_tokens") if getattr(config, name) is None]
    if missing:
        raise ValueError(f"set {', '.join(missing)} explicitly in GenerateConfig: the llama-server defaults are not "
                         "temperature 0, and every setting that affects the reply has to be written down")
    return config.temperature, config.seed, config.max_tokens


def output_from_response(model_name: str, resp: dict) -> ModelOutput:
    choice = resp["choices"][0]
    finish = choice.get("finish_reason")
    out = ModelOutput.from_content(model=model_name, content=choice["message"]["content"] or "",
                                   stop_reason=STOP_REASONS.get(finish, "unknown"))
    usage = resp.get("usage") or {}
    if usage:
        out.usage = ModelUsage(input_tokens=usage.get("prompt_tokens"), output_tokens=usage.get("completion_tokens"),
                               total_tokens=usage.get("total_tokens"))
    out.metadata = {"finish_reason": finish, "timings": resp.get("timings"),
                    "system_fingerprint": resp.get("system_fingerprint")}
    return out


@modelapi(name=PROVIDER)
def llamaserver():
    class LlamaServerAPI(ModelAPI):
        def __init__(self, model_name: str, base_url: str | None = None, api_key: str | None = None,
                     config: GenerateConfig = GenerateConfig(), **model_args):
            super().__init__(model_name=model_name, base_url=base_url, api_key=api_key, config=config)
            self.url = server_root(base_url)
            self.timeout = float(model_args.get("timeout", 600))

        def max_connections(self) -> int:
            return 1                       # a hint only; the lock in send() is what guarantees it

        async def generate(self, input, tools, tool_choice, config):
            if tools:
                raise ValueError("llamaserver does not support tools")
            temperature, seed, max_tokens = required_settings(config)
            resp = await asyncio.to_thread(send, self.url, to_messages(input), temperature=temperature, seed=seed,
                                           max_tokens=max_tokens, timeout=self.timeout)
            return output_from_response(self.model_name, resp)

    return LlamaServerAPI
