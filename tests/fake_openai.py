"""A fake OpenAI-compatible chat server for tests: records every request and answers from a script.

    with FakeServer(lambda body, n: "reply text") as srv:
        ... point Inspect at srv.base_url ...
        srv.requests   # the JSON body of every request, in arrival order
        srv.max_in_flight   # most requests being handled at the same moment
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeServer:
    def __init__(self, answer, delay: float = 0.0, extra: dict | None = None):
        """`answer(body, n)` returns the reply text for the n-th request (1-based), or a (text, finish_reason) pair.
        `extra` is merged into every response, e.g. llama-server's `timings` and `system_fingerprint`."""
        self.answer, self.delay, self.extra = answer, delay, extra or {}
        self.requests: list[dict] = []
        self.paths: list[str] = []
        self.max_in_flight = 0
        self._in_flight = 0
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with outer._lock:
                    outer.requests.append(body)
                    outer.paths.append(self.path)
                    n = len(outer.requests)
                    outer._in_flight += 1
                    outer.max_in_flight = max(outer.max_in_flight, outer._in_flight)
                try:
                    if outer.delay:
                        time.sleep(outer.delay)
                    out = outer.answer(body, n)
                    text, finish = out if isinstance(out, tuple) else (out, "stop")
                    payload = json.dumps({
                        "id": f"chatcmpl-{n}", "object": "chat.completion", "created": 0,
                        "model": body.get("model", "fake"),
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                     "finish_reason": finish}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                        **outer.extra,
                    }).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    with outer._lock:
                        outer._in_flight -= 1

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_port}/v1"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
