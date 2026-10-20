import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from bench.llama_client import chat, get_json, get_text


class Handler(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/props":
            self._send(200, json.dumps({"build_info": "b-test"}))
        elif self.path == "/metrics":
            self._send(501, "not implemented")
        else:
            self._send(404, "no")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.seen.append((self.path, body))
        self._send(200, json.dumps({"choices": [{"message": {"content": "hi"}}]}))

    def _send(self, code, text):
        data = text.encode()
        self.send_response(code)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def url():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    Handler.seen.clear()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_get_json(url):
    assert get_json(url, "/props") == {"build_info": "b-test"}


def test_get_text_returns_none_for_disabled_or_missing_endpoints(url):
    assert get_text(url, "/metrics") is None
    assert get_text(url, "/nothing") is None


def test_get_text_returns_none_when_server_is_down():
    assert get_text("http://127.0.0.1:9", "/props", timeout=1) is None


def test_chat_sends_every_sampling_choice_explicitly(url):
    out = chat(url + "/", [{"role": "user", "content": "x"}], temperature=0.0, seed=7, max_tokens=5)
    assert out["choices"][0]["message"]["content"] == "hi"
    path, body = Handler.seen[0]
    assert path == "/v1/chat/completions"
    assert body["temperature"] == 0.0 and body["seed"] == 7 and body["max_tokens"] == 5
    assert body["messages"] == [{"role": "user", "content": "x"}]
