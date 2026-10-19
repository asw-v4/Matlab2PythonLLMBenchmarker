import hashlib
import json
from pathlib import Path

import pytest

from bench import gguf_build as g

COMMIT = "f3fb2fcdbc93a8f677e893b8dbd36462b3190ffd"


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.bin"
    data = b"abc" * 1_000_003
    p.write_bytes(data)
    assert g.sha256_file(p, chunk=4096) == hashlib.sha256(data).hexdigest()


def test_sha256_file_empty(tmp_path):
    p = tmp_path / "empty"
    p.write_bytes(b"")
    assert g.sha256_file(p) == hashlib.sha256(b"").hexdigest()


def _cfg(tmp_path, commit=COMMIT, revision="a" * 40):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({
        "llama_cpp": {"build": "b1", "commit": commit},
        "models": {"m": {"hf_repo": "o/m", "revision": revision, "quant": "Q8_0"}},
    }))
    return p


def test_load_config_ok(tmp_path):
    assert g.load_config(_cfg(tmp_path))["models"]["m"]["quant"] == "Q8_0"


def test_load_config_rejects_short_commit(tmp_path):
    with pytest.raises(ValueError):
        g.load_config(_cfg(tmp_path, commit="f3fb2fcdb"))


def test_load_config_rejects_short_revision(tmp_path):
    with pytest.raises(ValueError):
        g.load_config(_cfg(tmp_path, revision="main"))


def test_repo_config_is_valid():
    cfg = g.load_config(Path(__file__).parent.parent / "configs" / "models.json")
    assert set(cfg["models"]) == {"olmo3-7b-instruct", "olmo3-7b-think"}


def test_llama_source_dir():
    assert g.llama_source_dir(Path("Z:/llama.cpp"), COMMIT) == Path("Z:/llama.cpp/src") / f"llama.cpp-{COMMIT}"


def test_fetch_llama_source_skips_download_when_present(tmp_path, monkeypatch):
    dest = g.llama_source_dir(tmp_path, COMMIT)
    dest.mkdir(parents=True)
    monkeypatch.setattr(g.urllib.request, "urlretrieve", lambda *a, **k: pytest.fail("should not download"))
    assert g.fetch_llama_source(tmp_path, COMMIT) == dest


def test_convert_command():
    cmd = g.convert_command("py", Path("src"), Path("model"), Path("out.gguf"), "bf16")
    assert cmd[0] == "py"
    assert Path(cmd[1]) == Path("src") / "convert_hf_to_gguf.py"
    assert cmd[cmd.index("--outtype") + 1] == "bf16"
    assert Path(cmd[cmd.index("--outfile") + 1]) == Path("out.gguf")


def test_quantize_command():
    assert g.quantize_command(Path("q.exe"), Path("a"), Path("b"), "Q8_0") == ["q.exe", "a", "b", "Q8_0"]


def test_converter_env_prepends_gguf_py_and_keeps_base():
    env = g.converter_env({"PYTHONPATH": "old", "X": "1"}, Path("src"))
    assert env["PYTHONPATH"].split(";")[0] == str(Path("src") / "gguf-py")
    assert env["PYTHONPATH"].endswith("old") and env["X"] == "1"


def test_converter_env_without_existing_pythonpath():
    assert g.converter_env({}, Path("src"))["PYTHONPATH"] == str(Path("src") / "gguf-py")


def test_build_manifest(tmp_path):
    f = tmp_path / "m-Q8_0.gguf"
    f.write_bytes(b"12345")
    m = g.build_manifest(
        name="m", hf_repo="o/m", revision="a" * 40, quant="Q8_0",
        llama_build="b1", llama_commit=COMMIT, backend_build_note="n",
        source_files={"b": "2", "a": "1"}, gguf_path=f, gguf_sha256="deadbeef",
    )
    assert m["output"] == {"file": "m-Q8_0.gguf", "size_bytes": 5, "sha256": "deadbeef"}
    assert list(m["source"]["files_sha256"]) == ["a", "b"]
    assert m["conversion"]["llama_cpp_commit"] == COMMIT
    json.dumps(m)
