"""Build a quantised GGUF from an official Hugging Face checkpoint.

Build-time tooling only: it downloads weights and llama.cpp source. The
benchmark itself makes no network calls. Every input is pinned (HF revision,
llama.cpp commit) and the output is described by a manifest with SHA-256s.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

LLAMA_CPP_TARBALL = "https://github.com/ggml-org/llama.cpp/archive/{commit}.tar.gz"


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    commit = cfg["llama_cpp"]["commit"]
    if len(commit) != 40:
        raise ValueError(f"llama.cpp commit must be a full 40-char hash, got {commit!r}")
    for name, m in cfg["models"].items():
        if len(m["revision"]) != 40:
            raise ValueError(f"{name}: revision must be a full 40-char hash")
    return cfg


def llama_source_dir(root: Path, commit: str) -> Path:
    return Path(root) / "src" / f"llama.cpp-{commit}"


def fetch_llama_source(root: Path, commit: str) -> Path:
    """Download and extract llama.cpp source at `commit` unless already present."""
    dest = llama_source_dir(root, commit)
    if dest.is_dir():
        return dest
    src_root = dest.parent
    src_root.mkdir(parents=True, exist_ok=True)
    tgz = src_root / f"llama.cpp-{commit}.tar.gz"
    urllib.request.urlretrieve(LLAMA_CPP_TARBALL.format(commit=commit), tgz)
    with tarfile.open(tgz) as tf:
        tf.extractall(src_root, filter="data")
    if not dest.is_dir():
        raise RuntimeError(f"expected {dest} after extracting {tgz}")
    return dest


def convert_command(python: str, src_dir: Path, model_dir: Path, outfile: Path, outtype: str) -> list[str]:
    return [
        str(python),
        str(Path(src_dir) / "convert_hf_to_gguf.py"),
        str(model_dir),
        "--outfile", str(outfile),
        "--outtype", outtype,
    ]


def quantize_command(quantize_exe: Path, src: Path, dst: Path, quant: str) -> list[str]:
    return [str(quantize_exe), str(src), str(dst), quant]


def converter_env(base_env: dict, src_dir: Path) -> dict:
    """Environment that makes the converter use the gguf-py shipped at the pinned commit."""
    env = dict(base_env)
    gguf_py = str(Path(src_dir) / "gguf-py")
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = gguf_py if not existing else gguf_py + ";" + existing
    return env


def build_manifest(
    *, name: str, hf_repo: str, revision: str, quant: str,
    llama_build: str, llama_commit: str, backend_build_note: str,
    source_files: dict[str, str], gguf_path: Path, gguf_sha256: str,
) -> dict:
    return {
        "name": name,
        "source": {"hf_repo": hf_repo, "revision": revision, "files_sha256": dict(sorted(source_files.items()))},
        "conversion": {
            "llama_cpp_commit": llama_commit,
            "llama_cpp_release_build": llama_build,
            "intermediate": "bf16",
            "quant": quant,
            "note": backend_build_note,
        },
        "output": {
            "file": Path(gguf_path).name,
            "size_bytes": Path(gguf_path).stat().st_size,
            "sha256": gguf_sha256,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
