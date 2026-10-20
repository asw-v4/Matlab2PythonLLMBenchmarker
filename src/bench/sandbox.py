"""Run model-written code in a throwaway Linux container (Docker Engine in WSL2).

One container per sample. It has no network, a read-only filesystem apart from a small tmpfs, no
capabilities, a non-root user, and limits on memory, CPUs, processes and time. The inputs, the code and
the runner are mounted read-only; the results come back on stdout.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from bench.runs import BAD_OUTPUT, ERROR, TIMEOUT, CaseRun

MARK = "@@RESULT@@"
REPO = Path(__file__).resolve().parents[2]
RUNNER = Path(__file__).with_name("sandbox_runner.py")


@dataclass(frozen=True)
class SandboxConfig:
    distro: str = "Ubuntu-Bench"
    image: str = "matlab2py-sandbox:1"
    memory: str = "512m"
    cpus: str = "1"
    pids: int = 128
    timeout_s: float = 60.0
    workroot: Path = REPO / "runs" / "sandbox"


def to_wsl(path) -> str:
    """Z:\\Coding\\x -> /mnt/z/Coding/x"""
    s = str(Path(path).resolve())
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", s)
    if not m:
        raise ValueError(f"not a Windows drive path: {s}")
    return f"/mnt/{m.group(1).lower()}/" + m.group(2).replace("\\", "/")


def encode_value(v, *, names_are_callables: bool = False):
    if isinstance(v, np.ndarray):
        return {"__ndarray__": True, "dtype": v.dtype.name, "shape": list(v.shape), "data": v.ravel().tolist()}
    if isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        return {"__callable__": v} if names_are_callables else v
    raise TypeError(f"cannot send a {type(v).__name__} to the sandbox")


def build_payload(task) -> dict:
    uses_callables = task.dir is not None and (Path(task.dir) / "functions.py").exists()
    return {
        "function": task.function,
        "arg_order": task.arg_order,
        "n_outputs": len(task.output_names),
        "uses_callables": uses_callables,
        "cases": [{name: encode_value(c.inputs[name], names_are_callables=uses_callables) for name in task.arg_order}
                  for c in task.cases],
    }


def prepare_workdir(root: Path, code: str, task) -> Path:
    work = Path(root) / uuid.uuid4().hex[:12]
    work.mkdir(parents=True)
    payload = build_payload(task)
    (work / "code.py").write_text(code, encoding="utf-8", newline="\n")
    (work / "payload.json").write_text(json.dumps(payload), encoding="utf-8")
    shutil.copyfile(RUNNER, work / "run.py")
    if payload["uses_callables"]:
        shutil.copyfile(Path(task.dir) / "functions.py", work / "task_functions.py")
    return work


def docker_command(cfg: SandboxConfig, container_name: str, wsl_workdir: str) -> list[str]:
    return [
        "wsl", "-d", cfg.distro, "--exec", "docker", "run", "--rm", "--name", container_name,
        "--network", "none",
        "--memory", cfg.memory, "--memory-swap", cfg.memory,
        "--cpus", cfg.cpus, "--pids-limit", str(cfg.pids),
        "--read-only", "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m",
        "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
        "--user", "65534:65534",
        "-v", f"{wsl_workdir}:/work:ro", "-w", "/work",
        cfg.image, "python", "-I", "/work/run.py",
    ]


def _clean(data: bytes) -> str:
    return data.decode("utf-8", errors="replace").replace("\x00", "")


def _tail(text: str, n: int = 300) -> str:
    text = text.strip()
    return text[-n:] if len(text) > n else text


def parse_output(stdout: str, returncode: int, stderr: str, n_cases: int) -> list[CaseRun]:
    line = next((ln for ln in reversed(stdout.splitlines()) if ln.startswith(MARK)), None)
    if line is None:
        if returncode == 137:
            msg = "killed: out of memory or another resource limit"
        else:
            msg = f"no result (exit {returncode}): {_tail(stderr)}"
        return [CaseRun.failed(ERROR, msg) for _ in range(n_cases)]
    try:
        items = json.loads(line[len(MARK):])
    except json.JSONDecodeError as exc:
        return [CaseRun.failed(ERROR, f"unreadable result: {exc}") for _ in range(n_cases)]
    if len(items) != n_cases:
        return [CaseRun.failed(ERROR, f"expected {n_cases} results, got {len(items)}") for _ in range(n_cases)]
    runs = []
    for it in items:
        if it["status"] == "ok":
            runs.append(CaseRun.ok([np.array(o["data"], dtype=o["dtype"]).reshape(o["shape"]) for o in it["outputs"]]))
        else:
            status = ERROR if it["status"] == "error" else BAD_OUTPUT
            runs.append(CaseRun.failed(status, it.get("error", "")))
    return runs


def run_code(code: str, task, cfg: SandboxConfig | None = None, *, keep: bool = False) -> list[CaseRun]:
    """Run `code` (Python source defining task.function) on every case of `task`, in a container."""
    cfg = cfg or SandboxConfig()
    n = len(task.cases)
    cfg.workroot.mkdir(parents=True, exist_ok=True)
    work = prepare_workdir(cfg.workroot, code, task)
    name = "m2p-" + uuid.uuid4().hex[:12]
    try:
        try:
            proc = subprocess.run(docker_command(cfg, name, to_wsl(work)), capture_output=True, timeout=cfg.timeout_s)
        except subprocess.TimeoutExpired:
            subprocess.run(["wsl", "-d", cfg.distro, "--exec", "docker", "rm", "-f", name],
                           capture_output=True, timeout=60)
            return [CaseRun.failed(TIMEOUT, f"no result within {cfg.timeout_s:g} s") for _ in range(n)]
        return parse_output(_clean(proc.stdout), proc.returncode, _clean(proc.stderr), n)
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


@lru_cache(maxsize=4)
def docker_available(cfg: SandboxConfig | None = None) -> bool:
    """True if the sandbox image exists in the WSL distro's Docker."""
    cfg = cfg or SandboxConfig()
    try:
        r = subprocess.run(["wsl", "-d", cfg.distro, "--exec", "docker", "image", "inspect", cfg.image],
                           capture_output=True, timeout=60)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def image_id(cfg: SandboxConfig | None = None) -> str | None:
    cfg = cfg or SandboxConfig()
    try:
        r = subprocess.run(["wsl", "-d", cfg.distro, "--exec", "docker", "image", "inspect",
                            "--format", "{{.Id}}", cfg.image], capture_output=True, timeout=60)
        return _clean(r.stdout).strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None
