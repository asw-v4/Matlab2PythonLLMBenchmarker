"""The log of a run: one JSON object per line, written as it goes.

    header   everything needed to know exactly what was run
    sample   one per (task, variant, condition)
    summary  the totals

so a crashed run still leaves its finished samples behind.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = 1


class RunLog:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", encoding="utf-8", newline="\n")

    def write(self, record: dict) -> None:
        self._fh.write(json.dumps(record, default=_default) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "RunLog":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def _default(o):
    """numpy scalars and arrays, dataclass-like objects, paths."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"cannot serialise {type(o).__name__}")


def read_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hash_task(task_dir: Path) -> str:
    """One hash over the files that define a task, so a result is tied to the exact task it was scored on."""
    h = hashlib.sha256()
    for name in ("task.toml", "spec.md", "spec_ambiguous.md", "expected.mat"):
        h.update(name.encode() + b"\0")
        h.update((Path(task_dir) / name).read_bytes() + b"\0")
    return h.hexdigest()
