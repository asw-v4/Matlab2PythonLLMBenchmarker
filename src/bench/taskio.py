"""Load a task directory: its metadata and the Octave answer key."""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_VAR = re.compile(r"^case(\d+)_(in_\w+|expected|wrong_\w+|reading_\w+)$")


@dataclass
class Case:
    index: int
    inputs: dict
    expected: np.ndarray
    wrong: dict[str, np.ndarray] = field(default_factory=dict)
    readings: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class Task:
    meta: dict
    cases: list[Case]
    octave_version: str

    @property
    def id(self) -> str:
        return self.meta["task"]["id"]

    @property
    def tolerance(self) -> dict:
        return self.meta["tolerance"]

    @property
    def vector_ok(self) -> bool:
        return bool(self.meta.get("shape", {}).get("vector_ok", False))


def _scalar_or_array(value: np.ndarray, name: str, as_int: set[str]):
    if value.size == 1 and name in as_int:
        return int(value.item())
    if value.size == 1:
        return float(value.item())
    return value


def load_task(task_dir: Path) -> Task:
    task_dir = Path(task_dir)
    with open(task_dir / "task.toml", "rb") as f:
        meta = tomllib.load(f)
    as_int = set(meta.get("inputs", {}).get("as_int", []))

    mat = loadmat(task_dir / "expected.mat", squeeze_me=False)
    cases: dict[int, Case] = {}
    for var, value in mat.items():
        m = _VAR.match(var)
        if not m:
            continue
        idx, kind = int(m.group(1)), m.group(2)
        case = cases.setdefault(idx, Case(index=idx, inputs={}, expected=None))
        if kind.startswith("in_"):
            name = kind[3:]
            case.inputs[name] = _scalar_or_array(value, name, as_int)
        elif kind == "expected":
            case.expected = value
        elif kind.startswith("wrong_"):
            case.wrong[kind[6:]] = value
        else:
            case.readings[kind[8:]] = value

    ordered = [cases[i] for i in sorted(cases)]
    for c in ordered:
        if c.expected is None:
            raise ValueError(f"{task_dir.name}: case {c.index} has no expected output")
    version = str(mat["octave_version"].item()) if "octave_version" in mat else "unknown"
    return Task(meta=meta, cases=ordered, octave_version=version)
