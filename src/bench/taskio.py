"""Load a task directory: its metadata and the Octave answer key.

Variable names in expected.mat:
    caseNN_in_<arg>                inputs
    caseNN_expected[__<out>]       the reference result (one array per output)
    caseNN_wrong_<name>[__<out>]   a known-wrong translation's result
    caseNN_reading_<name>[__<out>] an alternative reading's result (ambiguity variants)
A task with a single output omits `__<out>`; that output is called "result".
"""
from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.io import loadmat

_VAR = re.compile(r"^case(\d+)_(.+)$")
DEFAULT_OUTPUT = "result"


@dataclass
class Case:
    index: int
    inputs: dict
    expected: dict[str, np.ndarray] = field(default_factory=dict)
    wrong: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)
    readings: dict[str, dict[str, np.ndarray]] = field(default_factory=dict)


@dataclass
class Task:
    meta: dict
    cases: list[Case]
    octave_version: str
    dir: Path | None = None

    @property
    def id(self) -> str:
        return self.meta["task"]["id"]

    @property
    def output_names(self) -> list[str]:
        return list(self.meta.get("outputs", {}).get("names", [DEFAULT_OUTPUT]))

    @property
    def function(self) -> str:
        return self.meta["task"]["function"]

    @property
    def arg_order(self) -> list[str]:
        """The names of the function's arguments, in the order it is called with."""
        return list(self.meta["inputs"]["order"])

    @property
    def tolerance(self) -> dict:
        return self.meta["tolerance"]

    @property
    def tolerance_overrides(self) -> dict[str, dict]:
        """Per-output tolerances, from sub-tables such as [tolerance.c] in task.toml."""
        return {k: dict(v) for k, v in self.meta["tolerance"].items()
                if isinstance(v, dict) and k in self.output_names}

    def compare_kwargs(self) -> dict:
        """Everything compare_outputs needs from this task, apart from the outputs themselves."""
        return dict(names=self.output_names, rtol=self.tolerance["rtol"], atol=self.tolerance["atol"],
                    vector_ok=self.vector_ok, exact=self.exact_outputs, overrides=self.tolerance_overrides)

    @property
    def exact_outputs(self) -> list[str]:
        """Outputs compared with zero tolerance (integers, rounding results)."""
        return list(self.meta["tolerance"].get("exact_outputs", []))

    @property
    def vector_ok(self) -> bool:
        return bool(self.meta.get("shape", {}).get("vector_ok", False))


def _scalar_or_array(value: np.ndarray, name: str, as_int: set[str]):
    if value.dtype.kind in "US":  # text inputs, e.g. the name of a test function
        return str(value.item()) if value.size == 1 else value
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
    names = list(meta.get("outputs", {}).get("names", [DEFAULT_OUTPUT]))

    mat = loadmat(task_dir / "expected.mat", squeeze_me=False)
    cases: dict[int, Case] = {}
    for var, value in mat.items():
        m = _VAR.match(var)
        if not m:
            continue
        case = cases.setdefault(int(m.group(1)), Case(index=int(m.group(1)), inputs={}))
        head, _, out = m.group(2).partition("__")
        out = out or DEFAULT_OUTPUT
        if head.startswith("in_"):
            case.inputs[head[3:]] = _scalar_or_array(value, head[3:], as_int)
        elif head == "expected":
            case.expected[out] = value
        elif head.startswith("wrong_"):
            case.wrong.setdefault(head[6:], {})[out] = value
        elif head.startswith("reading_"):
            case.readings.setdefault(head[8:], {})[out] = value

    ordered = [cases[i] for i in sorted(cases)]
    for c in ordered:
        # the key and every known-wrong variant must give every output; an alternative reading may
        # cover only some (an ambiguity can concern a single output)
        for label, got in [("expected", c.expected), *[(f"wrong_{k}", v) for k, v in c.wrong.items()]]:
            if set(got) != set(names):
                raise ValueError(f"{task_dir.name}: case {c.index} {label} has outputs {sorted(got)}, "
                                 f"expected {sorted(names)}")
        for k, got in c.readings.items():
            if not got or not set(got) <= set(names):
                raise ValueError(f"{task_dir.name}: case {c.index} reading_{k} has outputs {sorted(got)}, "
                                 f"which are not a non-empty subset of {sorted(names)}")
    version = str(mat["octave_version"].item()) if "octave_version" in mat else "unknown"
    return Task(meta=meta, cases=ordered, octave_version=version, dir=task_dir)
