"""What running a model's function on one test case produced."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OK = "ok"
ERROR = "error"          # the code raised, or could not be loaded
TIMEOUT = "timeout"      # the sandbox killed it
BAD_OUTPUT = "bad-output"  # it returned something that is not numbers


@dataclass
class CaseRun:
    status: str
    outputs: list[np.ndarray] | None = None   # one array per returned value, when status is "ok"
    error: str | None = None

    @classmethod
    def ok(cls, outputs) -> "CaseRun":
        return cls(OK, [np.asarray(o) for o in outputs])

    @classmethod
    def failed(cls, status: str, error: str) -> "CaseRun":
        return cls(status, None, error)

    def as_result(self):
        """The value in the form compare_outputs takes: a tuple of arrays."""
        return tuple(self.outputs or ())
