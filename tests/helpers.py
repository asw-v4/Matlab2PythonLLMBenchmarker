"""Shared by tests. run_local runs model code IN THIS PROCESS through the sandbox runner: fast, no Docker,
and only for the test code written in this repository. Never use it on model output."""
import json
import tempfile
from pathlib import Path

from bench import sandbox as sb
from bench import sandbox_runner as runner


def run_local(code, task, cfg=None):
    with tempfile.TemporaryDirectory() as tmp:
        work = sb.prepare_workdir(Path(tmp), code, task)
        out = runner.execute(str(work))
    return sb.parse_output(sb.MARK + json.dumps(out) + "\n", 0, "", len(task.cases))


def reply_with(code: str, prose_before: str = "", prose_after: str = "", confidence: str = "CONFIDENCE: 90") -> str:
    return f"{prose_before}\n\n```python\n{code}\n```\n\n{prose_after}\n\n{confidence}\n"
