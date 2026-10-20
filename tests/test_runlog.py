import json
from pathlib import Path

import numpy as np
import pytest

from bench.runlog import RunLog, hash_task, read_log

TASKS = Path(__file__).parent.parent / "tasks"


def test_records_are_written_one_per_line_and_flushed_as_they_go(tmp_path):
    p = tmp_path / "sub" / "run.jsonl"
    log = RunLog(p)
    log.write({"type": "header", "x": 1})
    assert p.read_text(encoding="utf-8").count("\n") == 1          # readable before close
    log.write({"type": "sample", "text": "line one\nline two"})
    log.close()
    recs = read_log(p)
    assert [r["type"] for r in recs] == ["header", "sample"] and recs[1]["text"] == "line one\nline two"
    assert "\r" not in p.read_text(encoding="utf-8")


def test_works_as_a_context_manager_and_creates_folders(tmp_path):
    with RunLog(tmp_path / "a" / "b" / "r.jsonl") as log:
        log.write({"n": 1})
    assert read_log(tmp_path / "a" / "b" / "r.jsonl") == [{"n": 1}]


def test_numpy_values_and_paths_are_serialisable(tmp_path):
    with RunLog(tmp_path / "r.jsonl") as log:
        log.write({"a": np.float64(1.5), "b": np.int64(3), "c": np.array([1, 2]), "d": tmp_path, "e": np.bool_(True)})
    r = read_log(tmp_path / "r.jsonl")[0]
    assert r["a"] == 1.5 and r["b"] == 3 and r["c"] == [1, 2] and r["e"] is True and isinstance(r["d"], str)


def test_unserialisable_objects_fail_loudly(tmp_path):
    with RunLog(tmp_path / "r.jsonl") as log, pytest.raises(TypeError):
        log.write({"x": object()})


def test_task_hash_is_stable_and_changes_with_the_task(tmp_path):
    a = hash_task(TASKS / "t01_movavg")
    assert a == hash_task(TASKS / "t01_movavg") and len(a) == 64
    assert a != hash_task(TASKS / "t02_colscale_diff")


def test_task_hash_changes_when_a_spec_changes(tmp_path):
    for name in ("task.toml", "spec.md", "spec_ambiguous.md", "expected.mat"):
        (tmp_path / name).write_bytes((TASKS / "t01_movavg" / name).read_bytes())
    before = hash_task(tmp_path)
    (tmp_path / "spec.md").write_text("changed", encoding="utf-8")
    assert hash_task(tmp_path) != before
