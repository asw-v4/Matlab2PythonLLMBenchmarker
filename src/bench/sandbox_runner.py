"""Runs INSIDE the sandbox container (copied there as /work/run.py). Standard library and numpy only.

Loads the model's code, calls its function on every test case, and prints one result line:
    @@RESULT@@ [{"status": "ok", "outputs": [...]}, {"status": "error", "error": "..."}, ...]
Anything the model's code prints is swallowed. Each case gets freshly decoded inputs, so a function that
changes its arguments cannot affect the next case.

This assumes the code is careless, not adversarial: nothing stops code from printing a fake result line.
"""
import contextlib
import importlib.util
import io
import json
import os
import sys
import traceback

import numpy as np

MARK = "@@RESULT@@"
MAX_ELEMENTS = 1_000_000
OK, ERROR, BAD_OUTPUT = "ok", "error", "bad-output"


def decode(v, callables=None):
    if isinstance(v, dict) and "__ndarray__" in v:
        return np.array(v["data"], dtype=v["dtype"]).reshape(v["shape"])
    if isinstance(v, dict) and "__callable__" in v:
        return callables[v["__callable__"]]
    return v


def encode(a):
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"returned {a.dtype}, not numbers")
    if a.size > MAX_ELEMENTS:
        raise TypeError("returned too many values")
    return {"__ndarray__": True, "dtype": a.dtype.name, "shape": list(a.shape), "data": a.ravel().tolist()}


def to_outputs(res, n_outputs):
    """The returned value as a list of arrays: one per expected output."""
    if n_outputs == 1:
        if isinstance(res, tuple) and len(res) == 1:
            res = res[0]
        return [res]
    if isinstance(res, (tuple, list)):
        return list(res)
    return [res]


def short_error(exc, code_file):
    where = ""
    tb = traceback.extract_tb(exc.__traceback__)
    for frame in reversed(tb):
        if frame.filename == code_file:
            where = f" (line {frame.lineno})"
            break
    return f"{type(exc).__name__}: {str(exc)[:300]}{where}"[:500]


def load_callables(workdir):
    path = os.path.join(workdir, "task_functions.py")
    spec = importlib.util.spec_from_file_location("task_functions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.FUNCS


def execute(workdir):
    with open(os.path.join(workdir, "payload.json"), encoding="utf-8") as f:
        payload = json.load(f)
    code_file = os.path.join(workdir, "code.py")
    cases = payload["cases"]
    callables = load_callables(workdir) if payload.get("uses_callables") else None
    noise = io.StringIO()

    try:
        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            spec = importlib.util.spec_from_file_location("model_code", code_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, payload["function"])
    except BaseException as exc:          # includes SystemExit from exit() in the model's code
        return [{"status": ERROR, "error": short_error(exc, code_file)} for _ in cases]

    results = []
    for case in cases:
        try:
            args = [decode(case[name], callables) for name in payload["arg_order"]]
            with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
                res = fn(*args)
        except BaseException as exc:
            results.append({"status": ERROR, "error": short_error(exc, code_file)})
            continue
        try:
            results.append({"status": OK, "outputs": [encode(o) for o in to_outputs(res, payload["n_outputs"])]})
        except Exception as exc:
            results.append({"status": BAD_OUTPUT, "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
    return results


if __name__ == "__main__":
    out = execute(os.path.dirname(os.path.abspath(__file__)))
    sys.__stdout__.write(MARK + json.dumps(out) + "\n")
    sys.__stdout__.flush()
