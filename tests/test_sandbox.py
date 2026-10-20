"""Sandbox logic that doesn't need Docker: the runner is executed in-process on the same work folder."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from bench import sandbox as sb
from bench import sandbox_runner as runner
from bench.runs import BAD_OUTPUT, ERROR, OK, TIMEOUT, CaseRun
from bench.signatures import classify_sample
from bench.taskio import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())
TASKS = {p.name: load_task(p) for p in TASK_DIRS}
IDS = sorted(TASKS)
T1 = TASKS["t01_movavg"]


def run_local(code, task, tmp_path):
    work = sb.prepare_workdir(tmp_path, code, task)
    out = runner.execute(str(work))
    return sb.parse_output(sb.MARK + json.dumps(out) + "\n", 0, "", len(task.cases))


# ---- paths

@pytest.mark.parametrize("win,wsl", [
    (r"Z:\Coding\Portfolio\x", "/mnt/z/Coding/Portfolio/x"),
    (r"C:\Users\award\a b\c", "/mnt/c/Users/award/a b/c"),
    ("Z:/Coding/y", "/mnt/z/Coding/y"),
])
def test_to_wsl(win, wsl):
    assert sb.to_wsl(win) == wsl


def test_to_wsl_rejects_paths_without_a_drive_letter():
    with pytest.raises(ValueError):
        sb.to_wsl(r"\\server\share\x")


# ---- encoding

def test_array_roundtrip_through_the_runner_decoder():
    for a in [np.arange(6.0).reshape(2, 3), np.array([[250, 1]], dtype=np.uint8), np.zeros((0, 5)),
              np.array(3.5), np.array([[np.nan, np.inf, -np.inf]]), np.array([True, False])]:
        enc = json.loads(json.dumps(sb.encode_value(a)))
        back = runner.decode(enc)
        assert back.dtype == a.dtype and back.shape == a.shape
        np.testing.assert_array_equal(back, a)


def test_scalars_pass_through_and_text_becomes_a_callable_name_only_when_asked():
    assert sb.encode_value(3) == 3 and sb.encode_value(0.5) == 0.5
    assert sb.encode_value("cubic") == "cubic"
    assert sb.encode_value("cubic", names_are_callables=True) == {"__callable__": "cubic"}
    with pytest.raises(TypeError):
        sb.encode_value([1, 2])


@pytest.mark.parametrize("tid", IDS)
def test_payload_describes_the_call(tid):
    task = TASKS[tid]
    p = sb.build_payload(task)
    json.dumps(p)                                                      # serialisable
    assert p["function"] == task.function and p["arg_order"] == task.arg_order
    assert p["n_outputs"] == len(task.output_names) and len(p["cases"]) == len(task.cases)
    assert all(set(c) == set(task.arg_order) for c in p["cases"])
    assert p["uses_callables"] == (tid == "t03_simpson")


def test_workdir_has_code_payload_runner_and_task_functions_when_needed(tmp_path):
    w = sb.prepare_workdir(tmp_path, "def movavg(x, k):\n    return x\n", T1)
    assert {p.name for p in w.iterdir()} == {"code.py", "payload.json", "run.py"}
    assert (w / "run.py").read_text() == sb.RUNNER.read_text()
    w3 = sb.prepare_workdir(tmp_path, "def simpson(f, a, b, n):\n    return 0.0\n", TASKS["t03_simpson"])
    assert (w3 / "task_functions.py").exists()


def test_each_workdir_is_new(tmp_path):
    a = sb.prepare_workdir(tmp_path, "x = 1", T1)
    b = sb.prepare_workdir(tmp_path, "x = 1", T1)
    assert a != b


# ---- the docker command

def test_docker_command_applies_every_restriction():
    cmd = sb.docker_command(sb.SandboxConfig(), "m2p-test", "/mnt/z/w")
    s = " ".join(cmd)
    assert cmd[:6] == ["wsl", "-d", "Ubuntu-Bench", "--exec", "docker", "run"]    # --exec: no shell re-parsing
    for needle in ("--rm", "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
                   "--user 65534:65534", "--pids-limit 128", "--memory 512m", "--memory-swap 512m", "--cpus 1",
                   "-v /mnt/z/w:/work:ro", "python -I /work/run.py", "--name m2p-test"):
        assert needle in s, needle
    assert "--privileged" not in s and "--network host" not in s and "--network bridge" not in s
    assert s.count(" -v ") == 1                                                      # only the one read-only mount


# ---- reading results

def test_parse_output_ok_and_failures():
    items = [{"status": "ok", "outputs": [{"dtype": "float64", "shape": [1, 2], "data": [1.0, 2.0]}]},
             {"status": "error", "error": "ValueError: no"},
             {"status": "bad-output", "error": "TypeError: strings"}]
    runs = sb.parse_output("noise\n" + sb.MARK + json.dumps(items) + "\n", 0, "", 3)
    assert [r.status for r in runs] == [OK, ERROR, BAD_OUTPUT]
    np.testing.assert_array_equal(runs[0].outputs[0], [[1.0, 2.0]])
    assert runs[1].error == "ValueError: no"


def test_parse_output_uses_the_last_result_line_and_ignores_noise():
    good = [{"status": "ok", "outputs": [{"dtype": "int64", "shape": [], "data": [7]}]}]
    runs = sb.parse_output(sb.MARK + "[]\nprinted by the model\n" + sb.MARK + json.dumps(good) + "\n", 0, "", 1)
    assert runs[0].outputs[0] == 7


@pytest.mark.parametrize("rc,stderr,expect", [
    (137, "", "killed"), (1, "Traceback\nboom", "boom"), (0, "", "no result"),
])
def test_parse_output_without_a_result_line(rc, stderr, expect):
    runs = sb.parse_output("nothing useful", rc, stderr, 2)
    assert len(runs) == 2 and all(r.status == ERROR and expect in r.error for r in runs)


def test_parse_output_rejects_garbled_or_miscounted_results():
    assert sb.parse_output(sb.MARK + "{not json", 0, "", 2)[0].error.startswith("unreadable result")
    assert "expected 2 results" in sb.parse_output(sb.MARK + "[]", 0, "", 2)[0].error


def test_nan_and_empty_arrays_survive_the_round_trip():
    items = [{"status": "ok", "outputs": [{"dtype": "float64", "shape": [0, 5], "data": []},
                                          {"dtype": "float64", "shape": [1, 2], "data": [float("nan"), 1.0]}]}]
    r = sb.parse_output(sb.MARK + json.dumps(items) + "\n", 0, "", 1)[0]
    assert r.outputs[0].shape == (0, 5) and np.isnan(r.outputs[1][0, 0])


# ---- the golden solutions, through the runner, then classified

@pytest.mark.parametrize("tid", IDS)
def test_every_golden_solution_passes_through_the_whole_pipeline(tid, tmp_path):
    task = TASKS[tid]
    code = (task.dir / "solution.py").read_text(encoding="utf-8")
    runs = run_local(code, task, tmp_path)
    assert [r.status for r in runs] == [OK] * len(task.cases)
    res = classify_sample(runs, task)
    assert res.passed, [c.detail for c in res.cases]


# ---- what can go wrong in the model's code

def test_a_function_that_raises(tmp_path):
    runs = run_local("def movavg(x, k):\n    raise ValueError('nope')\n", T1, tmp_path)
    assert all(r.status == ERROR and "ValueError: nope" in r.error and "line 2" in r.error for r in runs)


def test_a_syntax_error_fails_every_case(tmp_path):
    runs = run_local("def movavg(x, k)\n    return x\n", T1, tmp_path)
    assert all(r.status == ERROR and "SyntaxError" in r.error for r in runs)


def test_a_missing_function_fails_every_case(tmp_path):
    runs = run_local("def other(x, k):\n    return x\n", T1, tmp_path)
    assert all(r.status == ERROR and "AttributeError" in r.error for r in runs)


def test_an_import_that_fails(tmp_path):
    runs = run_local("import no_such_module\ndef movavg(x, k):\n    return x\n", T1, tmp_path)
    assert all(r.status == ERROR and "ModuleNotFoundError" in r.error for r in runs)


def test_exit_in_the_models_code_is_an_error_not_a_crash(tmp_path):
    runs = run_local("import sys\ndef movavg(x, k):\n    sys.exit(3)\n", T1, tmp_path)
    assert all(r.status == ERROR and "SystemExit" in r.error for r in runs)


def test_printing_is_swallowed_and_does_not_break_the_result(tmp_path, capsys):
    code = "import numpy as np\ndef movavg(x, k):\n    print('hello', flush=True)\n    return np.array([[1.0]])\n"
    runs = run_local(code, T1, tmp_path)
    assert all(r.status == OK for r in runs)
    assert "hello" not in capsys.readouterr().out


def test_an_error_in_one_case_does_not_stop_the_others(tmp_path):
    code = ("import numpy as np\n"
            "def movavg(x, k):\n"
            "    if k == 1:\n"
            "        raise RuntimeError('only k=1')\n"
            "    return np.array([[1.0]])\n")
    statuses = [r.status for r in run_local(code, T1, tmp_path)]
    assert statuses == [OK, OK, OK, ERROR, OK]


@pytest.mark.parametrize("body", ["return 'text'", "return None", "return 1+2j", "return [[1, 'a']]",
                                  "return np.array(['a', 'b'])", "return {'a': 1}"])
def test_returning_something_that_is_not_numbers_is_bad_output(body, tmp_path):
    runs = run_local(f"import numpy as np\ndef movavg(x, k):\n    {body}\n", T1, tmp_path)
    assert all(r.status == BAD_OUTPUT for r in runs), [r.error for r in runs]


def test_a_huge_result_is_rejected(tmp_path):
    runs = run_local("import numpy as np\ndef movavg(x, k):\n    return np.zeros(2_000_000)\n", T1, tmp_path)
    assert all(r.status == BAD_OUTPUT and "too many" in r.error for r in runs)


def test_plain_lists_and_python_numbers_are_accepted_as_numbers(tmp_path):
    runs = run_local("def movavg(x, k):\n    return [[1.0, 2.0]]\n", T1, tmp_path)
    assert all(r.status == OK for r in runs)
    runs = run_local("def simpson(f, a, b, n):\n    return 1.5\n", TASKS["t03_simpson"], tmp_path)
    assert all(r.status == OK and r.outputs[0] == 1.5 for r in runs)


def test_a_one_tuple_is_unwrapped_for_a_single_output_task(tmp_path):
    runs = run_local("import numpy as np\ndef movavg(x, k):\n    return (np.array([[1.0]]),)\n", T1, tmp_path)
    assert all(len(r.outputs) == 1 for r in runs)


def test_a_bare_array_for_a_multi_output_task_gives_one_output(tmp_path):
    t2 = TASKS["t02_colscale_diff"]
    runs = run_local("import numpy as np\ndef colscale_diff(A):\n    return A\n", t2, tmp_path)
    assert all(r.status == OK and len(r.outputs) == 1 for r in runs)
    assert classify_sample(runs, t2).signature == "wrong-output-count"


def test_a_naive_numpy_translation_of_task_4_is_labelled_by_its_first_wrong_output(tmp_path):
    t4 = TASKS["t04_robust_stats"]
    naive = ("import numpy as np\n"
             "def robust_stats(x, a, b, u, w):\n"
             "    return np.std(x), np.var(x), np.round(x), a % b, np.fmod(a, b), u + w\n")
    res = classify_sample(run_local(naive, t4, tmp_path), t4)
    assert not res.passed and res.signature == "value-mismatch"      # the commonest label: s is one number
    assert res.n_failed == len(t4.cases)
    # case 2 is the plain one, where the naive version is exactly the `ddof0` known-wrong translation;
    # on the others several things are wrong at once, so the first wrong output (s) is named
    by_case = {i + 1: c for i, c in enumerate(res.cases)}
    assert by_case[2].signature == "scale-mismatch" and "ddof0" in by_case[2].detail
    for i in (1, 3, 4, 5):
        assert by_case[i].signature == "value-mismatch" and by_case[i].detail.startswith("s: ")


def test_task_3_receives_real_callables(tmp_path):
    code = ("def simpson(f, a, b, n):\n"
            "    import numpy as np\n"
            "    return float(f(np.array([2.0])).item())\n")
    runs = run_local(code, TASKS["t03_simpson"], tmp_path)
    assert all(r.status == OK for r in runs)
    assert [round(float(r.outputs[0]), 6) for r in runs] == [8.0, round(float(np.exp(-4.0)), 6),
                                                              round(float(np.sin(2.0)), 6), 16.0, 0.2]


# ---- run_code's own handling (no container needed)

def test_a_timeout_returns_timeout_for_every_case_kills_the_container_and_cleans_up(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        if "run" in cmd:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(sb.subprocess, "run", fake_run)
    cfg = sb.SandboxConfig(workroot=tmp_path, timeout_s=1)
    runs = sb.run_code("def movavg(x, k):\n    pass\n", T1, cfg)
    assert [r.status for r in runs] == [TIMEOUT] * 5 and "1 s" in runs[0].error
    assert any(c[-3:-1] == ["rm", "-f"] for c in calls), "the container must be removed after a timeout"
    assert list(tmp_path.iterdir()) == [], "the work folder must be removed"


def test_workdir_is_removed_after_a_normal_run_and_kept_on_request(tmp_path, monkeypatch):
    monkeypatch.setattr(sb.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 1, b"", b"boom"))
    cfg = sb.SandboxConfig(workroot=tmp_path)
    runs = sb.run_code("x = 1", T1, cfg)
    assert all("boom" in r.error for r in runs) and list(tmp_path.iterdir()) == []
    sb.run_code("x = 1", T1, cfg, keep=True)
    assert len(list(tmp_path.iterdir())) == 1
