"""The sandbox for real: model code runs in a container and the restrictions are checked by trying to break them.

Needs the image from scripts/build_sandbox.ps1; skipped if Docker or the image isn't there.
Skip these for a fast run:  pytest -m "not docker"
"""
import subprocess
import textwrap
from pathlib import Path

import pytest

from bench import sandbox as sb
from bench.runs import ERROR, OK, TIMEOUT
from bench.signatures import classify_sample
from bench.taskio import load_task

CFG = sb.SandboxConfig(timeout_s=90)

pytestmark = [pytest.mark.docker,
              pytest.mark.skipif(not sb.docker_available(CFG), reason="sandbox image not available in WSL Docker")]

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())
TASKS = {p.name: load_task(p) for p in TASK_DIRS}
T1 = TASKS["t01_movavg"]


def probe(body: str) -> str:
    """Code for Task 1's function that returns [[1.0]] if `flag` ends up set by the body, else [[0.0]]."""
    return ("import numpy as np\n"
            "def movavg(x, k):\n"
            "    flag = 0.0\n"
            + textwrap.indent(textwrap.dedent(body).strip("\n"), "    ") + "\n"
            "    return np.array([[flag]])\n")


def assert_blocked(body: str):
    runs = sb.run_code(probe(body), T1, CFG)
    assert [r.status for r in runs] == [OK] * 5, [r.error for r in runs]
    assert [float(r.outputs[0].item()) for r in runs] == [1.0] * 5, "the restriction did not hold"


@pytest.mark.parametrize("tid", sorted(TASKS))
def test_each_golden_solution_passes_in_the_real_container(tid):
    task = TASKS[tid]
    runs = sb.run_code((task.dir / "solution.py").read_text(encoding="utf-8"), task, CFG)
    assert [r.status for r in runs] == [OK] * len(task.cases), [r.error for r in runs]
    res = classify_sample(runs, task)
    assert res.passed, [c.detail for c in res.cases]


def test_a_known_wrong_translation_gets_its_label_through_the_container():
    t4 = TASKS["t04_robust_stats"]
    code = ("import numpy as np\n"
            "def robust_stats(x, a, b, u, w):\n"
            "    s = np.std(x, ddof=1); v = np.var(x, ddof=1)\n"
            "    r = np.sign(x) * np.floor(np.abs(x) + 0.5)\n"
            "    return s, v, r, np.mod(a, b), np.fmod(a, b), u + w\n")           # only the uint8 sum is wrong
    res = classify_sample(sb.run_code(code, t4, CFG), t4)
    assert not res.passed and res.signature == "wraparound-mismatch"
    assert [c.passed for c in res.cases] == [False, True, True, False, False]


def test_the_container_runs_the_same_numpy_as_the_lock_file():
    import numpy
    assert_blocked("import numpy\nflag = 1.0 if numpy.__version__ == %r else 0.0" % numpy.__version__)


# ---- the restrictions, by trying to break them

def test_there_is_no_network():
    assert_blocked("""
        import socket
        try:
            socket.create_connection(("1.1.1.1", 53), timeout=3)
        except OSError:
            flag = 1.0
    """)


def test_there_is_no_dns_either():
    assert_blocked("""
        import socket
        try:
            socket.gethostbyname("example.com")
        except OSError:
            flag = 1.0
    """)


def test_the_work_folder_and_the_system_are_read_only():
    # /var/tmp is world-writable in the image, so only --read-only stops a non-root user writing there;
    # /work is a bind mount, so only the :ro on the mount stops writes there.
    assert_blocked("""
        ok = 0
        for path in ("/work/x.txt", "/var/tmp/x.txt", "/etc/x.txt", "/usr/x.txt", "/root/x.txt"):
            try:
                open(path, "w").write("x")
            except OSError:
                ok += 1
        flag = 1.0 if ok == 5 else 0.0
    """)


def test_a_small_tmp_is_writable_but_not_executable():
    assert_blocked("""
        import os, stat
        p = "/tmp/note.txt"
        open(p, "w").write("hi")
        flag = 1.0 if open(p).read() == "hi" else 0.0
        try:
            os.chmod(p, 0o755)
            os.execv(p, [p])
            flag = 0.0
        except OSError:
            pass
    """)


def test_the_user_is_not_root():
    assert_blocked("import os\nflag = 1.0 if os.getuid() == 65534 and os.getgid() == 65534 else 0.0")


def test_the_host_filesystem_is_not_visible():
    assert_blocked("""
        import os
        flag = 1.0 if not any(os.path.exists(p) for p in ("/mnt/z", "/mnt/c", "/Users", "/mnt/wsl")) else 0.0
    """)


def test_there_are_no_capabilities():
    assert_blocked("""
        status = open("/proc/self/status").read()
        eff = [l for l in status.splitlines() if l.startswith("CapEff")][0].split()[1]
        flag = 1.0 if int(eff, 16) == 0 else 0.0
    """)


def test_the_process_limit_stops_a_thread_bomb():
    assert_blocked("""
        import threading, time
        threads, blocked = [], False
        try:
            for _ in range(400):
                t = threading.Thread(target=time.sleep, args=(1.0,))
                t.start()
                threads.append(t)
        except RuntimeError:
            blocked = True
        flag = 1.0 if blocked else 0.0
    """)


def test_blas_runs_on_one_thread():
    assert_blocked("import os\nflag = 1.0 if os.environ.get('OPENBLAS_NUM_THREADS') == '1' and os.environ.get('OMP_NUM_THREADS') == '1' else 0.0")


def test_the_memory_limit_kills_a_runaway_allocation():
    runs = sb.run_code("import numpy as np\ndef movavg(x, k):\n    a = np.ones(400_000_000)\n    return a\n", T1, CFG)
    assert all(r.status == ERROR for r in runs), [r.status for r in runs]


def test_a_runaway_loop_times_out_and_the_container_is_removed():
    cfg = sb.SandboxConfig(timeout_s=8)
    runs = sb.run_code("def movavg(x, k):\n    while True:\n        pass\n", T1, cfg)
    assert [r.status for r in runs] == [TIMEOUT] * 5
    left = subprocess.run(["wsl", "-d", cfg.distro, "--exec", "docker", "ps", "-a", "--filter", "name=m2p-",
                           "--format", "{{.Names}}"], capture_output=True, timeout=60)
    assert left.stdout.decode().replace("\x00", "").strip() == "", "a container was left behind"


def test_no_work_folders_are_left_behind():
    sb.run_code("def movavg(x, k):\n    raise ValueError()\n", T1, CFG)
    assert not CFG.workroot.exists() or not any(CFG.workroot.iterdir())


def test_a_failure_in_the_code_comes_back_as_an_error_not_a_crash():
    runs = sb.run_code("def movavg(x, k):\n    return 1 / 0\n", T1, CFG)
    assert all(r.status == ERROR and "ZeroDivisionError" in r.error for r in runs)


def test_the_models_prints_do_not_disturb_the_result():
    code = ("import numpy as np, sys\n"
            "print('@@RESULT@@ not real')\n"
            "def movavg(x, k):\n"
            "    print('noise'); sys.stderr.write('more noise\\n')\n"
            "    return np.array([[2.0]])\n")
    runs = sb.run_code(code, T1, CFG)
    assert all(r.status == OK and r.outputs[0][0, 0] == 2.0 for r in runs)
