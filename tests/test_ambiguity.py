from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from bench import ambiguity as amb
from bench.runs import ERROR, CaseRun
from bench.taskio import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())
TASKS = {p.name: load_task(p) for p in TASK_DIRS}
IDS = sorted(TASKS)


def outputs_with(task, case, reading=None):
    """The key's outputs, with those a reading concerns replaced by the reading's values."""
    out = dict(case.expected)
    if reading:
        out.update(case.readings[reading])
    return CaseRun.ok([out[n] for n in task.output_names])


# ---- choices, on every real task

@pytest.mark.parametrize("tid", IDS)
def test_the_key_chooses_reading_a_for_every_ambiguous_output(tid):
    task = TASKS[tid]
    choices = amb.output_choices([outputs_with(task, c) for c in task.cases], task)
    assert choices and set(choices.values()) == {"A"}
    assert amb.summary_label(choices) == "A"


@pytest.mark.parametrize("tid", IDS)
def test_each_reading_is_recognised_for_the_outputs_it_covers_and_a_for_the_rest(tid):
    task = TASKS[tid]
    readings = set(task.meta["readings"])
    for r in readings:
        covered = set(task.cases[0].readings[r])
        choices = amb.output_choices([outputs_with(task, c, r) for c in task.cases], task)
        for name, choice in choices.items():
            assert choice == (r if name in covered else "A"), (tid, r, name, choice)
        assert amb.summary_label(choices) in (r, "mixed")


def test_a_reading_covering_every_ambiguous_output_gives_one_summary_label():
    task = TASKS["t01_movavg"]                                     # a single output
    choices = amb.output_choices([outputs_with(task, c, "same") for c in task.cases], task)
    assert choices == {"result": "same"} and amb.summary_label(choices) == "same"


def test_two_readings_at_once_are_reported_per_output_and_summarised_as_mixed():
    task = TASKS["t04_robust_stats"]
    runs = []
    for c in task.cases:
        out = dict(c.expected)
        out.update(c.readings["half_even"])         # r the numpy way
        out.update(c.readings["wraparound"])        # sat the numpy way
        runs.append(CaseRun.ok([out[n] for n in task.output_names]))
    choices = amb.output_choices(runs, task)
    assert choices["r"] == "half_even" and choices["sat"] == "wraparound"
    assert choices["s"] == choices["v"] == choices["m"] == choices["rm"] == "A"
    assert amb.summary_label(choices) == "mixed"


def test_t5_c_has_two_readings_and_each_is_found():
    task = TASKS["t05_rk4_fit"]
    runs = [outputs_with(task, c, "coef_highest_first") for c in task.cases]
    choices = amb.output_choices(runs, task)
    assert choices == {"Y": "A", "c": "coef_highest_first", "yflat": "A"}
    runs = [outputs_with(task, c, "no_initial") for c in task.cases]
    assert set(amb.output_choices(runs, task).values()) == {"no_initial"}


def test_something_matching_neither_reading_is_other():
    task = TASKS["t01_movavg"]
    runs = [CaseRun.ok([c.expected["result"] + 1.0]) for c in task.cases]
    assert amb.output_choices(runs, task) == {"result": "other"}


def test_code_that_did_not_run_is_error_for_every_output():
    task = TASKS["t04_robust_stats"]
    runs = [CaseRun.failed(ERROR, "boom") for _ in task.cases]
    assert set(amb.output_choices(runs, task).values()) == {"error"}


def test_returning_the_wrong_number_of_values_is_error():
    task = TASKS["t02_colscale_diff"]
    runs = [CaseRun.ok([c.expected["B"]]) for c in task.cases]
    assert amb.output_choices(runs, task) == {"B": "error", "D": "error"}


def test_two_alignments_of_the_same_length_reading_are_told_apart_only_by_the_even_windows():
    task = TASKS["t01_movavg"]
    assert amb.readings_for(task, "result") == ["same", "same_numpy"]
    for reading in ("same", "same_numpy"):
        runs = [outputs_with(task, c, reading) for c in task.cases]
        assert amb.output_choices(runs, task) == {"result": reading}
    # if every window were odd the two readings would be indistinguishable: restrict to the odd cases
    odd = [c for c in task.cases if c.inputs["k"] % 2 == 1 and c.inputs["k"] > 1]
    assert odd and all(c.readings["same"]["result"].tolist() == c.readings["same_numpy"]["result"].tolist() for c in odd)


def test_a_failure_on_an_uninformative_case_does_not_count():
    # T1 reading "same" is only informative where k > 1. A crash on the k = 1 case (case 4) must not matter.
    task = TASKS["t01_movavg"]
    runs = [outputs_with(task, c, "same") for c in task.cases]
    assert task.cases[3].inputs["k"] == 1
    runs[3] = CaseRun.failed(ERROR, "boom")
    assert amb.output_choices(runs, task) == {"result": "same"}


# ---- which cases tell the readings apart

def test_informative_cases_for_each_task():
    expect = {
        ("t01_movavg", "result", "same"): [1, 2, 3, 5],
        ("t01_movavg", "result", "same_numpy"): [1, 2, 3, 5],
        ("t02_colscale_diff", "B", "global"): [1, 2, 5],
        ("t02_colscale_diff", "D", "global"): [1, 2, 5],
        ("t03_simpson", "result", "panels"): [2, 3, 4, 5],
        ("t04_robust_stats", "s", "population"): [1, 2, 3, 4, 5],
        ("t04_robust_stats", "r", "half_even"): [1, 3, 4],
        ("t04_robust_stats", "m", "truncated_modulo"): [1, 4, 5],
        ("t04_robust_stats", "rm", "floored_remainder"): [1, 4, 5],
        ("t04_robust_stats", "sat", "wraparound"): [1, 4, 5],
        ("t05_rk4_fit", "Y", "no_initial"): [1, 2, 3, 4, 5],
        ("t05_rk4_fit", "c", "coef_highest_first"): [1, 2, 3, 4, 5],
        ("t05_rk4_fit", "yflat", "row_by_row"): [1, 2, 3, 4, 5],
    }
    for (tid, name, reading), cases in expect.items():
        got = [c.index for c in amb.informative_cases(TASKS[tid], name, reading)]
        assert got == cases, (tid, name, reading)


def test_readings_for_lists_every_reading_that_covers_an_output():
    t5 = TASKS["t05_rk4_fit"]
    assert amb.readings_for(t5, "c") == ["coef_highest_first", "no_initial"]
    assert amb.readings_for(t5, "Y") == ["no_initial"]
    assert amb.readings_for(TASKS["t04_robust_stats"], "v") == ["population"]


def test_an_output_with_no_informative_case_is_not_informative(tmp_path):
    R = np.array([[1.0, 2.0]])
    (tmp_path / "task.toml").write_text('[task]\nid = "t"\nfunction = "f"\n[tolerance]\nrtol = 1e-9\natol = 1e-9\n',
                                        encoding="utf-8")
    savemat(tmp_path / "expected.mat", {"case01_expected": R, "case01_reading_alt": R.copy()})
    task = load_task(tmp_path)
    choices = amb.output_choices([CaseRun.ok([R])], task)
    assert choices == {"result": amb.NOT_INFORMATIVE} and amb.summary_label(choices) == amb.NOT_INFORMATIVE


# ---- outcome from the reply's behaviour

@pytest.mark.parametrize("has_code,asks,states,label", [
    (False, True, False, "asked"), (False, True, True, "asked"),
    (False, False, False, "no-answer"), (False, False, True, "no-answer"),
    (True, True, False, "asked-and-proceeded"), (True, True, True, "asked-and-proceeded"),
    (True, False, True, "stated-assumption"), (True, False, False, "silent"),
])
def test_outcome_truth_table(has_code, asks, states, label):
    assert amb.outcome(has_code=has_code, asks_clarifying_question=asks, states_assumption=states) == label


def test_summary_label_ignores_uninformative_outputs():
    assert amb.summary_label({"a": "A", "b": amb.NOT_INFORMATIVE}) == "A"
    assert amb.summary_label({"a": "A", "b": "x"}) == "mixed"
    assert amb.summary_label({}) == amb.NOT_INFORMATIVE
