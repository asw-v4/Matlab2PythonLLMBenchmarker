import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers import reply_with, run_local  # noqa: E402

from bench.evaluate import AMBIGUOUS, PLAIN, evaluate_reply  # noqa: E402
from bench.report import format_report, summarise  # noqa: E402
from bench.taskio import load_task  # noqa: E402

TASKS_DIR = Path(__file__).parent.parent / "tasks"
T1 = load_task(TASKS_DIR / "t01_movavg")
GOLDEN = (TASKS_DIR / "t01_movavg" / "solution.py").read_text()
DROP_LAST = ("import numpy as np\n"
             "def movavg(x, k):\n"
             "    n = x.shape[1]\n"
             "    return np.array([[x[0, i:i + k].mean() for i in range(n - k)]])\n")
SAME = ("import numpy as np\n"
        "def movavg(x, k):\n"
        "    return np.convolve(x.ravel(), np.ones(k) / k, mode='same').reshape(1, -1)\n")


def sample(reply, variant=PLAIN, confirmed=True, differs=False):
    e = evaluate_reply(reply, T1, variant=variant, condition="free", runner=run_local)
    return {"type": "sample", "evaluation": e.as_dict(),
            "confirmation": {"confirmed": confirmed, "first_reply_differs": differs}}


def test_counts_passes_and_names_the_failures():
    s = summarise([sample(reply_with(GOLDEN)), sample(reply_with(DROP_LAST)), sample(reply_with(DROP_LAST)),
                   sample("no idea")])
    g = s["groups"][("t01_movavg", "plain", "free")]
    assert g["n"] == 4 and g["passed"] == 1
    assert g["signatures"] == {"shifted-output": 2, "no-code": 1}


def test_ambiguity_groups_report_outcomes_and_choices_per_output():
    s = summarise([
        sample(reply_with(GOLDEN), AMBIGUOUS),
        sample(reply_with(SAME), AMBIGUOUS),
        sample(reply_with(SAME, prose_after="I assumed the output keeps the input length."), AMBIGUOUS),
        sample("Do you mean the full windows only, or the same length as the input?", AMBIGUOUS),
    ])
    g = s["groups"][("t01_movavg", "ambiguous", "free")]
    assert g["outcomes"] == {"silent": 2, "stated-assumption": 1, "asked": 1}
    assert g["choices"] == {"A": 1, "same_numpy": 2}
    assert g["per_output_choices"] == {"result": {"A": 1, "same_numpy": 2}}


def test_calibration_uses_plain_samples_only():
    s = summarise([sample(reply_with(GOLDEN, confidence="CONFIDENCE: 100")),
                   sample(reply_with(DROP_LAST, confidence="CONFIDENCE: 100")),
                   sample(reply_with(SAME, confidence="CONFIDENCE: 0"), AMBIGUOUS)])
    c = s["calibration_plain"]
    assert c.n == 2 and c.pass_rate == 0.5 and c.brier == pytest.approx(0.5)
    assert c.mean_confidence_when_failing == 100 and c.mean_confidence_when_passing == 100


def test_confirmation_trouble_is_counted():
    s = summarise([sample(reply_with(GOLDEN), differs=True), sample(reply_with(GOLDEN), confirmed=False),
                   sample(reply_with(GOLDEN))])
    g = s["groups"][("t01_movavg", "plain", "free")]
    assert g["first_reply_differed"] == 1 and g["unconfirmed"] == 1


def test_groups_are_separate_per_condition():
    a = sample(reply_with(GOLDEN))
    b = sample(reply_with(GOLDEN))
    b["evaluation"]["condition"] = "format"
    assert len(summarise([a, b])["groups"]) == 2


def test_empty_run():
    s = summarise([])
    assert s["n_samples"] == 0 and s["groups"] == {} and s["calibration_plain"].n == 0
    assert "samples: 0" in format_report(s)


def test_the_text_report_mentions_each_group_and_the_calibration_lines():
    s = summarise([sample(reply_with(GOLDEN)), sample(reply_with(DROP_LAST)),
                   sample(reply_with(SAME), AMBIGUOUS, differs=True)])
    text = format_report(s)
    assert "t01_movavg" in text and "plain" in text and "ambiguous" in text
    assert "shifted-output x1" in text and "silent x1" in text and "chose: same_numpy x1" in text
    assert "1 differed" in text
    assert "calibration (plain specs only)" in text and "Brier" in text and "skill" in text


def test_report_handles_missing_calibration_numbers():
    text = format_report(summarise([sample("no idea")]))        # no confidence at all
    assert "scored 0" in text and "Brier -" in text


DOCUMENTED = ("import numpy as np\n"
              "def movavg(x, k):\n"
              "    \"\"\"Returns: array of length n - k + 1\"\"\"\n"
              "    n = x.shape[1]\n"
              "    return np.array([[x[0, i:i + k].mean() for i in range(n - k + 1)]])\n")


def test_the_report_counts_documentation_in_code_and_quiet_replies():
    s = summarise([sample(reply_with(GOLDEN), AMBIGUOUS), sample(reply_with(DOCUMENTED), AMBIGUOUS),
                   sample("Do you mean the full windows only, or the same length as the input?", AMBIGUOUS)])
    g = s["groups"][("t01_movavg", "ambiguous", "free")]
    assert g["with_code"] == 2 and g["documented_in_code"] == 1 and g["quiet"] == 1 and g["flagged_in_code"] == 0
    text = format_report(s)
    assert "documented 1/2" in text and "quiet 1/2" in text and "flagged 0/2" in text


def test_the_report_leaves_the_code_line_out_when_no_ambiguous_reply_had_code():
    text = format_report(summarise([sample("Do you mean the full windows only, or the same length?", AMBIGUOUS)]))
    assert "in the code's comments" not in text
