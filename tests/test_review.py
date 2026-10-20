import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from fake_openai import FakeServer  # noqa: E402
from helpers import reply_with, run_local  # noqa: E402

from inspect_ai import eval as inspect_eval  # noqa: E402

from bench import inspect_pilot as ip  # noqa: E402
from bench import pilot, review  # noqa: E402
from bench.runlog import RunLog  # noqa: E402

T1 = ip._task("t01_movavg")
GOLDEN = (T1.dir / "solution.py").read_text()
LOCAL = lambda code, task: run_local(code, task)  # noqa: E731


def sample(sid, outcome, reply="", **extra):
    return review.ReviewSample(sid, sid.split(":")[0], "ambiguous", "free", reply, {"outcome": outcome, **extra})


# ---- the review sheet

def test_code_blocks_are_collapsed_to_a_placeholder():
    text = "Before.\n\n```python\ndef f():\n    return 1\n    # x\n```\n\nAfter.\nCONFIDENCE: 80"
    out = review.redact_code(text)
    assert "def f" not in out and "[code block, 3 lines]" in out and "Before." in out and "After." in out


def test_an_unclosed_block_is_marked_and_two_blocks_are_both_collapsed():
    out = review.redact_code("A\n```python\nx = 1\n```\nB\n```python\ndef f(x):\n    return x +")
    assert out.count("[code block") == 2 and "never closed" in out and "x = 1" not in out and "return x" not in out


def test_text_without_code_is_unchanged_apart_from_trimming():
    assert review.redact_code("  Do you mean n or n-1?\n") == "Do you mean n or n-1?"


def test_the_sheet_shows_the_rule_decision_the_matched_words_and_a_place_for_the_hand_label():
    s = sample("t01_movavg:ambiguous:free", "stated-assumption", reply_with(GOLDEN, prose_after="I assumed full windows."),
               assumption_cues=["assumed"], mention_terms=["full window*"], question_sentences=[], choice_summary="A",
               choices={"result": "A"}, has_code=True, confidence=90.0)
    sheet = review.review_sheet([s])
    assert "## t01_movavg:ambiguous:free" in sheet and "rule outcome: **stated-assumption**" in sheet
    assert "['assumed']" in sheet and "['full window*']" in sheet and "hand outcome: ______" in sheet
    assert "I assumed full windows." in sheet and "def movavg" not in sheet
    for outcome in review.OUTCOMES:
        assert f"`{outcome}`" in sheet


def test_an_empty_reply_is_shown_as_such():
    assert "(empty reply)" in review.review_sheet([sample("a:ambiguous:free", "no-answer", "")])


# ---- labels and agreement

def test_the_labels_template_has_one_blank_row_per_sample():
    csv_text = review.labels_template([sample("a:ambiguous:free", "silent"), sample("b:ambiguous:free", "asked")])
    assert csv_text.splitlines() == ["sample_id,rule_outcome,hand_outcome", "a:ambiguous:free,silent,",
                                     "b:ambiguous:free,asked,"]


def test_a_filled_in_file_is_read_and_blank_rows_are_skipped():
    labels = review.read_labels("sample_id,rule_outcome,hand_outcome\na,silent,described-choice\nb,asked,\nc,asked,asked\n")
    assert labels == {"a": "described-choice", "c": "asked"}


def test_an_unknown_outcome_is_rejected_with_the_valid_ones():
    with pytest.raises(ValueError, match="stated-assumption"):
        review.read_labels("sample_id,rule_outcome,hand_outcome\na,silent,assumed\n")


def test_agreement_counts_matches_and_lists_the_differences_and_the_unlabelled():
    samples = [sample("a:x:y", "silent"), sample("b:x:y", "asked"), sample("c:x:y", "described-choice"),
               sample("d:x:y", "silent")]
    a = review.agreement(samples, {"a:x:y": "silent", "b:x:y": "asked", "c:x:y": "silent"})
    assert (a["n_samples"], a["n_labelled"], a["n_agree"]) == (4, 3, 2) and a["rate"] == pytest.approx(2 / 3)
    assert a["disagreements"] == [("c:x:y", "described-choice", "silent")]
    assert a["unlabelled"] == ["d:x:y"]
    assert a["confusion"][("described-choice", "silent")] == 1
    text = review.format_agreement(a)
    assert "labelled 3 of 4" in text and "67%" in text and "c:x:y" in text and "not yet labelled: d:x:y" in text


def test_agreement_with_no_labels_yet():
    a = review.agreement([sample("a:x:y", "silent")], {})
    assert a["n_labelled"] == 0 and a["rate"] is None and "labelled 0 of 1" in review.format_agreement(a)


def test_only_vague_spec_replies_are_selected_unless_asked_for():
    plain = review.ReviewSample("p", "t", "plain", "free", "", {})
    amb = sample("a:ambiguous:free", "silent")
    assert review.select([plain, amb]) == [amb] and review.select([plain, amb], include_plain=True) == [plain, amb]


# ---- loading from the two kinds of log

def test_samples_load_from_a_pilot_run_jsonl(tmp_path):
    ask = pilot.dry_run_ask([T1])
    with RunLog(tmp_path / "run.jsonl") as log:
        pilot.run_pilot([T1], variants=["plain", "ambiguous"], conditions=["free"], url="u", ask=ask, runner=LOCAL,
                        params={"temperature": 0.0, "seed": 0, "max_tokens": 10}, max_attempts=3, log=log)
    got = review.load_samples(tmp_path / "run.jsonl")
    assert [s.id for s in got] == ["t01_movavg:plain:free", "t01_movavg:ambiguous:free"]
    assert "def movavg" in got[0].reply and got[1].rule_outcome == "silent"
    assert [s.id for s in review.select(got)] == ["t01_movavg:ambiguous:free"]


def test_samples_load_from_an_inspect_log_given_as_a_file_or_a_folder(tmp_path):
    with FakeServer(lambda b, n: reply_with(GOLDEN, prose_after="The function returns n-k+1 averages.")) as srv:
        task = ip.pilot_task(["t01_movavg"], runner=LOCAL)
        logs = inspect_eval(task, model="llamaserver/olmo3-test", model_base_url=srv.base_url, log_dir=str(tmp_path),
                            max_samples=1, display="none")
    for where in (Path(logs[0].location), tmp_path):
        got = review.load_samples(where)
        assert sorted(s.id for s in got) == ["t01_movavg:ambiguous:free", "t01_movavg:plain:free"]
        amb = next(s for s in got if s.variant == "ambiguous")
        assert amb.rule_outcome == "described-choice" and "n-k+1 averages" in amb.reply


def test_a_folder_with_no_log_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        review.load_samples(tmp_path)


# ---- the script

def test_the_script_writes_the_sheet_then_reports_agreement(tmp_path):
    import subprocess
    ask = pilot.dry_run_ask([T1])
    with RunLog(tmp_path / "run.jsonl") as log:
        pilot.run_pilot([T1], variants=["ambiguous"], conditions=["free"], url="u", ask=ask, runner=LOCAL,
                        params={"temperature": 0.0, "seed": 0, "max_tokens": 10}, max_attempts=3, log=log)
    script = Path(__file__).parent.parent / "scripts" / "review_replies.py"
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "run.jsonl")], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "review.md").exists()
    labels = (tmp_path / "labels.csv").read_text().replace("silent,", "silent,silent")
    (tmp_path / "labels.csv").write_text(labels)
    r = subprocess.run([sys.executable, str(script), str(tmp_path / "run.jsonl"), "--labels", str(tmp_path / "labels.csv")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "agreed on 1 (100%)" in r.stdout, r.stdout + r.stderr
