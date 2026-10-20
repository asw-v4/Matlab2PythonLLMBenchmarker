"""Rules that must hold for every task, whatever it tests."""
import re
from pathlib import Path

import pytest

from bench.taskio import load_task

TASKS_DIR = Path(__file__).parent.parent / "tasks"
TASK_DIRS = sorted(p for p in TASKS_DIR.iterdir() if (p / "task.toml").exists())
IDS = [p.name for p in TASK_DIRS]

HEADER = re.compile(r"^function\s+(?:\[?[^=\n]*\]?\s*=\s*)?(\w+)\((.*?)\)", re.MULTILINE)


def matlab_signature(spec_text: str) -> str:
    """'movavg(x, k)' from the MATLAB header in the plain spec."""
    code = spec_text.split("```matlab")[1].split("```")[0]
    m = HEADER.search(code)
    return f"{m.group(1)}({m.group(2)})"


def test_there_are_five_pilot_tasks():
    assert len(TASK_DIRS) == 5


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_every_task_has_the_same_files(task_dir):
    for name in ("task.toml", "spec.md", "spec_ambiguous.md", "solution.py", "expected.mat",
                 "generate_expected.m", "reference"):
        assert (task_dir / name).exists(), name
    assert any((task_dir / "reference").glob("*.m"))


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_the_ambiguous_spec_is_about_the_same_function_as_the_plain_spec(task_dir):
    plain = (task_dir / "spec.md").read_text(encoding="utf-8")
    ambiguous = (task_dir / "spec_ambiguous.md").read_text(encoding="utf-8")
    task = load_task(task_dir)
    signature = matlab_signature(plain)
    assert signature.split("(")[0] == task.meta["task"]["function"]
    prose = plain.split("```")[2]                                                   # the text after the code
    assert f"`{signature}`" in prose, f"{task_dir.name}: plain spec does not ask for {signature}"
    assert f"`{signature}`" in ambiguous, f"{task_dir.name}: ambiguous spec does not ask for {signature}"
    assert "```" not in ambiguous                                                   # in words, not code


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_both_specs_ask_for_the_same_return_value(task_dir):
    names = load_task(task_dir).output_names
    if len(names) == 1:
        return
    tup = "(" + ", ".join(names) + ")"
    for spec in ("spec.md", "spec_ambiguous.md"):
        assert tup in (task_dir / spec).read_text(encoding="utf-8"), f"{task_dir.name}/{spec} lacks {tup}"


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_a_declared_ambiguity_is_about_the_tasks_own_function(task_dir):
    meta = load_task(task_dir).meta
    if "ambiguity" in meta:
        assert meta["ambiguity"]["function"] == meta["task"]["function"]
        assert set(meta["ambiguity"]["outputs"]) <= set(meta.get("outputs", {}).get("names", ["result"]))


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_every_declared_reading_exists_in_every_case_and_nothing_undeclared_does(task_dir):
    task = load_task(task_dir)
    declared = set(task.meta["readings"])
    assert declared, "every task needs at least one alternative reading"
    for c in task.cases:
        assert set(c.readings) == declared, f"case {c.index}"
        for name, outs in c.readings.items():
            assert outs and set(outs) <= set(task.output_names)


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_known_wrong_variants_declared_equal_those_in_the_key(task_dir):
    task = load_task(task_dir)
    for c in task.cases:
        assert set(c.wrong) == set(task.meta["known_wrong"])


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_the_octave_key_was_made_with_the_pinned_version(task_dir):
    version = (TASKS_DIR.parent / "octave" / "VERSION").read_text().strip()
    assert load_task(task_dir).octave_version == version


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_argument_order_matches_the_matlab_header_and_the_inputs_in_the_key(task_dir):
    task = load_task(task_dir)
    plain = (task_dir / "spec.md").read_text(encoding="utf-8")
    header_args = [a.strip() for a in HEADER.search(plain.split("```matlab")[1]).group(2).split(",")]
    assert task.arg_order == header_args
    for c in task.cases:
        assert set(c.inputs) == set(task.arg_order), f"case {c.index}"
    assert task.function == matlab_signature(plain).split("(")[0]


# ---- the vocabulary each task uses to recognise a reply that talks about the vague point

@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_every_task_declares_an_ambiguity_with_a_vocabulary(task_dir):
    task = load_task(task_dir)
    amb = task.meta["ambiguity"]
    assert amb["function"] == task.function and set(amb["outputs"]) <= set(task.output_names)
    vocab = task.ambiguity_vocabulary
    assert vocab and len(vocab) == len(set(vocab)) and len(vocab) >= 8
    assert all(isinstance(t, str) and t.strip() and t == t.lower() and t == t.strip() for t in vocab)


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_no_vocabulary_word_is_already_in_the_vague_spec(task_dir):
    # a reply that echoes the spec's own words would look like it had said something about the vague point
    from bench.parsing import find_terms, normalise
    spec = normalise((task_dir / "spec_ambiguous.md").read_text(encoding="utf-8"))
    assert find_terms(spec, load_task(task_dir).ambiguity_vocabulary) == set()


@pytest.mark.parametrize("task_dir", TASK_DIRS, ids=IDS)
def test_the_vocabulary_recognises_every_reading_the_task_defines(task_dir):
    # each alternative reading should be nameable in the vocabulary, otherwise a reply that explains its choice
    # would go unnoticed. The check is on the task's own description of the readings.
    task = load_task(task_dir)
    from bench.parsing import find_terms, normalise
    for name, description in task.meta["readings"].items():
        hit = find_terms(normalise(description), task.ambiguity_vocabulary)
        assert hit, f"{task_dir.name}: no vocabulary word appears in the description of reading {name!r}: {description}"
