from pathlib import Path

import pytest

from bench import prompts as p

TASKS = Path(__file__).parent.parent / "tasks"


def test_conversation_is_exactly_one_system_and_one_user_message():
    msgs = p.build_messages("free", "Translate this.")
    assert [m["role"] for m in msgs] == ["system", "user"]
    assert msgs[0]["content"] == p.SYSTEM_FREE
    assert msgs[1]["content"] == "Translate this.\n"


@pytest.mark.parametrize("condition", sorted(p.CONDITIONS))
def test_system_message_is_always_present_and_non_empty(condition):
    msgs = p.build_messages(condition, "spec")
    assert msgs[0]["role"] == "system" and len(msgs[0]["content"]) > 50


def test_the_spec_is_the_only_task_content_and_nothing_leaks_between_samples():
    a = p.build_messages("free", "first spec")
    b = p.build_messages("free", "second spec")
    assert "first spec" not in str(b) and "second spec" not in str(a)
    assert a[0] == b[0]                     # same system message each time
    assert a is not b and a[1] is not b[1]  # no shared state


@pytest.mark.parametrize("spec", ["", "   ", "\n\n"])
def test_empty_spec_is_rejected(spec):
    with pytest.raises(ValueError):
        p.build_messages("free", spec)


def test_unknown_condition_is_rejected_with_the_valid_names():
    with pytest.raises(ValueError, match="free"):
        p.build_messages("nope", "spec")


def test_free_condition_does_not_mention_asking_questions():
    low = p.SYSTEM_FREE.lower()
    for word in ("question", "clarif", "ask", "ambig", "assum"):
        assert word not in low, word


def test_format_condition_offers_question_and_code_markers():
    assert "QUESTION:" in p.SYSTEM_FORMAT and "CODE:" in p.SYSTEM_FORMAT


@pytest.mark.parametrize("condition", sorted(p.CONDITIONS))
def test_both_conditions_ask_for_the_confidence_line_in_the_form_the_parser_reads(condition):
    from bench.parsing import parse_confidence
    text = p.system_prompt(condition)
    assert "CONFIDENCE: <number from 0 to 100>" in text
    assert parse_confidence("CONFIDENCE: 80") == (80.0, "ok")


def test_system_prompts_do_not_name_any_task_or_trap():
    text = (p.SYSTEM_FREE + p.SYSTEM_FORMAT).lower()
    for word in ("movavg", "simpson", "rk4", "colscale", "robust", "uint8", "round", "indexing", "orientation"):
        assert word not in text, word


def test_sha256_is_stable_and_differs_between_conditions():
    assert p.sha256(p.SYSTEM_FREE) == p.sha256(p.SYSTEM_FREE)
    assert p.sha256(p.SYSTEM_FREE) != p.sha256(p.SYSTEM_FORMAT)
    assert len(p.sha256("x")) == 64


def test_every_real_spec_builds_a_valid_conversation():
    for spec in sorted(TASKS.glob("*/spec*.md")):
        msgs = p.build_messages("free", spec.read_text(encoding="utf-8"))
        assert len(msgs) == 2 and msgs[1]["content"].strip()
