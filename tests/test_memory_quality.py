from __future__ import annotations

from convsearch.memory.extract import extract_from_message
from convsearch.memory.quality import is_usable_statement

# --- Real offending strings pulled from the workspace (see task report) ---


def test_rejects_trailing_colon_fragment() -> None:
    ok, reason = is_usable_statement("For your strategy, you need to know:", "task")
    assert ok is False
    assert reason is not None and "colon" in reason


def test_rejects_trailing_colon_fragment_decision() -> None:
    ok, reason = is_usable_statement("Decision:", "decision")
    assert ok is False
    assert reason is not None


def test_rejects_table_row_debris() -> None:
    text = (
        "3\tMassive / Polygon-style\tGood\tTrying APIs, historical market data"
        "\tNeed to verify Greeks availability on free tier"
    )
    ok, reason = is_usable_statement(text, "task")
    assert ok is False
    assert reason is not None and "table-row" in reason


def test_rejects_negation_of_need() -> None:
    text = "That means you do not need to compute delta yourself."
    ok, reason = is_usable_statement(text, "task")
    assert ok is False
    assert reason is not None and "negates" in reason


def test_rejects_pure_question_task() -> None:
    text = (
        "�When an agent fails an evaluation, what does the team need to determine "
        "before deciding how to improve it?�"
    )
    ok, reason = is_usable_statement(text, "task")
    assert ok is False
    assert reason is not None and "question" in reason


def test_rejects_short_fragment() -> None:
    ok, reason = is_usable_statement("Account constraint", "constraint")
    assert ok is False
    assert reason is not None and "word count" in reason


# --- Positive cases that must survive ---


def test_keeps_clear_imperative_task() -> None:
    ok, reason = is_usable_statement("We need to add retry logic to the importer.", "task")
    assert ok is True
    assert reason is None


def test_keeps_decision_with_rationale() -> None:
    ok, reason = is_usable_statement(
        "We decided to use PostgreSQL over MySQL because of JSON support.", "decision"
    )
    assert ok is True
    assert reason is None


def test_keeps_constraint_phrased_as_negation() -> None:
    # This negation asserts an action IS still required (with a precondition) -- it must not
    # be caught by the task-negation rule, which only rejects "do not NEED to" phrasing.
    text = "Do not deploy without running migrations first."
    ok, reason = is_usable_statement(text, "constraint")
    assert ok is True
    assert reason is None


def test_keeps_normal_declarative_statement() -> None:
    ok, reason = is_usable_statement(
        "The stack uses FastAPI for the backend and SQLite for storage.", "project_state"
    )
    assert ok is True
    assert reason is None


def test_keeps_borderline_conditional_task() -> None:
    """A conditional/weak task is not one of the targeted rejection patterns -- kept."""
    ok, reason = is_usable_statement(
        "If it only gives quotes/bars and not Greeks, then you still need to compute delta"
        " yourself.",
        "task",
    )
    assert ok is True
    assert reason is None


# --- Extract-level integration: a known-junk candidate yields no memory ---


def test_extract_drops_trailing_colon_task_candidate() -> None:
    text = "For your strategy, you need to know: latency and cost."
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    assert not any(m.statement.rstrip().endswith(":") for m in memories)


def test_five_known_junk_statements_still_rejected() -> None:
    """Regression: the five real-workspace junk statements that motivated quality.py must

    still be rejected after the decision-recall pattern additions in extract.py -- new
    trigger phrases must not create a path around the filter for these.
    """
    cases = [
        ("For your strategy, you need to know:", "task"),
        ("Decision:", "decision"),
        (
            "3\tMassive / Polygon-style\tGood\tTrying APIs, historical market data"
            "\tNeed to verify Greeks availability on free tier",
            "task",
        ),
        ("That means you do not need to compute delta yourself.", "task"),
        (
            "“When an agent fails an evaluation, what does the team need to determine "
            "before deciding how to improve it?”",
            "task",
        ),
    ]
    for text, kind in cases:
        ok, reason = is_usable_statement(text, kind)
        assert ok is False, f"expected rejection for {text!r}, got accepted"
        assert reason is not None


def test_extract_drops_table_row_task_candidate() -> None:
    text = (
        "Some intro text.\n"
        "3\tMassive / Polygon-style\tGood\tTrying APIs, historical market data"
        "\tNeed to verify Greeks availability on free tier"
    )
    memories = extract_from_message(text, conversation_id=1, message_id=1, created_at=None)
    assert not any("\t" in m.statement for m in memories)
