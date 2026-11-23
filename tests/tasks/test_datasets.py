# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from synclaw.runners.model_runner import load_task
from synclaw.tasks.datasets import (
    PROMPT_FINGERPRINT,
    build_llm_task,
    generate_llm_taskset,
    parse_budget_set,
)


def test_wildcard_targets_are_rejected_because_they_can_never_match_exactly():
    with pytest.raises(ValueError, match="wildcard atom"):
        build_llm_task("*c1ccccc1", task_id="t", dataset="d", source_file="f")


def test_ordinary_targets_still_build():
    assert (
        build_llm_task("CCO", task_id="t", dataset="d", source_file="f")["canonical_target_smiles"]
        == "CCO"
    )


def test_prompt_fingerprint_forces_regeneration_when_prompt_text_moves_on(tmp_path):
    """Stale wording is regenerated rather than rewritten in place."""
    source = tmp_path / "targets.smi"
    source.write_text("SMILES\nCCO\n", encoding="utf-8")
    taskset = tmp_path / "tasks.jsonl"
    generate_llm_taskset(source, taskset, dataset="fixture")
    assert load_task(str(taskset), 0)["prompt_fingerprint"] == PROMPT_FINGERPRINT

    for stamp, reported in (
        ({}, "with no prompt fingerprint"),
        ({"prompt_fingerprint": "0123456789ab"}, "0123456789ab"),
    ):
        (tmp_path / "stale.jsonl").write_text(json.dumps({"task_id": "old", **stamp}) + "\n")
        with pytest.raises(RuntimeError, match=reported):
            load_task(str(tmp_path / "stale.jsonl"), 0)


def test_taskset_draws_one_tool_exclusion_set_per_task(tmp_path):
    source = tmp_path / "targets.smi"
    source.write_text(
        "SMILES\n" + "\n".join(["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCl"]) + "\n"
    )
    sets = [[], ["mces", "similarity"]]

    first = generate_llm_taskset(
        source, tmp_path / "a.jsonl", dataset="d", tool_exclusion_sets=sets
    )
    generate_llm_taskset(source, tmp_path / "b.jsonl", dataset="d", tool_exclusion_sets=sets)
    rows = [json.loads(line) for line in (tmp_path / "a.jsonl").read_text().splitlines()]

    assert (tmp_path / "a.jsonl").read_text() == (tmp_path / "b.jsonl").read_text()
    assert {tuple(row["excluded_tools"]) for row in rows} == {(), ("mces", "similarity")}
    assert sum(first["tool_exclusion_counts"].values()) == len(rows)
    for row in rows:
        assert not set(row["excluded_tools"]) & set(row["available_tools"])
        assert "submit_pathway" in row["available_tools"]


def test_task_cannot_exclude_submit_pathway():
    with pytest.raises(RuntimeError, match="Cannot exclude required tool"):
        build_llm_task(
            "CCO", task_id="t", dataset="d", source_file="f", excluded_tools=["submit_pathway"]
        )


def test_taskset_draws_budget_sets_independently_of_tool_exclusions(tmp_path):
    source = tmp_path / "targets.smi"
    source.write_text(
        "SMILES\n" + "\n".join(["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCl"]) + "\n"
    )
    tool_sets = [[], ["mces", "similarity"]]
    budget_sets = [{}, parse_budget_set("max_turns=40,max_tool_calls=75,budget_feedback=false")]

    generate_llm_taskset(
        source, tmp_path / "tools.jsonl", dataset="d", tool_exclusion_sets=tool_sets
    )
    stats = generate_llm_taskset(
        source,
        tmp_path / "both.jsonl",
        dataset="d",
        tool_exclusion_sets=tool_sets,
        budget_sets=budget_sets,
    )
    tools_only = [json.loads(line) for line in (tmp_path / "tools.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (tmp_path / "both.jsonl").read_text().splitlines()]

    assert [row["excluded_tools"] for row in rows] == [row["excluded_tools"] for row in tools_only]
    budgeted = [row for row in rows if "max_turns" in row]
    assert budgeted and len(budgeted) < len(rows)
    for row in budgeted:
        assert (row["max_turns"], row["max_tool_calls"], row["budget_feedback"]) == (40, 75, False)
    # Rows without a set carry no budget fields, so they inherit the runner's.
    assert all(
        "max_tokens" not in row and "budget_feedback" not in row
        for row in rows
        if row not in budgeted
    )
    assert stats["budget_set_counts"] == {
        "none": len(rows) - len(budgeted),
        "budget_feedback=false,max_tool_calls=75,max_turns=40": len(budgeted),
    }


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "max_turns=inf,max_model_len=inf,max_tool_calls=40",
            {"max_turns": "inf", "max_model_len": "inf", "max_tool_calls": 40},
        ),
        ("budget_feedback=false,max_turns=40", {"max_turns": 40, "budget_feedback": False}),
    ],
)
def test_budget_sets_accept_inf_and_flags(text, expected):
    assert parse_budget_set(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("max_turns=0", "positive integer"),
        ("max_turns=4.5", "must be an integer"),
        ("budget_feedback=no", "true or false"),
        ("max_turns", "name=value"),
        ("max_context=1000", "Unknown task budget"),
        ("max_turn=inf2", "Unknown task budget"),  # the name is checked before the value
        ("max_turns=40,max_turns=5", "more than once"),
    ],
)
def test_bad_budget_sets_are_refused(text, message):
    with pytest.raises(ValueError, match=message):
        parse_budget_set(text)


def test_task_rows_carry_only_fields_something_reads():
    task = build_llm_task("CCO", task_id="t", dataset="d", source_file="f")

    assert not {"expected_final_tool", "expected_submission", "scoring"} & set(task)
    assert task["question"] == "Target molecule: CCO"
