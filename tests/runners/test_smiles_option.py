# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`run-model --smiles` plans one target without a task set."""

import argparse
import json

import pytest

from synclaw.rollout import prompts
from synclaw.runners import model_runner
from synclaw.tasks.datasets import PROMPT_FINGERPRINT, build_adhoc_task


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    return parser.parse_args(["--model", "fixture", *argv])


def test_adhoc_task_is_valid_and_keyed_by_its_smiles() -> None:
    task = build_adhoc_task("  OCC \n")

    assert task["target_smiles"] == "OCC"
    assert task["canonical_target_smiles"] == "CCO"
    assert task["question"] == "Target molecule: OCC"
    assert task["task_id"] == build_adhoc_task("OCC")["task_id"] != build_adhoc_task("CCN")["task_id"]
    assert task["available_tools"] and task["excluded_tools"] == []


def test_smiles_replaces_the_task_set() -> None:
    task = model_runner.resolve_run_task(_args("--smiles", "CCO"))

    assert task["target_smiles"] == "CCO"


@pytest.mark.parametrize("extra", [["--taskset", "tasks.jsonl"], ["--task-index", "0"]])
def test_smiles_conflicts_with_task_set_flags(extra) -> None:
    with pytest.raises(RuntimeError, match="cannot be combined"):
        model_runner.resolve_run_task(_args("--smiles", "CCO", *extra))


def test_invalid_smiles_is_a_clean_error() -> None:
    with pytest.raises(RuntimeError, match="Invalid target SMILES"):
        model_runner.resolve_run_task(_args("--smiles", "not a molecule"))


def test_task_set_selection_is_unchanged(tmp_path) -> None:
    taskset = tmp_path / "tasks.jsonl"
    rows = [
        {"task_id": f"t{i}", "target_smiles": "CCO", "system_prompt": "s", "question": "q",
         "prompt_fingerprint": PROMPT_FINGERPRINT}
        for i in range(2)
    ]
    taskset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert model_runner.resolve_run_task(_args("--taskset", str(taskset)))["task_id"] == "t0"
    assert model_runner.resolve_run_task(_args("--taskset", str(taskset), "--task-index", "1"))["task_id"] == "t1"


def test_run_sends_the_smiles_target_to_the_model(tmp_path, monkeypatch) -> None:
    args = _args(
        "--api-mode", "chat", "--base-url", "https://example.test/v1",
        "--smiles", "CCO", "--db", str(tmp_path / "unused.sqlite"),
        "--trace", str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *a, **k: "fixture-key")
    monkeypatch.setattr(prompts, "validate_model_run_database", lambda *a, **k: None)
    requests = []

    def fail_request(**kwargs):
        requests.append(kwargs)
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(model_runner, "chat_completion", fail_request)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        model_runner.run_model_from_args(args)

    assert "Target molecule: CCO" in json.dumps(requests[0]["messages"])
