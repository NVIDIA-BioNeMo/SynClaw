# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The trace file is appended to as a run goes, including across a resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from synclaw.core import db
from synclaw.runners import model_runner
from synclaw.tasks.datasets import PROMPT_FINGERPRINT


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def run(tmp_path, monkeypatch):
    """Return ``(args, write_trace_calls)`` for a chat run whose model request always fails."""

    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text(
        json.dumps(
            {
                "task_id": "fixture",
                "target_smiles": "CCO",
                "system_prompt": "Plan syntheses.",
                "question": "Make CCO.",
                "prompt_fingerprint": PROMPT_FINGERPRINT,
            }
        )
        + "\n"
    )
    db.init_db(tmp_path / "empty.sqlite")
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    args = parser.parse_args(
        [
            "--model",
            "fixture",
            "--api-mode",
            "chat",
            "--base-url",
            "https://example.test/v1",
            "--taskset",
            str(taskset),
            "--db",
            str(tmp_path / "empty.sqlite"),
            "--trace",
            str(tmp_path / "trace.jsonl"),
            "--checkpoint",
            str(tmp_path / "checkpoint.json"),
            "--max-model-len",
            "50000",
        ]
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *a, **k: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *a, **k: None)

    def fail_request(**_kwargs):
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(model_runner, "chat_completion", fail_request)
    writes: list[int] = []
    real_write_trace = model_runner.write_trace

    def counting_write_trace(path, events):
        writes.append(len(events))
        real_write_trace(path, events)

    monkeypatch.setattr(model_runner, "write_trace", counting_write_trace)
    return args, writes


def _run_expecting_failure(args) -> None:
    with pytest.raises(RuntimeError, match="fixture interruption"):
        model_runner.run_model_from_args(args)


def test_resume_appends_to_the_trace_instead_of_rewriting_it(run, tmp_path) -> None:
    args, writes = run
    _run_expecting_failure(args)
    first = _events(tmp_path / "trace.jsonl")
    _run_expecting_failure(args)  # resumes from the checkpoint

    resumed = _events(tmp_path / "trace.jsonl")
    assert len(writes) == 1  # only the first run created the file
    assert resumed[: len(first)] == first
    # The resumed turn records its request budget, then fails again.
    assert [e["event_type"] for e in resumed[len(first) :]] == ["request_budget", "run_error"]
    assert sum(e["event_type"] == "message" for e in resumed) == 2  # the prompt, once


def test_resume_drops_events_the_checkpoint_does_not_cover(run, tmp_path) -> None:
    args, writes = run
    _run_expecting_failure(args)
    covered = _events(tmp_path / "trace.jsonl")
    with (tmp_path / "trace.jsonl").open("a") as handle:
        handle.write(json.dumps({"event_type": "stray"}) + "\n")  # as if a crash left it ahead

    _run_expecting_failure(args)

    resumed = _events(tmp_path / "trace.jsonl")
    assert all(e["event_type"] != "stray" for e in resumed)
    assert resumed[: len(covered)] == covered
    assert len(writes) == 2  # the trace had to be rewritten once


def test_resume_into_a_different_trace_path_writes_the_whole_trace(run, tmp_path) -> None:
    args, writes = run
    _run_expecting_failure(args)
    first = _events(tmp_path / "trace.jsonl")
    args.trace = str(tmp_path / "moved.jsonl")

    _run_expecting_failure(args)

    moved = _events(tmp_path / "moved.jsonl")
    assert moved[: len(first)] == first
    assert len(writes) == 2


def test_resume_refuses_a_trace_with_fewer_events_than_the_checkpoint_covers(run, tmp_path) -> None:
    args, _ = run
    _run_expecting_failure(args)
    trace = tmp_path / "trace.jsonl"
    lines = trace.read_text().splitlines()
    trace.write_text("\n".join(lines[:-1]) + "\n")  # the last covered event is gone

    with pytest.raises(RuntimeError, match="was damaged or replaced"):
        model_runner.run_model_from_args(args)


def test_the_checkpoint_records_how_many_trace_events_it_covers(run, tmp_path) -> None:
    args, _ = run
    _run_expecting_failure(args)

    checkpoint = json.loads((tmp_path / "checkpoint.json").read_text())

    assert checkpoint["trace_event_count"] == len(_events(tmp_path / "trace.jsonl"))


def test_flush_appends_only_the_new_events(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    events = [{"n": 1}, {"n": 2}]
    trace = model_runner._TraceFile(path, events)
    trace.flush()
    events.append({"n": 3})
    trace.flush()
    trace.flush()  # nothing new

    assert _events(path) == [{"n": 1}, {"n": 2}, {"n": 3}]


@pytest.mark.parametrize(
    ("text", "events", "intact"),
    [
        ('{"n": 1}\n{"n": 2}\n', [{"n": 1}, {"n": 2}], True),
        ("", [], True),
        ('{"n": 1}\n{"n": 2', [{"n": 1}], False),  # cut short mid-event
        ('{"n": 1}\n{"n": 2}', [{"n": 1}, {"n": 2}], False),  # complete event, no newline
    ],
)
def test_read_trace_flags_a_last_line_cut_short_by_a_crash(tmp_path, text, events, intact) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(text)

    assert model_runner._read_trace(path) == (events, intact)


def test_read_trace_rejects_corruption_before_the_last_line(tmp_path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text('{"n": 1}\nnot json\n{"n": 3}\n')

    with pytest.raises(RuntimeError, match="corrupt at line 2"):
        model_runner._read_trace(path)
