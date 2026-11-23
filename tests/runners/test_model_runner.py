# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for ``synclaw.runners.model_runner``: task selection, provider defaults and flags, the
hosted transports and their request and reply formats, the shared HTTP helper, and checkpoint
and trace resume."""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from conftest import prepare_minimal_model_db

from synclaw.core import db
from synclaw.rollout import policy
from synclaw.runners import model_runner
from synclaw.runners.model_runner import (
    _HostedTransport,
    add_arguments,
    anthropic_completion,
    anthropic_response_to_chat_message,
    apply_provider_defaults,
    build_anthropic_initial_state,
    checkpoint_matches_run,
    resolve_task_db_path,
    responses_completion,
    responses_to_chat_message,
    run_model_from_args,
)
from synclaw.tasks.datasets import PROMPT_FINGERPRINT, build_adhoc_task, build_llm_task
from synclaw.token_usage import usage_from_payload


def _hosted_transport(api_mode: str, **state):
    args = SimpleNamespace(
        api_mode=api_mode,
        model="mock-model",
        responses_state_mode="full",
        responses_url="https://example.test/v1/responses",
        max_tokens=16,
        temperature=None,
        top_p=None,
        sampling_seed=None,
        reasoning_effort=None,
        retries=0,
        request_timeout=1.0,
    )
    fields = {
        "responses_previous_id": None,
        "responses_pending_input": [],
        "responses_input_history": [],
        "anthropic_system": [],
        "anthropic_messages": [],
        "anthropic_tool_results": [],
        "chat_tool_schemas": [],
        "responses_tool_schemas": [],
        "messages_tool_schemas": [],
        **state,
    }
    config = policy.RolloutPolicyConfig(max_tokens=args.max_tokens)
    return _HostedTransport(args=args, config=config, api_key="test-key", **fields)


def _provider_args(**overrides):
    import argparse

    fields = {
        "provider": "nvbuild",
        "model": "example-model",
        "api_mode": None,
        "base_url": None,
        "responses_url": None,
        "messages_url": None,
        "responses_state_mode": None,
        "reasoning_effort": None,
        **overrides,
    }
    return argparse.Namespace(**fields)


# --- Task selection (`--smiles`, `--taskset`, `--task-index`) -------------------------------------


def _args(*argv: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)
    return parser.parse_args(["--model", "fixture", *argv])


def test_adhoc_task_is_valid_and_keyed_by_its_smiles() -> None:
    task = build_adhoc_task("  OCC \n")

    assert task["target_smiles"] == "OCC"
    assert task["canonical_target_smiles"] == "CCO"
    assert task["question"] == "Target molecule: OCC"
    assert task["task_id"].startswith("smiles-") and len(task["task_id"]) == len("smiles-") + 12
    assert task["task_id"] == build_adhoc_task("CCO")["task_id"]  # same molecule, other spelling
    assert task["task_id"] != build_adhoc_task("CCN")["task_id"]
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
        {
            "task_id": f"t{i}",
            "target_smiles": "CCO",
            "system_prompt": "s",
            "question": "q",
            "prompt_fingerprint": PROMPT_FINGERPRINT,
        }
        for i in range(2)
    ]
    taskset.write_text("".join(json.dumps(row) + "\n" for row in rows))

    assert model_runner.resolve_run_task(_args("--taskset", str(taskset)))["task_id"] == "t0"
    assert (
        model_runner.resolve_run_task(_args("--taskset", str(taskset), "--task-index", "1"))[
            "task_id"
        ]
        == "t1"
    )


def test_empty_taskset_is_an_error_not_the_default(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    with pytest.raises(RuntimeError, match="Cannot load the task"):
        model_runner.resolve_run_task(_args("--taskset", ""))


def test_a_task_index_past_the_end_is_a_clean_error(tmp_path) -> None:
    taskset = tmp_path / "tasks.jsonl"
    taskset.write_text("")

    with pytest.raises(RuntimeError, match="No task at index 3"):
        model_runner.resolve_run_task(_args("--taskset", str(taskset), "--task-index", "3"))


@pytest.mark.parametrize(
    ("flags", "message"),
    [
        (["--smiles", "not a molecule"], "Invalid target SMILES"),
        (["--smiles", "CCO", "--taskset", "t.jsonl"], "cannot be combined"),
    ],
)
def test_bad_target_is_reported_before_the_missing_api_key(monkeypatch, flags, message) -> None:
    monkeypatch.delenv("TEST_SYNCLAW_API_KEY", raising=False)
    args = _args(
        "--provider",
        "custom",
        "--api-mode",
        "chat",
        "--base-url",
        "https://example.test/v1",
        "--api-key-env",
        "TEST_SYNCLAW_API_KEY",
        *flags,
    )

    with pytest.raises(RuntimeError, match=message):
        model_runner.run_model_from_args(args)


def test_missing_api_key_is_reported_for_a_valid_target(monkeypatch) -> None:
    monkeypatch.delenv("TEST_SYNCLAW_API_KEY", raising=False)
    args = _args(
        "--provider",
        "custom",
        "--api-mode",
        "chat",
        "--base-url",
        "https://example.test/v1",
        "--api-key-env",
        "TEST_SYNCLAW_API_KEY",
        "--smiles",
        "CCO",
    )

    with pytest.raises(RuntimeError, match="TEST_SYNCLAW_API_KEY is not set"):
        model_runner.run_model_from_args(args)


def test_run_sends_the_smiles_target_to_the_model(tmp_path, monkeypatch) -> None:
    db.init_db(tmp_path / "empty.sqlite")
    args = _args(
        "--api-mode",
        "chat",
        "--base-url",
        "https://example.test/v1",
        "--smiles",
        "CCO",
        "--db",
        str(tmp_path / "empty.sqlite"),
        "--trace",
        str(tmp_path / "trace.jsonl"),
    )
    monkeypatch.setattr(model_runner, "resolve_api_key", lambda *a, **k: "fixture-key")
    monkeypatch.setattr(db, "validate_model_run_database", lambda *a, **k: None)
    requests = []

    def fail_request(**kwargs):
        requests.append(kwargs)
        raise RuntimeError("fixture interruption")

    monkeypatch.setattr(model_runner, "chat_completion", fail_request)
    with pytest.raises(RuntimeError, match="fixture interruption"):
        model_runner.run_model_from_args(args)

    assert "Target molecule: CCO" in json.dumps(requests[0]["messages"])


# --- Shared HTTP helper and flag ranges ----------------------------------------------------------


class _Reply:
    def __init__(self, body: dict[str, Any]) -> None:
        self._body = json.dumps(body).encode()

    def __enter__(self) -> _Reply:
        return self

    def __exit__(self, *_exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _http_error(code: int, body: str = "") -> urllib.error.HTTPError:
    headers: dict = {}
    return urllib.error.HTTPError("http://x", code, "err", headers, io.BytesIO(body.encode()))


def _post(retries: int = 2) -> dict[str, Any]:
    return model_runner._post_json(
        "http://x/v1",
        {"a": 1},
        {"Authorization": "Bearer k"},
        api_name="Chat Completions",
        retries=retries,
        request_timeout=5,
    )


def test_a_transient_http_error_is_retried_then_succeeds() -> None:
    outcomes = [_http_error(503), _Reply({"ok": True})]

    def fake_urlopen(request, timeout):
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    with (
        patch("urllib.request.urlopen", side_effect=fake_urlopen),
        patch("synclaw.runners.model_runner.time.sleep") as sleep,
    ):
        assert _post() == {"ok": True}
    sleep.assert_called_once()


def test_a_non_retryable_http_error_names_the_api_and_does_not_retry() -> None:
    with (
        patch("urllib.request.urlopen", side_effect=_http_error(400, "bad request")) as urlopen,
        patch("synclaw.runners.model_runner.time.sleep") as sleep,
        pytest.raises(RuntimeError, match="Chat Completions API HTTP 400: bad request"),
    ):
        _post()
    assert urlopen.call_count == 1
    sleep.assert_not_called()


def test_timeouts_stop_after_the_retries_are_used() -> None:
    with (
        patch("urllib.request.urlopen", side_effect=TimeoutError("slow")) as urlopen,
        patch("synclaw.runners.model_runner.time.sleep"),
        pytest.raises(RuntimeError, match="Chat Completions API request timed out or failed"),
    ):
        _post(retries=2)
    assert urlopen.call_count == 3


def test_the_request_carries_the_json_body_and_headers() -> None:
    seen = {}

    def fake_urlopen(request, timeout):
        seen["body"] = json.loads(request.data)
        seen["headers"] = {k.lower(): v for k, v in request.header_items()}
        return _Reply({})

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        _post()
    assert seen["body"] == {"a": 1}
    assert seen["headers"]["content-type"] == "application/json"
    assert seen["headers"]["authorization"] == "Bearer k"


@pytest.mark.parametrize(
    "flags",
    [
        ["--retries", "-1"],
        ["--request-timeout", "0"],
        ["--max-turns", "0"],
        ["--max-tool-calls", "-1"],
        ["--turn-warning-remaining", "-1"],
        ["--same-pathway-finish-after", "-1"],
    ],
)
def test_out_of_range_numeric_flags_are_rejected_when_parsing(flags: list[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "m", *flags])


def test_zero_disables_the_budget_flags_that_say_so() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    model_runner.add_arguments(parser)

    args = parser.parse_args(
        [
            "--model",
            "m",
            "--max-tool-calls",
            "0",
            "--turn-warning-remaining",
            "0",
            "--same-pathway-finish-after",
            "0",
            "--retries",
            "0",
        ]
    )

    assert (args.max_tool_calls, args.turn_warning_remaining) == (0, 0)
    assert (args.same_pathway_finish_after, args.retries) == (0, 0)


# --- Trace and checkpoint across a resume ---------------------------------------------------------


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


def test_resume_keeps_the_existing_trace_and_adds_only_new_events(run, tmp_path) -> None:
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


# --- Provider defaults, checkpoints, transports and reply parsing ---------------------------------


def test_checkpoint_rejects_a_different_submission_floor():
    import argparse

    args = argparse.Namespace(
        model="m",
        provider="p",
        api_mode="chat",
        responses_state_mode=None,
        stop_condition="exact",
    )
    task = {"task_id": "t", "target_smiles": "CCO"}
    checkpoint = {
        "model": "m",
        "provider": "p",
        "api_mode": "chat",
        "task_id": "t",
        "target_smiles": "CCO",
        "stop_condition": 1.0,
    }

    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    checkpoint["stop_condition"] = "exact"
    assert checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())


def test_checkpoint_rejects_a_different_endpoint_or_auth_header():
    import argparse

    args = argparse.Namespace(
        model="m",
        provider="custom",
        api_mode="responses",
        responses_state_mode="previous_id",
        base_url="",
        responses_url="https://a.example/v1/responses",
        messages_url="",
        auth_header="bearer",
    )
    task = {"task_id": "t", "target_smiles": "CCO"}
    checkpoint = {
        "model": "m",
        "provider": "custom",
        "api_mode": "responses",
        "responses_state_mode": "previous_id",
        "responses_previous_id": "resp_1",
        "base_url": "",
        "responses_url": "https://a.example/v1/responses",
        "messages_url": "",
        "auth_header": "bearer",
        "task_id": "t",
        "target_smiles": "CCO",
    }

    assert checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    args.responses_url = "https://b.example/v1/responses"
    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())
    args.responses_url = checkpoint["responses_url"]
    args.auth_header = "x-api-key"
    assert not checkpoint_matches_run(checkpoint, args, task, policy.RolloutPolicyConfig())


class ModelRunnerTests(unittest.TestCase):
    def test_model_runner_provider_defaults(self):
        import argparse

        args = argparse.Namespace(
            provider="nvbuild",
            api_mode=None,
            base_url=None,
            responses_url=None,
            responses_state_mode=None,
        )

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "chat")
        self.assertEqual(args.base_url, "https://integrate.api.nvidia.com/v1")
        self.assertEqual(args.responses_state_mode, "previous_id")
        self.assertEqual(args.responses_url, "")
        self.assertEqual(args.messages_url, "")

    def test_model_runner_api_mode_comes_from_the_preset_not_the_model_name(self):
        for provider, api_mode in (
            ("openrouter", "chat"),
            ("openai", "responses"),
            ("anthropic", "messages"),
        ):
            args = _provider_args(provider=provider, model="anthropic/example-model")

            apply_provider_defaults(args)

            self.assertEqual(args.api_mode, api_mode, provider)
            self.assertEqual(args.responses_state_mode, "previous_id", provider)
            self.assertIsNone(args.reasoning_effort, provider)

    def test_model_runner_flags_override_the_preset(self):
        args = _provider_args(
            provider="custom",
            api_mode="responses",
            responses_url="https://example.test/v1/responses",
            responses_state_mode="full",
        )

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "responses")
        self.assertEqual(args.responses_url, "https://example.test/v1/responses")
        self.assertEqual(args.responses_state_mode, "full")

    def test_model_runner_anthropic_provider_uses_the_messages_endpoint(self):
        args = _provider_args(provider="anthropic")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "messages")
        self.assertEqual(args.messages_url, "https://api.anthropic.com/v1/messages")

    def test_checkpoint_must_record_the_same_responses_state_mode(self):
        import argparse

        args = argparse.Namespace(
            model="example-model",
            provider="openai",
            api_mode="responses",
            responses_state_mode="previous_id",
        )
        task = {"task_id": "fixture-0001", "target_smiles": "CCO"}
        messages = [{"role": "user", "content": "Plan CCO."}]
        checkpoint = {
            "model": args.model,
            "provider": args.provider,
            "api_mode": args.api_mode,
            "task_id": task["task_id"],
            "target_smiles": task["target_smiles"],
            "next_turn_index": 0,
            "responses_pending_input": messages,
            "responses_input_history": messages,
        }
        config = policy.RolloutPolicyConfig()

        self.assertFalse(checkpoint_matches_run(checkpoint, args, task, config))
        self.assertTrue(
            checkpoint_matches_run(
                {**checkpoint, "responses_state_mode": "previous_id"}, args, task, config
            )
        )

    def test_model_runner_keeps_max_effort_for_messages_api(self):
        args = _provider_args(provider="anthropic", reasoning_effort="max")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "messages")
        self.assertEqual(args.reasoning_effort, "max")

    def test_model_runner_maps_max_effort_to_xhigh_for_other_apis(self):
        for provider in ("openai", "nvbuild"):
            args = _provider_args(provider=provider, reasoning_effort="max")

            apply_provider_defaults(args)

            self.assertEqual(args.reasoning_effort, "xhigh", provider)

    def test_responses_completion_sets_store_for_previous_response_id_mode(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self):
                return b'{"id":"resp_123","output":[],"usage":{"input_tokens":1,"output_tokens":1}}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = responses_completion(
                responses_url="https://example.test/v1/responses",
                api_key="test-key",
                model="example-model",
                input_items=[{"role": "user", "content": "Question"}],
                previous_response_id=None,
                store=True,
                max_output_tokens=128,
                temperature=None,
                top_p=1.0,
                sampling_seed=7,
                reasoning_effort="xhigh",
                retries=0,
                request_timeout=12.0,
                tool_choice="auto",
                tools=[],
            )

        self.assertEqual(response["id"], "resp_123")
        payload = captured["payload"]
        self.assertIs(payload["store"], True)
        self.assertEqual(payload["reasoning"], {"effort": "xhigh"})
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["seed"], 7)

    def test_model_runner_low_similarity_cap_defaults_to_none(self):
        import argparse

        parser = argparse.ArgumentParser()
        add_arguments(parser)
        args = parser.parse_args(
            [
                "--provider",
                "openai",
                "--api-key-env",
                "TEST_AGENTIC_SYNTHESIS_API_KEY",
                "--taskset",
                "tasks.jsonl",
                "--task-index",
                "0",
                "--db",
                "test.sqlite",
                "--model",
                "mock-model",
            ]
        )

        self.assertIsNone(args.max_low_similarity_submissions)
        self.assertEqual(args.max_turns, 80)
        self.assertEqual(args.max_tool_calls, 150)
        self.assertEqual(args.turn_warning_remaining, 3)
        self.assertEqual(args.same_pathway_finish_after, 5)

    def test_model_runner_openai_provider_defaults(self):
        args = _provider_args(provider="openai")

        apply_provider_defaults(args)

        self.assertEqual(args.api_mode, "responses")
        self.assertEqual(args.base_url, "https://api.openai.com/v1")
        self.assertEqual(args.responses_url, "https://api.openai.com/v1/responses")

    def test_model_runner_defaults_to_the_synclaw_db(self):
        self.assertEqual(resolve_task_db_path(None), Path("var/synclaw.sqlite"))

    def test_model_runner_respects_explicit_db_override(self):
        self.assertEqual(resolve_task_db_path("var/custom.sqlite"), Path("var/custom.sqlite"))

    def test_model_runner_omits_run_and_task_excluded_tools_from_schema(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            prepare_minimal_model_db(db_path)
            task = build_llm_task(
                "CCO",
                task_id="fixture-0001",
                dataset="fixture",
                source_file="fixture.smi",
                excluded_tools=["search_reaction_templates", "mces"],
            )
            taskset.write_text(json.dumps(task) + "\n", encoding="utf-8")
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "1",
                    "--max-tokens",
                    "16",
                    "--exclude-tool",
                    "simulate_reverse_reaction",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_chat_completion(**kwargs):
                seen_kwargs.append(kwargs)
                return {"choices": [{"message": {"role": "assistant", "content": "No tool call."}}]}

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=fake_chat_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertEqual(
                summary["excluded_tools"],
                ["mces", "search_reaction_templates", "simulate_reverse_reaction"],
            )
            tool_names = {schema["function"]["name"] for schema in seen_kwargs[0]["tools"]}
            prompt = (
                seen_kwargs[0]["messages"][0]["content"]
                + "\n\n"
                + seen_kwargs[0]["messages"][1]["content"]
            )
            self.assertNotIn("simulate_reverse_reaction", tool_names)
            self.assertNotIn("search_reaction_templates", tool_names)
            self.assertNotIn("mces", tool_names)
            for name in ("simulate_reverse_reaction", "search_reaction_templates", "mces"):
                self.assertNotIn(name, prompt)

    def test_anthropic_initial_state_separates_system_and_messages(self):
        system, messages = build_anthropic_initial_state(
            [
                {"role": "system", "content": "Stable harness instructions."},
                {"role": "user", "content": "Target-specific question."},
            ]
        )

        self.assertEqual(
            system,
            [
                {
                    "type": "text",
                    "text": "Stable harness instructions.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(
            messages,
            [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Target-specific question."}],
                }
            ],
        )

    def test_anthropic_response_to_chat_message_converts_tool_use(self):
        message = anthropic_response_to_chat_message(
            {
                "id": "msg_123",
                "stop_reason": "tool_use",
                "content": [
                    {"type": "text", "text": "I will search."},
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "search_building_blocks",
                        "input": {"query": "CCO"},
                    },
                ],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "output_tokens_details": {"thinking_tokens": 2},
                },
            },
        )

        self.assertEqual(message["content"], "I will search.")
        self.assertEqual(message["anthropic_stop_reason"], "tool_use")
        self.assertEqual(usage_from_payload(message)["reasoning_tokens"], 2)
        self.assertEqual(message["tool_calls"][0]["id"], "toolu_123")
        self.assertEqual(message["tool_calls"][0]["function"]["name"], "search_building_blocks")
        self.assertEqual(
            json.loads(message["tool_calls"][0]["function"]["arguments"]),
            {"query": "CCO"},
        )

    def test_anthropic_completion_uses_api_key_header_cache_control_and_adaptive_thinking(self):
        captured: dict[str, object] = {}

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return None

            def read(self):
                return b'{"id":"msg_123","content":[],"usage":{"input_tokens":1,"output_tokens":1}}'

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["headers"] = dict(request.header_items())
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            response = anthropic_completion(
                messages_url="https://api.anthropic.com/v1/messages",
                api_key="test-key",
                model="example-model",
                system="Stable harness instructions.",
                messages=[{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
                max_tokens=1024,
                temperature=None,
                top_p=None,
                reasoning_effort="high",
                retries=0,
                request_timeout=12.0,
                tool_choice={"type": "tool", "name": "submit_pathway"},
                tools=[
                    {
                        "name": "similarity",
                        "description": "Compute",
                        "input_schema": {"type": "object"},
                    }
                ],
            )

        self.assertEqual(response["id"], "msg_123")
        self.assertEqual(captured["url"], "https://api.anthropic.com/v1/messages")
        headers = {str(key).lower(): value for key, value in captured["headers"].items()}
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertNotIn("authorization", headers)
        payload = captured["payload"]
        self.assertEqual(
            payload["system"],
            [
                {
                    "type": "text",
                    "text": "Stable harness instructions.",
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        )
        self.assertEqual(payload["cache_control"], {"type": "ephemeral"})
        self.assertEqual(payload["thinking"], {"type": "adaptive"})
        self.assertEqual(payload["output_config"], {"effort": "high"})
        self.assertEqual(payload["tool_choice"], {"type": "tool", "name": "submit_pathway"})
        self.assertEqual(payload["tools"][0]["input_schema"], {"type": "object"})

    def test_model_runner_anthropic_messages_mode_uses_tool_results(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "example-model",
                    "--reasoning-effort",
                    "high",
                    "--max-turns",
                    "4",
                    "--max-tokens",
                    "1024",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_anthropic_completion(**kwargs):
                seen_kwargs.append(
                    {
                        **kwargs,
                        "messages": json.loads(json.dumps(kwargs["messages"])),
                        "tools": json.loads(json.dumps(kwargs["tools"])),
                    }
                )
                if len(seen_kwargs) == 1:
                    return {
                        "id": "msg-search",
                        "stop_reason": "tool_use",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "toolu-search",
                                "name": "search_building_blocks",
                                "input": {"query": "CO", "limit": 1},
                            }
                        ],
                        "usage": {"input_tokens": 10, "output_tokens": 3},
                    }
                return {
                    "id": "msg-submit",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-submit",
                            "name": "submit_pathway",
                            "input": {
                                "steps": [
                                    {
                                        "reaction_id": "R001",
                                        "reactants": ["BB:1"],
                                        "product_smiles": "C=O",
                                    }
                                ],
                                "pathway_text": "Oxidize methanol to formaldehyde.",
                            },
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 4},
                }

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.anthropic_completion",
                    side_effect=fake_anthropic_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertEqual(args.api_mode, "messages")
            self.assertEqual(args.reasoning_effort, "high")
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)
            self.assertEqual(seen_kwargs[0]["reasoning_effort"], "high")
            tool_result = next(
                block
                for message in seen_kwargs[1]["messages"]
                for block in message.get("content", [])
                if block.get("type") == "tool_result"
            )
            self.assertEqual(tool_result["type"], "tool_result")
            self.assertEqual(tool_result["tool_use_id"], "toolu-search")
            self.assertIn("results", json.loads(tool_result["content"]))

    def test_anthropic_transport_checkpoints_and_flushes_tool_results_before_repairs(self):
        assistant_message = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu-invalid",
                    "name": "search_building_blocks",
                    "input": {"query": ""},
                }
            ],
        }
        transport = _hosted_transport(
            "messages",
            anthropic_messages=[assistant_message],
        )
        transport.append_tool(
            SimpleNamespace(call_id="toolu-invalid"),
            '{"error":"invalid_tool_call","retryable":true}',
        )

        checkpoint = transport.checkpoint_fields()

        self.assertEqual(
            checkpoint["anthropic_tool_results"][0]["tool_use_id"],
            "toolu-invalid",
        )
        resumed_messages = [assistant_message]
        resumed = _hosted_transport(
            "messages",
            anthropic_messages=resumed_messages,
            anthropic_tool_results=checkpoint["anthropic_tool_results"],
        )
        resumed.append_user({"role": "user", "content": "Repair the invalid call."})

        self.assertEqual(resumed_messages[1]["role"], "user")
        self.assertEqual(resumed_messages[1]["content"][0]["type"], "tool_result")
        self.assertEqual(resumed_messages[2]["role"], "user")
        self.assertEqual(resumed_messages[2]["content"][0]["type"], "text")
        self.assertEqual(resumed.checkpoint_fields()["anthropic_tool_results"], [])

    def test_model_runner_anthropic_messages_mode_requests_final_submit_without_forced_tool_choice(
        self,
    ):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "anthropic",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "example-model",
                    "--max-turns",
                    "1",
                    "--max-tokens",
                    "1024",
                ]
            )
            seen_kwargs: list[dict] = []

            def fake_anthropic_completion(**kwargs):
                seen_kwargs.append(
                    {
                        **kwargs,
                        "messages": json.loads(json.dumps(kwargs["messages"])),
                        "tools": json.loads(json.dumps(kwargs["tools"])),
                    }
                )
                return {
                    "id": "msg-submit",
                    "stop_reason": "tool_use",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu-submit",
                            "name": "submit_pathway",
                            "input": {
                                "steps": [
                                    {
                                        "reaction_id": "R001",
                                        "reactants": ["BB:1"],
                                        "product_smiles": "C=O",
                                    }
                                ],
                                "pathway_text": "Oxidize methanol to formaldehyde.",
                            },
                        }
                    ],
                    "usage": {"input_tokens": 20, "output_tokens": 4},
                }

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.anthropic_completion",
                    side_effect=fake_anthropic_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertIsNone(seen_kwargs[0]["tool_choice"])
            final_turn_reminders = [
                block["text"]
                for message in seen_kwargs[0]["messages"]
                for block in message.get("content", [])
                if block.get("type") == "text" and "final allowed turn" in block.get("text", "")
            ]
            self.assertEqual(len(final_turn_reminders), 1)
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)

    def test_model_runner_checkpoint_survives_failure_and_can_resume(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            checkpoint = tmp_path / "checkpoints" / "run.json"
            db_path = tmp_path / "test.sqlite"
            prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "C=O",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan C=O.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--checkpoint",
                    str(checkpoint),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "3",
                    "--max-tokens",
                    "16",
                ]
            )
            calls = 0

            def failing_chat_completion(**_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    return {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [
                                        {
                                            "id": "call-search",
                                            "type": "function",
                                            "function": {
                                                "name": "search_building_blocks",
                                                "arguments": json.dumps(
                                                    {"query": "CO", "limit": 1}
                                                ),
                                            },
                                        }
                                    ],
                                }
                            }
                        ]
                    }
                raise RuntimeError("rate limit after first turn")

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=failing_chat_completion,
                ),
            ):
                with self.assertRaises(RuntimeError):
                    run_model_from_args(args)

            self.assertTrue(checkpoint.exists())
            events = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines()]
            self.assertIn("tool_result", [event["event_type"] for event in events])
            self.assertEqual(events[-1]["event_type"], "run_error")
            search_result_event = next(
                event
                for event in events
                if event["event_type"] == "tool_result"
                and event.get("tool_name") == "search_building_blocks"
            )
            self.assertIn("submission_guidance", search_result_event["payload"]["result"])
            self.assertIn("model_result", search_result_event["payload"])
            self.assertIn("model_result_compaction", search_result_event["payload"])
            self.assertNotIn(
                "submission_note", search_result_event["payload"]["model_result"]["results"][0]
            )
            checkpoint_payload = json.loads(checkpoint.read_text(encoding="utf-8"))
            tool_messages = [
                message
                for message in checkpoint_payload["messages"]
                if message.get("role") == "tool"
            ]
            self.assertEqual(len(tool_messages), 1)
            model_visible_result = json.loads(tool_messages[0]["content"])
            self.assertIn("results", model_visible_result)
            self.assertNotIn("submission_note", model_visible_result["results"][0])
            self.assertNotIn("trace_events", checkpoint_payload)
            self.assertLess(
                search_result_event["payload"]["model_result_compaction"]["model_json_bytes"],
                search_result_event["payload"]["model_result_compaction"]["full_json_bytes"],
            )

            def successful_chat_completion(**_kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "id": "call-submit",
                                        "type": "function",
                                        "function": {
                                            "name": "submit_pathway",
                                            "arguments": json.dumps(
                                                {
                                                    "steps": [
                                                        {
                                                            "reaction_id": "R001",
                                                            "reactants": ["BB:1"],
                                                            "product_smiles": "C=O",
                                                        }
                                                    ],
                                                    "pathway_text": "Oxidize methanol to formaldehyde.",
                                                }
                                            ),
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                }

            moved_trace = trace.with_name("moved-trace.jsonl")
            trace.rename(moved_trace)
            with patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}):
                with self.assertRaisesRegex(RuntimeError, "which does not exist"):
                    run_model_from_args(args)
            moved_trace.rename(trace)

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=successful_chat_completion,
                ),
            ):
                summary = run_model_from_args(args)

            self.assertFalse(checkpoint.exists())
            self.assertTrue(summary["accepted"])
            self.assertEqual(summary["similarity_to_target"], 1.0)

    def test_responses_parser_keeps_text_and_response_metadata(self):
        response = {
            "id": "resp-test",
            "model": "example-org/example-model-revision-1",
            "system_fingerprint": "deployment-abc",
            "status": "completed",
            "output": [
                {
                    "type": "reasoning",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": 'No results. Search building blocks next. {"limit": 10, "query": "C1CC2CN"}',
                        }
                    ],
                },
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Searching next."}],
                },
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["content"], "Searching next.")
        self.assertNotIn("tool_calls", message)
        self.assertEqual(message["response_model"], "example-org/example-model-revision-1")
        self.assertEqual(message["system_fingerprint"], "deployment-abc")

    def test_responses_parser_preserves_token_usage(self):
        response = {
            "id": "resp-test",
            "status": "completed",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 3,
                "input_tokens_details": {"cached_tokens": 3},
                "output_tokens_details": {"reasoning_tokens": 2},
                "cost": 0.001,
            },
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["response_usage"], response["usage"])
        self.assertEqual(
            usage_from_payload(message),
            {
                "input_tokens": 10,
                "output_tokens": 5,
                "total_tokens": 15,
                "cache_creation_input_tokens": 7,
                "cache_read_input_tokens": 3,
                "reasoning_tokens": 2,
                "cost": 0.001,
            },
        )

    def test_responses_transport_propagates_incomplete_output_limit(self):
        transport = _hosted_transport("responses")
        response = {
            "id": "resp-incomplete",
            "status": "incomplete",
            "incomplete_details": {"reason": "max_output_tokens"},
            "output": [{"type": "reasoning", "summary": []}],
        }

        with patch("synclaw.runners.model_runner.responses_completion", return_value=response):
            turn = asyncio.run(
                transport.complete([], tools=[], tool_choice="auto", max_output_tokens=16)
            )

        self.assertEqual(turn.finish_reason, "max_output_tokens")

    def test_responses_parser_never_estimates_cost_when_provider_omits_it(self):
        response = {
            "id": "resp-test",
            "status": "completed",
            "usage": {
                "input_tokens": 1000,
                "input_tokens_details": {"cached_tokens": 200},
                "output_tokens": 100,
                "total_tokens": 1100,
                "cost": None,
            },
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "Done."}],
                }
            ],
        }

        message = responses_to_chat_message(response)

        self.assertEqual(message["response_usage"]["cost"], None)
        self.assertEqual(usage_from_payload(message)["cache_read_input_tokens"], 200)
        self.assertNotIn("cost", usage_from_payload(message))

    def test_model_runner_defers_repair_until_after_all_tool_outputs(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            taskset = tmp_path / "tasks.jsonl"
            trace = tmp_path / "trace.jsonl"
            db_path = tmp_path / "test.sqlite"
            prepare_minimal_model_db(db_path)
            taskset.write_text(
                json.dumps(
                    {
                        "task_id": "fixture-0001",
                        "target_smiles": "CCO",
                        "system_prompt": "You plan syntheses.",
                        "prompt_fingerprint": PROMPT_FINGERPRINT,
                        "question": "Plan CCO.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            parser = argparse.ArgumentParser()
            add_arguments(parser)
            args = parser.parse_args(
                [
                    "--provider",
                    "openai",
                    "--api-mode",
                    "chat",
                    "--api-key-env",
                    "TEST_AGENTIC_SYNTHESIS_API_KEY",
                    "--taskset",
                    str(taskset),
                    "--task-index",
                    "0",
                    "--db",
                    str(db_path),
                    "--trace",
                    str(trace),
                    "--model",
                    "mock-model",
                    "--max-turns",
                    "2",
                    "--max-tokens",
                    "16",
                ]
            )
            first_message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "search_building_blocks", "arguments": "{}"},
                    },
                    {
                        "id": "call_good",
                        "type": "function",
                        "function": {
                            "name": "similarity",
                            "arguments": json.dumps({"smiles_a": "CCO", "smiles_b": "CCO"}),
                        },
                    },
                ],
            }
            seen_messages: list[list[dict]] = []

            def fake_chat_completion(**kwargs):
                seen_messages.append(list(kwargs["messages"]))
                if len(seen_messages) == 1:
                    return {"choices": [{"message": first_message}]}
                return {"choices": [{"message": {"role": "assistant", "content": "No tool call."}}]}

            with (
                patch.dict(os.environ, {"TEST_AGENTIC_SYNTHESIS_API_KEY": "test-key"}),
                patch(
                    "synclaw.runners.model_runner.chat_completion",
                    side_effect=fake_chat_completion,
                ),
            ):
                run_model_from_args(args)

            second_messages = seen_messages[1]
            call_bad_index = next(
                i
                for i, message in enumerate(second_messages)
                if message.get("tool_call_id") == "call_bad"
            )
            call_good_index = next(
                i
                for i, message in enumerate(second_messages)
                if message.get("tool_call_id") == "call_good"
            )
            self.assertLess(call_bad_index, call_good_index)
            self.assertNotIn("notice", json.loads(second_messages[call_bad_index]["content"]))
            self.assertIn(
                "tool call failed",
                json.loads(second_messages[call_good_index]["content"])["notice"],
            )
            self.assertEqual(second_messages[-1]["tool_call_id"], "call_good")


if __name__ == "__main__":
    unittest.main()
