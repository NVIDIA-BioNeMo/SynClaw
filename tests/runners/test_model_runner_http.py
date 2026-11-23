# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared POST helper behind the chat, Responses and Messages requests."""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any
from unittest.mock import patch

import pytest

from synclaw.runners import model_runner


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
    return urllib.error.HTTPError("http://x", code, "err", {}, io.BytesIO(body.encode()))  # type: ignore[arg-type]


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
