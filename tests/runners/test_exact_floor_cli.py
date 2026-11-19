# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The exact floor must be usable through both hosted-run and evaluation CLIs."""

import pytest

from synclaw.cli import build_parser


@pytest.mark.parametrize("floor", [None, "exact", "model", "0", "0.8", "1.0"])
def test_run_and_eval_cli_accept_the_same_floor(floor):
    parser = build_parser()
    floor_args = [] if floor is None else ["--stop-condition", floor]
    run = parser.parse_args(
        [
            "run-model",
            "--model",
            "example",
            "--taskset",
            "tasks.jsonl",
            *floor_args,
        ]
    )
    evaluate = parser.parse_args(
        [
            "eval-trace",
            "trace.jsonl",
            *floor_args,
        ]
    )
    assert run.stop_condition == evaluate.stop_condition
    assert run.stop_condition == (
        "exact" if floor is None else floor if floor in ("exact", "model") else float(floor)
    )
