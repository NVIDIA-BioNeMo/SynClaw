# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from synclaw.cli import build_parser


def test_search_reactions_library_does_not_consume_query():
    args = build_parser().parse_args(
        ["tool", "search-reaction-templates", "--library", "synformer", "--library", "ord", "amide coupling"]
    )

    assert args.library == ["synformer", "ord"]
    assert args.query == "amide coupling"
