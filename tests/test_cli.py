# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from synclaw.cli import build_parser


def test_tool_search_limits_default_to_the_tool_contract():
    parser = build_parser()

    assert parser.parse_args(["tool", "search-building-blocks", "CCO"]).limit == 10
    assert parser.parse_args(["tool", "search-reaction-templates"]).limit == 10
