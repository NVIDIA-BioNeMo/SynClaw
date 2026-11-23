# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import pytest

from synclaw import cli
from synclaw.cli import build_parser
from synclaw.tasks.targets import GenerationOptions
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS


def test_tool_flags_default_to_the_tool_contract():
    parser = build_parser()
    search = parser.parse_args(["tool", "search-building-blocks", "CCO"])
    reverse = parser.parse_args(
        ["tool", "simulate-reverse-reaction", "--reaction-id", "R1", "--product-smiles", "CC"]
    )
    forward = parser.parse_args(
        ["tool", "simulate-reaction", "--reaction-id", "R1", "--reactants-json", "[]"]
    )

    assert (search.limit, search.min_similarity) == (
        TOOL_ARGUMENT_DEFAULTS["search_building_blocks"]["limit"],
        TOOL_ARGUMENT_DEFAULTS["search_building_blocks"]["min_similarity"],
    )
    assert parser.parse_args(["tool", "search-reaction-templates"]).limit == 10
    assert forward.max_products == TOOL_ARGUMENT_DEFAULTS["simulate_reaction"]["max_products"]
    defaults = TOOL_ARGUMENT_DEFAULTS["simulate_reverse_reaction"]
    assert (reverse.max_outcomes, reverse.min_similarity, reverse.top_near_matches) == (
        defaults["max_outcomes"],
        defaults["min_similarity"],
        defaults["top_near_matches"],
    )


def test_arguments_a_tool_rejects_are_a_failed_command(tmp_path, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(
            ["tool", "submit-pathway", "--steps-json", '["R001"]', "--db", str(tmp_path / "x.db")]
        )

    assert exit_info.value.code == 1
    assert "must be an array of objects" in capsys.readouterr().err


def test_debug_shows_the_traceback_instead_of_one_line():
    with pytest.raises(RuntimeError, match="--steps-json"):
        cli.main(["--debug", "tool", "submit-pathway", "--steps-json", "@/no/such/steps.json"])


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["tool", "simulate-reaction", "--reaction-id", "R1", "--reactants-json", "[BB:1"],
            "--reactants-json:",
        ),
        (["tool", "submit-pathway", "--steps-json", "@/no/such/steps.json"], "--steps-json:"),
        (["generate-targets", "--matrix-dir", "/no/such/matrix", "--output", "t.tsv"], "error:"),
    ],
)
def test_bad_input_is_a_one_line_error_not_a_traceback(argv, message, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)

    assert exit_info.value.code == 1
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("group", ["db", "tool", "mcp"])
def test_a_command_group_without_a_subcommand_says_so(group, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main([group])

    assert exit_info.value.code == 2
    assert "required: COMMAND" in capsys.readouterr().err


def test_generation_options_accept_a_comma_separated_library_string():
    assert GenerationOptions(libraries="ord, synformer").libraries == ("ord", "synformer")
