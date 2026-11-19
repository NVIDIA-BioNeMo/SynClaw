# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A template is named reaction_id to the model; the catalog's template_id never reaches it."""

from typing import Any

import pytest

from synclaw.core import db, reactions
from synclaw.tools.compaction import compact_tool_result
from synclaw.tools.synthesis import SynthesisTools

DIOL = "OCCCCC(C)O"  # BB:1
ALDEHYDE = "CC(O)CCCC=O"


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    root = tmp_path_factory.mktemp("reaction-id-naming")
    db_path = root / "fixture.sqlite"
    templates = root / "templates.txt"
    stock = root / "stock.txt"
    templates.write_text("[C:1][O:2]>>[C:1]=[O:2]\n", encoding="utf-8")
    stock.write_text(f"SMILES\n{DIOL}\n", encoding="utf-8")
    reactions.ingest_reaction_templates(templates, db_path=db_path, library="demo")
    db.ingest_building_blocks(stock, db_path)
    db.prepare_building_block_search(db_path)
    return SynthesisTools(db_path)


def _keys(value: Any) -> set[str]:
    """Every dict key anywhere in a nested result."""

    if isinstance(value, dict):
        return set(value) | {key for item in value.values() for key in _keys(item)}
    if isinstance(value, list):
        return {key for item in value for key in _keys(item)}
    return set()


@pytest.mark.parametrize("tool", ["search_reaction_templates", "simulate_reaction", "simulate_reverse_reaction"])
def test_the_model_is_shown_reaction_id_for_a_template(tools, tool):
    if tool == "search_reaction_templates":
        result = tools.search_reaction_templates(query="demo:R000")
    elif tool == "simulate_reaction":
        result = tools.simulate_reaction("demo:R000", [DIOL])
    else:
        result = tools.simulate_reverse_reaction("demo:R000", ALDEHYDE)
    # The tool's own result, which the trace keeps, still names the catalog's identifier.
    assert "template_id" in _keys(result)

    compact = compact_tool_result(tool, result)

    assert "template_id" not in _keys(compact)
    assert "reaction_id" in _keys(compact)


def test_a_step_must_name_its_template_as_reaction_id(tools):
    step = {"reactants": ["BB:1"], "product_smiles": ALDEHYDE}

    named = tools.submit_pathway(steps=[{"reaction_id": "demo:R000", **step}])
    catalog_named = tools.submit_pathway(steps=[{"template_id": "demo:R000", **step}])

    assert named["accepted"] is True
    assert catalog_named["accepted"] is False
    assert "step 1: reaction_id is required" in catalog_named["steps"][0]["issues"]
