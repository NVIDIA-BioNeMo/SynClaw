# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from synclaw.core import db
from synclaw.rollout.loop import call_local_tool
from synclaw.tools.compaction import compact_tool_result
from synclaw.tools.synthesis import SynthesisTools


def test_reaction_template_search_returns_short_hits():
    result = {
        "valid": True,
        "query": "amide",
        "search_mode": "text",
        "count": 1,
        "results": [
            {
                "reaction_id": "synformer:R008",
                "name": "Amide coupling",
                "description": "Couples amines and carboxylic acids.",
                "reaction_family": "amide formation",
                "smarts": "[N:1].[C:2](=O)O>>[N:1][C:2]=O",
                "execution_hints": ["Coupling reagent is implicit."],
                "reactant_roles": [
                    {"position": 1, "role": "amine", "smarts": "[N:1]"},
                    {"position": 2, "role": "acid", "smarts": "[C:2](=O)O"},
                ],
                "structure_match": {
                    "score": 20.0,
                    "reactant_match": True,
                    "product_match": True,
                    "forward_simulation": None,
                },
                "reverse_usefulness": {
                    "score": 0.0,
                    "reverse_executed": True,
                    "candidate_sets": 1,
                    "fully_grounded_candidate_sets": 0,
                    "precursors_with_exact_bb_matches": 0,
                    "best_candidate_reactants": ["NC", "*C(=O)C"],
                    "best_candidate_exact_submit_references": [],
                    "issues": [],
                },
                "reactant_count": 2,
                "product_count": 1,
                "introduces_unmapped_product_atoms": False,
                "query_hints": ["amide coupling"],
                "tags": ["amide coupling"],
                "tags_json": '["amide coupling"]',
                "metadata": {"annotation": {"description": "duplicate"}},
                "metadata_json": '{"annotation":{"description":"duplicate"}}',
                "library": "synformer",
            }
        ],
    }

    compact = compact_tool_result("search_reaction_templates", result)
    hit = compact["results"][0]

    simulated = compact_tool_result(
        "simulate_reaction", {"valid": False, "template": result["results"][0]}
    )

    assert hit == {
        "reaction_id": "synformer:R008",
        "name": "Amide coupling",
        "reaction_family": "amide formation",
        "smarts": "[N:1].[C:2](=O)O>>[N:1][C:2]=O",
        "reactant_roles": ["amine", "acid"],
        "structure_match": {"score": 20.0},
        "reverse_usefulness": {
            "score": 0.0,
            "fully_grounded_candidate_sets": 0,
            "best_candidate_reactants": ["NC", "*C(=O)C"],
        },
    }
    assert simulated["template"]["description"] == "Couples amines and carboxylic acids."
    assert simulated["template"]["execution_hints"] == ["Coupling reagent is implicit."]
    assert simulated["template"]["reactant_roles"][1] == {"role": "acid", "smarts": "[C:2](=O)O"}


def test_submit_pathway_compaction_keeps_validation_summary_without_verbose_echoes():
    result = {
        "accepted": False,
        "issues": ["step 1 invalid"],
        "error_message": "The submitted pathway was rejected. " * 100,
        "pathway_text": "A very long route explanation. " * 100,
        "actual_final_smiles": "CCO",
        "submitted_final": {"valid": True, "canonical_smiles": "CCO"},
        "final": {"valid": True, "canonical_smiles": "CCO"},
        "reaction_validation": "invalid",
        "building_block_grounded": True,
        "steps": [
            {
                "step_index": 1,
                "valid": False,
                "reaction_id": "synformer:R001",
                "issues": ["Claimed product was not generated"],
                "template": {"metadata_json": "huge", "reaction_id": "synformer:R001"},
                "submitted_step": {
                    "reaction_id": "synformer:R001",
                    "reactants": ["BB:1"],
                    "product_smiles": "CCO",
                },
                "resolved_reactants": [
                    {
                        "ref": "BB:1",
                        "smiles": "CO",
                        "source": "building_block",
                        "building_block": {"id": 1, "source_file": "data/enamine.txt"},
                    }
                ],
                "generated_products": ["CCN"],
            }
        ],
    }

    compact = compact_tool_result("submit_pathway", result)
    step = compact["steps"][0]

    assert compact["accepted"] is False
    assert compact["actual_final_smiles"] == "CCO"
    assert "pathway_text" not in compact
    assert "pathway_text_summary" not in compact
    assert "submitted_final" not in compact
    assert "final" not in compact
    assert len(compact["error_message"]) < len(result["error_message"])
    assert "template" not in step
    assert step["submitted_step_summary"]["reactants"] == ["BB:1"]
    assert step["resolved_reactants"][0]["building_block"] == {"id": 1}


def test_simulate_reaction_compaction_keeps_products_and_short_template():
    result = {
        "valid": True,
        "reaction_id": "synformer:R008",
        "template": {
            "reaction_id": "synformer:R008",
            "name": "Amide coupling",
            "metadata_json": "huge",
        },
        "reactants": [{"valid": True, "canonical_smiles": "CN", "properties": {"heavy": 2}}],
        "resolved_reactants": [
            {
                "ref": "BB:1",
                "smiles": "CN",
                "source": "building_block",
                "building_block": {"id": 1, "source_file": "data/enamine.txt"},
            }
        ],
        "generated_products": ["CNC=O"],
        "generated_product_count": 1,
    }

    compact = compact_tool_result("simulate_reaction", result)

    assert compact["template"] == {"reaction_id": "synformer:R008", "name": "Amide coupling"}
    assert compact["reactants"] == [{"valid": True, "canonical_smiles": "CN"}]
    assert compact["resolved_reactants"][0]["building_block"] == {"id": 1}
    assert compact["generated_products"] == ["CNC=O"]


def test_mces_compaction_keeps_actionable_overlap_fields():
    result = {
        "valid": True,
        "mces_smiles": "CCC",
        "similarity": 0.5,
        "matched_bonds": 2,
        "left_bonds": 3,
        "right_bonds": 3,
        "union_bonds": 4,
        "timed_out": False,
        "timeout_seconds": 20,
        "method": "rdkit_rascal_mces_bond_tanimoto",
        "left": {
            "valid": True,
            "canonical_smiles": "CCCO",
            "properties": {"formula": "C3H8O"},
        },
        "right": {
            "valid": True,
            "canonical_smiles": "CCCN",
            "properties": {"formula": "C3H9N"},
        },
    }

    compact = compact_tool_result("mces", result)

    assert compact["mces_smiles"] == "CCC"
    assert compact["similarity"] == 0.5
    assert compact["matched_bonds"] == 2
    assert compact["left"] == {"valid": True, "canonical_smiles": "CCCO"}
    assert compact["right"] == {"valid": True, "canonical_smiles": "CCCN"}


def test_building_block_compaction_keeps_one_reference_one_smiles_and_the_match():
    hit = {
        "id": 7,
        "submit_reference": "BB:7",
        "submit_as": {"preferred": "BB:7", "raw_smiles": "OCC", "canonical_smiles": "CCO"},
        "smiles": "OCC",
        "canonical_smiles": "CCO",
        "match_type": "exact",
        "similarity": 1.0,
        "heavy_atom_count": 3,
    }

    compact = compact_tool_result("search_building_blocks", {"query": "CCO", "results": [hit]})

    assert compact["results"] == [
        {
            "submit_reference": "BB:7",
            "smiles": "CCO",
            "match_type": "exact",
            "similarity": 1.0,
        }
    ]



def test_building_block_compaction_keeps_a_rejected_query_error(tmp_path):
    database = tmp_path / "test.sqlite"
    (tmp_path / "bbs.txt").write_text("SMILES\nCCO\n", encoding="utf-8")
    db.ingest_building_blocks(tmp_path / "bbs.txt", database)
    db.prepare_building_block_search(database)

    rejected = call_local_tool(SynthesisTools(database), "search_building_blocks", {"query": "C1CC("})
    compact = compact_tool_result("search_building_blocks", rejected)

    assert compact["valid"] is False
    assert "valid SMILES" in compact["issues"][0]
    assert compact["results"] == []


def test_template_compaction_omits_empty_annotation_fields():
    template = {
        "reaction_id": "demo:R001",
        "name": "R001",
        "description": "",
        "reaction_family": "",
        "smarts": "[C:1]=[O:2]>>[C:1][O:2]",
        "execution_hints": [],
        "reactant_roles": [],
    }

    search = compact_tool_result("search_reaction_templates", {"query": "R001", "results": [template]})
    simulate = compact_tool_result(
        "simulate_reaction", {"reaction_id": "demo:R001", "valid": True, "template": template}
    )

    expected = {"reaction_id": "demo:R001", "name": "R001", "smarts": "[C:1]=[O:2]>>[C:1][O:2]"}
    assert search["results"] == [expected]
    assert simulate["template"] == expected

def test_reverse_planning_compaction_drops_fixed_next_step_text():
    next_step = "No fully grounded reverse candidate was found."
    search = compact_tool_result(
        "search_reaction_templates",
        {
            "reverse_ranking": {
                "applied": True,
                "ranking_note": "Re-ranked.",
                "enriched_result_count": 1,
            },
            "results": [
                {
                    "reaction_id": "ord:R001",
                    "reverse_usefulness": {
                        "fully_grounded_candidate_sets": 0,
                        "recommended_next_step": next_step,
                    },
                }
            ],
        },
    )
    reverse = compact_tool_result(
        "simulate_reverse_reaction",
        {
            "reverse_planning_summary": {
                "has_fully_grounded_candidate": False,
                "recommended_next_step": next_step,
            }
        },
    )

    assert search["reverse_ranking"] == {"applied": True, "enriched_result_count": 1}
    assert search["results"][0]["reverse_usefulness"] == {"fully_grounded_candidate_sets": 0}
    assert reverse["reverse_planning_summary"] == {"has_fully_grounded_candidate": False}


def test_compaction_drops_echoed_settings_and_derivable_counts():
    blocks = compact_tool_result(
        "search_building_blocks",
        {
            "query": "OCC",
            "query_canonical_smiles": "CCO",
            "limit": 10,
            "include_near": True,
            "min_similarity": 0.15,
            "count": 0,
            "exact_match_count": 0,
            "near_match_count": 0,
            "chem_backend": "rdkit",
            "results": [],
        },
    )
    templates = compact_tool_result(
        "search_reaction_templates",
        {
            "valid": True,
            "issues": [],
            "query": "amide",
            "search_mode": "text",
            "limit": 10,
            "count": 0,
            "matched_template_count": 0,
            "scanned_template_count": 501,
            "results": [],
        },
    )
    similarity = compact_tool_result(
        "similarity",
        {
            "similarity": 0.5,
            "method": "rdkit_morgan_tanimoto",
            "left": {"backend": "rdkit", "canonical_smiles": "CCO"},
        },
    )

    assert blocks == {"query": "OCC", "query_canonical_smiles": "CCO", "results": []}
    assert templates == {"query": "amide", "matched_template_count": 0, "results": []}
    assert similarity == {"similarity": 0.5, "left": {"canonical_smiles": "CCO"}}


def test_simulate_reaction_compaction_returns_one_product_listing():
    single = compact_tool_result(
        "simulate_reaction",
        {
            "valid": True,
            "issues": [],
            "product": {"valid": True, "canonical_smiles": "CCO"},
            "generated_products": ["CCO", "CCN"],
            "generated_product_sets": [["CCO"], ["CCN"]],
            "generated_product_count": 2,
            "generated_product_set_count": 2,
            "skipped_invalid_product_set_count": 0,
            "expected_reactant_count": 2,
            "provided_reactant_count": 2,
            "attempted_reactant_orders": 1,
        },
    )
    coproducts = compact_tool_result(
        "simulate_reaction",
        {
            "valid": True,
            "generated_products": ["CCO", "O"],
            "generated_product_sets": [["CCO", "O"]] * 12,
            "generated_product_set_count": 12,
        },
    )

    assert {key: value for key, value in single.items() if value != []} == {
        "valid": True,
        "generated_products": ["CCO", "CCN"],
    }
    assert "generated_products" not in coproducts
    assert len(coproducts["generated_product_sets"]) == 10
    assert coproducts["generated_product_set_count"] == 12


def test_compaction_trims_template_detail_and_echoed_routes():
    template = {
        "reaction_id": "ord:R011",
        "reactant_count": 1,
        "product_count": 1,
        "reactant_roles": [
            {"position": 1, "role": "aryl bromide", "smarts": "Br-c", "mapped_atoms": [1]}
        ],
        "structure_match": {"score": 10.0, "product_match": True, "provided_product_count": 1},
    }
    search = compact_tool_result("search_reaction_templates", {"results": [template]})
    simulated = compact_tool_result(
        "simulate_reaction",
        {"valid": True, "template": template, "generated_products": ["c"], "product_matched": True},
    )
    unmatched = compact_tool_result(
        "simulate_reaction",
        {
            "valid": True,
            "template": template,
            "generated_products": ["c"],
            "product_matched": False,
        },
    )
    accepted = compact_tool_result(
        "submit_pathway",
        {
            "accepted": True,
            "final_smiles": "CCO",
            "actual_final_smiles": "CCO",
            "submitted_final_smiles": "CCO",
            "pathway_text": "Route.",
            "steps": [{"step_index": 1, "valid": True, "reaction_id": "ord:R011"}],
        },
    )
    similarity = compact_tool_result(
        "similarity",
        {
            "similarity": 0.5,
            "left": {"valid": True, "input": "OCC", "canonical_smiles": "CCO", "properties": {}},
        },
    )
    reverse = compact_tool_result(
        "simulate_reverse_reaction",
        {
            "building_block_match_summary": {
                "exact_match_count": 0,
                "note": "Planning evidence only.",
            },
            "candidate_reactant_sets": [
                {
                    "reactants": ["Brc1ccccc1"],
                    "grounding_summary": "Lacks exact matches.",
                    "precursor_grounding": [{"smiles": "Brc1ccccc1"}],
                    "reactant_details": [
                        {"smiles": "Brc1ccccc1", "grounding_note": "No exact match."}
                    ],
                }
            ],
        },
    )

    assert search["results"][0] == {
        "reaction_id": "ord:R011",
        "reactant_roles": ["aryl bromide"],
        "structure_match": {"score": 10.0},
    }
    assert simulated["template"] == {"reactant_roles": [{"role": "aryl bromide", "smarts": "Br-c"}]}
    assert unmatched["template"] == {
        "reaction_id": "ord:R011",
        "reactant_roles": [{"role": "aryl bromide", "smarts": "Br-c"}],
    }
    assert accepted == {"accepted": True, "final_smiles": "CCO"}
    assert similarity["left"] == {"valid": True, "canonical_smiles": "CCO"}
    assert reverse["building_block_match_summary"] == {"exact_match_count": 0}
    assert reverse["candidate_reactant_sets"] == [
        {
            "reactants": ["Brc1ccccc1"],
            "reactant_details": [
                {
                    "smiles": "Brc1ccccc1",
                    "exact_building_block_matches": [],
                    "top_near_building_block_matches": [],
                }
            ],
        }
    ]


def test_submit_pathway_compaction_shows_whether_the_product_is_the_exact_target():
    result = {
        "accepted": True,
        "actual_final_smiles": "C[C@H](N)O",
        "similarity_to_target": 1.0,
        "identical_to_target": False,
        "reaction_validation": "valid",
    }

    compact = compact_tool_result("submit_pathway", result)

    # Similarity 1.0 can still miss the target, e.g. as the other stereoisomer.
    assert compact["similarity_to_target"] == 1.0
    assert compact["identical_to_target"] is False
