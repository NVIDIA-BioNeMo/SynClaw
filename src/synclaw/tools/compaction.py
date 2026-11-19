# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compact tool outputs before sending them back to hosted models.

The full tool result remains useful for local debugging, report rendering, and
evaluation. The model-facing result should be much smaller and should contain
only evidence the model can act on in the next turn.
"""

from __future__ import annotations

from typing import Any

MAX_RETURNED_PRODUCTS = 20
MAX_RETURNED_PRODUCT_SETS = 10
MAX_ISSUES = 8
MAX_ERROR_MESSAGE_CHARS = 1600
# Which chemistry backend and method produced a result is fixed for a run; the model can't act on it.
BACKEND_KEYS = frozenset({"backend", "chem_backend", "execution_backend", "method"})


def compact_tool_result(tool_name: str | None, result: Any) -> Any:
    """Return a compact, model-facing copy of a tool result."""

    if not isinstance(result, dict):
        return result
    name = tool_name or str(result.get("tool_name") or "")
    if name == "search_reaction_templates":
        return _compact_search_reaction_templates(result)
    if name == "search_building_blocks":
        return _compact_search_building_blocks(result)
    if name == "simulate_reaction":
        return _compact_simulate_reaction(result)
    if name == "submit_pathway":
        return _compact_submit_pathway(result)
    if name == "simulate_reverse_reaction":
        return _compact_simulate_reverse_reaction(result)
    if name == "mces":
        return _compact_mces(result)
    if name == "similarity":
        return _compact_similarity(result)
    return _compact_generic_result(result)


def compact_json_size(value: Any) -> int:
    """Approximate serialized size with stable separators."""

    import json

    return len(json.dumps(value, sort_keys=True, separators=(",", ":")))


def _compact_search_reaction_templates(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(
        result,
        [
            "query",
            "matched_template_count",
            "template_count",
            "family_count",
            "families_note",
            "families",
            "less_common_transformations",
        ],
    )
    output.update(_problems(result))
    if result.get("reverse_ranking"):
        ranking = result["reverse_ranking"]
        output["reverse_ranking"] = _copy_keys(
            ranking,
            ["applied", "reason", "enriched_result_count"],
        )
    for key in ("results", "structural_results", "text_results"):
        if key in result:
            output[key] = [_summarize_template(item) for item in result[key] or []]
    return _drop_none(output)


def _compact_search_building_blocks(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(result, ["query"])
    if result.get("query_canonical_smiles") not in (None, result.get("query")):
        output["query_canonical_smiles"] = result["query_canonical_smiles"]
    # A rejected call, such as an unparsable query, must not read like a valid query with no match.
    output.update(_problems(result))
    output["results"] = [_compact_building_block(item) for item in result.get("results") or []]
    return _drop_none(output)


def _compact_mces(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(
        result,
        [
            "valid",
            "mces_smiles",
            "similarity",
            "matched_bonds",
            "left_bonds",
            "right_bonds",
            "union_bonds",
            "timed_out",
            "error",
        ],
    )
    output["left"] = _compact_validation(result.get("left"))
    output["right"] = _compact_validation(result.get("right"))
    return _drop_none(output)


def _compact_similarity(result: dict[str, Any]) -> dict[str, Any]:
    output = {
        key: value
        for key, value in result.items()
        if key not in BACKEND_KEYS and key not in ("left", "right")
    }
    output["left"] = _compact_validation(result.get("left"))
    output["right"] = _compact_validation(result.get("right"))
    return _drop_none(output)


def _compact_simulate_reaction(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(
        result,
        [
            "valid",
            "reaction_id",
            "product_smiles",
            "product_matched",
            "used_reactant_order",
            "reactant_order_permuted",
        ],
    )
    output.update(_problems(result))
    if result.get("skipped_invalid_product_set_count"):
        output["skipped_invalid_product_set_count"] = result["skipped_invalid_product_set_count"]
    template = _compact_template(result.get("template"))
    if result.get("product_matched") and isinstance(template, dict):
        # Search hits omit the template detail, so it is returned here unless the claimed
        # product was generated; then the role order is enough to read the products.
        template = _copy_keys(template, ["reactant_roles"]) or None
    output["template"] = template
    output["reactants"] = [_compact_validation(item) for item in result.get("reactants") or []]
    if isinstance(result.get("product"), dict) and not result["product"].get("valid", True):
        output["product"] = _compact_validation(result["product"])
    output["resolved_reactants"] = [
        _compact_resolved_reactant(item) for item in result.get("resolved_reactants") or []
    ]
    output.update(_generated_products(result))
    return _drop_none(output)


def _compact_submit_pathway(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(
        result,
        [
            "accepted",
            "issues",
            "error_message",
            "final_smiles",
            "similarity_to_target",
            "identical_to_target",
            "constitutional_similarity_to_target",
            "stereo_aware_similarity_to_target",
            "reaction_validation",
            "pathway_type",
            "direct_building_block_ref",
            "direct_building_block",
            "building_block_grounded",
            "referenced_building_block_ids",
        ],
    )
    if result.get("refused"):
        output["refused"] = True
    if output.get("issues"):
        output["issues"] = _limit_list(output["issues"], MAX_ISSUES)
    if output.get("error_message"):
        output["error_message"] = _truncate(str(output["error_message"]), MAX_ERROR_MESSAGE_CHARS)
    final = result.get("final_smiles")
    for key in ("actual_final_smiles", "submitted_final_smiles"):
        if result.get(key) not in (None, final):
            output[key] = result[key]
    if result.get("ungrounded_reactants"):
        output["ungrounded_reactants"] = [
            _compact_ungrounded_reactant(item) for item in result["ungrounded_reactants"]
        ]
    # The model wrote the route; echo only the steps that failed validation.
    failed_steps = [
        _compact_validated_step(item)
        for item in result.get("steps") or []
        if not isinstance(item, dict) or item.get("valid") is not True
    ]
    if failed_steps:
        output["steps"] = failed_steps
    return _drop_none(output)


def _compact_simulate_reverse_reaction(result: dict[str, Any]) -> dict[str, Any]:
    output = _copy_keys(
        result,
        [
            "valid",
            "reaction_id",
            "issues",
            "product_smiles",
            "product_valid",
            "product_canonical_smiles",
            "reaction_executed",
            "candidate_count",
            "returned_candidate_count",
        ],
    )
    output["template_summary"] = _compact_template(result.get("template_summary"))
    if isinstance(result.get("reverse_planning_summary"), dict):
        output["reverse_planning_summary"] = {
            key: value
            for key, value in result["reverse_planning_summary"].items()
            if key != "recommended_next_step"
        }
    if isinstance(result.get("building_block_match_summary"), dict):
        output["building_block_match_summary"] = {
            key: value
            for key, value in result["building_block_match_summary"].items()
            if key != "note"
        }
    output["candidate_reactant_sets"] = [
        _compact_reverse_candidate(item) for item in result.get("candidate_reactant_sets") or []
    ]
    return _drop_none(output)


def _compact_generic_result(result: dict[str, Any]) -> dict[str, Any]:
    output = {key: value for key, value in result.items() if key not in BACKEND_KEYS}
    if isinstance(output.get("template"), dict):
        output["template"] = _compact_template(output["template"])
    if isinstance(output.get("results"), list):
        output["results"] = [
            _compact_template(item) if isinstance(item, dict) and item.get("template_id") else item
            for item in output["results"]
        ]
    return _drop_none(output)


def _compact_template(template: Any) -> Any:
    if not isinstance(template, dict):
        return template
    output = _copy_template_keys(
        template,
        [
            "template_id",
            "name",
            "description",
            "reaction_family",
            "smarts",
            "execution_hints",
        ],
    )
    roles = template.get("reactant_roles")
    if isinstance(roles, list):
        # Roles are listed in reactant order, so position and atom maps add nothing.
        output["reactant_roles"] = [
            _copy_keys(role, ["role", "smarts"]) if isinstance(role, dict) else role
            for role in roles
        ]
    if template.get("introduces_unmapped_product_atoms"):
        output["introduces_unmapped_product_atoms"] = True
    return _drop_empty(output)


def _summarize_template(template: Any) -> Any:
    """Return a search hit without the detail simulate_reaction returns when nothing is generated."""

    if not isinstance(template, dict):
        return template
    output = _copy_template_keys(template, ["template_id", "name", "reaction_family", "smarts"])
    roles = template.get("reactant_roles")
    if isinstance(roles, list):
        # Role SMARTS are the reactant side of smarts, in this order.
        output["reactant_roles"] = [
            role.get("role") if isinstance(role, dict) else role for role in roles
        ]
    if template.get("introduces_unmapped_product_atoms"):
        output["introduces_unmapped_product_atoms"] = True
    structure = template.get("structure_match")
    if isinstance(structure, dict):
        # Every hit matches all supplied molecules, so only the score and any forward run differ.
        output["structure_match"] = _copy_keys(structure, ["score"])
        forward = structure.get("forward_simulation")
        if isinstance(forward, dict):
            output["structure_match"]["forward_simulation"] = _drop_empty(
                _copy_keys(
                    forward,
                    [
                        "reaction_executed",
                        "generated_products",
                        "requested_products_generated",
                        "requested_product_set_generated",
                        "issues",
                    ],
                )
            )
    if isinstance(template.get("reverse_usefulness"), dict):
        output["reverse_usefulness"] = _drop_empty(
            _copy_keys(
                template["reverse_usefulness"],
                [
                    "score",
                    "fully_grounded_candidate_sets",
                    "best_candidate_reactants",
                    "best_candidate_exact_submit_references",
                    "issues",
                ],
            )
        )
    return _drop_empty(output)


def _compact_building_block(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    # One reference and one SMILES: submit_pathway accepts either, and the ID is in the reference.
    output = _copy_keys(item, ["submit_reference", "match_type", "similarity"])
    output["smiles"] = item.get("canonical_smiles") or item.get("smiles")
    return _drop_none(output)


def _compact_validation(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return _drop_none(
        _copy_keys(
            item,
            ["valid", "canonical_smiles", "error"],
        )
    )


def _compact_resolved_reactant(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    output = _copy_keys(item, ["ref", "smiles", "source", "issue"])
    block = item.get("building_block")
    if isinstance(block, dict):
        output["building_block"] = _copy_keys(block, ["id", "smiles", "canonical_smiles"])
    return _drop_none(output)


def _compact_ungrounded_reactant(item: Any) -> Any:
    if not isinstance(item, dict):
        return item
    return _drop_none(_copy_keys(item, ["step_index", "ref", "smiles", "source", "issue"]))


def _compact_validated_step(step: Any) -> Any:
    if not isinstance(step, dict):
        return step
    output = _copy_keys(
        step,
        [
            "step_index",
            "valid",
            "reaction_id",
            "issues",
            "product_smiles",
            "actual_product_smiles",
            "product_matched",
            "used_reactant_order",
            "reactant_order_permuted",
            "skipped_invalid_product_set_count",
        ],
    )
    output.update(_generated_products(step))
    output["resolved_reactants"] = [
        _compact_resolved_reactant(item) for item in step.get("resolved_reactants") or []
    ]
    submitted = step.get("submitted_step")
    if isinstance(submitted, dict):
        output["submitted_step_summary"] = _copy_keys(
            submitted,
            ["reaction_id", "reactants", "product_smiles"],
        )
    return _drop_none(output)


def _compact_reverse_candidate(candidate: Any) -> Any:
    if not isinstance(candidate, dict):
        return candidate
    output = _copy_keys(
        candidate,
        [
            "reactants",
            "all_reactants_have_exact_building_block_match",
            "all_precursors_grounded",
        ],
    )
    details = []
    for item in candidate.get("reactant_details") or []:
        if not isinstance(item, dict):
            continue
        details.append(
            _drop_none(
                {
                    "smiles": item.get("smiles"),
                    "valid_smiles": item.get("valid_smiles"),
                    "has_exact_building_block_match": item.get("has_exact_building_block_match"),
                    "exact_building_block_matches": [
                        _compact_building_block(match)
                        for match in item.get("exact_building_block_matches") or []
                    ],
                    "top_near_building_block_matches": [
                        _compact_building_block(match)
                        for match in item.get("top_near_building_block_matches") or []
                    ],
                }
            )
        )
    if details:
        output["reactant_details"] = details
    return _drop_none(output)


def _problems(result: dict[str, Any]) -> dict[str, Any]:
    """Report validity and issues only when something went wrong."""

    output: dict[str, Any] = {}
    if result.get("valid") is False:
        output["valid"] = False
    if result.get("issues"):
        output["issues"] = result["issues"]
    return output


def _generated_products(result: dict[str, Any]) -> dict[str, Any]:
    """The flat product list, or product sets when some outcome has coproducts.

    The flat list is the union of the sets, so only one is needed. Counts are kept
    only when the returned list is truncated.
    """

    sets = result.get("generated_product_sets")
    if isinstance(sets, list) and any(isinstance(item, list) and len(item) > 1 for item in sets):
        key, limit = "generated_product_sets", MAX_RETURNED_PRODUCT_SETS
    else:
        key, limit = "generated_products", MAX_RETURNED_PRODUCTS
    values = result.get(key)
    output: dict[str, Any] = {key: _limit_list(values, limit)}
    count_key = (
        "generated_product_set_count"
        if key == "generated_product_sets"
        else "generated_product_count"
    )
    total = result.get(count_key, len(values) if isinstance(values, list) else 0)
    if isinstance(total, int) and total > len(output[key]):
        output[count_key] = total
    return output


def _copy_keys(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source.get(key) for key in keys if key in source}


def _copy_template_keys(template: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    """Copy template keys, naming the catalog's ``template_id`` as the ``reaction_id`` a step takes.

    The catalog calls a template's identifier ``template_id``. The model only ever submits
    and simulates by ``reaction_id``, so that is the one name it is shown.
    """

    output = _copy_keys(template, keys)
    if "template_id" in output:
        output = {"reaction_id": output.pop("template_id"), **output}
    return output


def _drop_none(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item is not None}


def _drop_empty(value: dict[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if item not in (None, "", [], {})}


def _limit_list(value: Any, limit: int) -> list[Any]:
    if not isinstance(value, list):
        return []
    return value[: max(0, int(limit))]


def _truncate(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 20].rstrip() + " ... [truncated]"
