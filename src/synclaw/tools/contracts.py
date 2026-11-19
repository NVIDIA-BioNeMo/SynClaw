# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical model-visible contracts and dispatch defaults for BioNeMo-SynClaw tools."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_building_blocks",
            "description": (
                "Search purchasable building blocks by exact and near-match SMILES. Starting "
                "reactants of a pathway must be exact building-block matches; pass a hit's "
                "submit_reference (BB:<id>) or its exact SMILES. A miss means the molecule is not "
                "purchasable in this database, not that it cannot be used: it can still be made as "
                "an intermediate in an earlier step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Molecule SMILES."},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                    "include_near": {"type": "boolean", "default": True},
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.15,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "similarity",
            "description": (
                "Compute molecular fingerprint similarity between two SMILES strings, for example "
                "a candidate product and the target. similarity is the Tanimoto similarity of "
                "chirality-aware Morgan fingerprints (radius 2, 2048 bits); "
                "constitutional_similarity is the same measure ignoring stereochemistry. When "
                "several candidate routes are possible, compare their products with the target "
                "before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles_a": {"type": "string"},
                    "smiles_b": {"type": "string"},
                },
                "required": ["smiles_a", "smiles_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mces",
            "description": (
                "Find the full Maximum Common Edge Subgraph between two molecules, for example a "
                "candidate product and the target, and compute its normalized common-bond overlap: "
                "the matched bonds divided by the union of both molecules' bonds. The MCES SMILES "
                "may contain dot-separated fragments. When several candidate routes are possible, "
                "compare their products with the target before submitting."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles_a": {"type": "string"},
                    "smiles_b": {"type": "string"},
                },
                "required": ["smiles_a", "smiles_b"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reaction_templates",
            "description": (
                "Search available reaction templates by text and/or structural molecule "
                "constraints, or call it with no arguments to list the largest reaction families "
                "and their search terms. Use query for IDs, names, descriptions, tags, or SMARTS. "
                "Use product_smiles and/or reactant_smiles when you need templates whose SMARTS "
                "roles structurally match molecules. When query and molecule fields are both "
                "supplied, they are searched independently and returned as structural_results and "
                "text_results. Product-side structural searches are ranked by shallow reverse "
                "simulation and exact building-block grounding evidence. Search by "
                "functional-group transformation and product motif rather than by the target's "
                "name. Results list each template's reaction_id, name, family, SMARTS and reactant "
                "role names in reactant order; they omit the full description and execution hints."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Optional text, reaction family, template ID, search term, or SMARTS "
                            "query."
                        ),
                    },
                    "reactant_smiles": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Optional reactant or precursor SMILES to match against template "
                            "reactant roles."
                        ),
                    },
                    "product_smiles": {
                        "anyOf": [
                            {"type": "string"},
                            {"type": "array", "items": {"type": "string"}},
                        ],
                        "description": (
                            "Optional product SMILES to match against template product roles."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 20,
                        "default": 10,
                    },
                    "run_forward": {
                        "type": "boolean",
                        "default": False,
                        "description": (
                            "If all template reactants are provided, run forward simulation and "
                            "report whether requested products are generated."
                        ),
                    },
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_reaction",
            "description": (
                "Run a reaction template forward on reactants and return the generated product "
                "SMILES. reaction_id is a library template ID or a reaction SMARTS; a SMARTS is "
                "used only if a library template generates the same product from these reactants "
                "(with no claimed product, every product the SMARTS generates), and that "
                "template's ID is returned. Reactants follow the template's reactant-role order; "
                "if that order generates nothing, alternate orders are tried. Pass product_smiles "
                "to check a claimed product. A pathway step is valid only if this tool generates "
                "its product_smiles from the same reaction_id and reactants. Returns the "
                "template's description and execution hints unless the claimed product was "
                "generated."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_id": {
                        "type": "string",
                        "description": "Library template ID or reaction SMARTS.",
                    },
                    "reactants": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Reactant SMILES or BB:<id> references, in role order.",
                    },
                    "product_smiles": {
                        "type": "string",
                        "description": (
                            "Optional claimed product to check against the generated products."
                        ),
                    },
                    "max_products": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 100,
                    },
                },
                "required": ["reaction_id", "reactants"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_reverse_reaction",
            "description": (
                "Run a reaction template backwards from product SMILES to candidate reactant sets, "
                "with each precursor checked against the building blocks (exact and near matches). "
                "Once a template's product role matches the target or an intermediate, one call "
                "returns candidate precursor sets for the template (up to max_outcomes) and shows "
                "which precursors are purchasable, in place of guessing precursors and searching "
                "for each one. Precursors without an exact match may still be made in earlier "
                "steps, and a precursor set is not guaranteed to work forward: the step is valid "
                "only if the template generates the product from those reactants."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reaction_id": {
                        "type": "string",
                        "description": "Library template ID.",
                    },
                    "product_smiles": {
                        "anyOf": [
                            {"type": "string"},
                            {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                            },
                        ],
                        "description": (
                            "Product SMILES. Provide all product components for multi-product "
                            "templates."
                        ),
                    },
                    "max_outcomes": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 50,
                    },
                    "include_near": {"type": "boolean", "default": True},
                    "min_similarity": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.15,
                    },
                    "top_near_matches": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 10,
                        "default": 3,
                    },
                },
                "required": ["reaction_id", "product_smiles"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_pathway",
            "description": (
                "Submit a synthesis pathway for scoring, or a building block for direct purchase. "
                "Each step names a reaction_id, its reactants in reactant-role order, and the "
                "product_smiles that reaction generates from them. A reaction_id is a library "
                "template ID or a reaction SMARTS; like a building block, a SMARTS must match the "
                "library: it is accepted only if a library template generates the same product "
                "from the same reactants. Starting reactants must be purchasable building blocks, "
                "given as BB:<id> or exact building-block SMILES; use INT:<step_number> for an "
                "earlier step's product. For direct purchase, pass steps: [] and one exact "
                "building_block. The route is scored on the RDKit-generated product of its final "
                "step. If you submit more than once, the best accepted route is kept, so a later "
                "submission cannot replace a better earlier one."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "reaction_id": {
                                    "type": "string",
                                    "description": "Library template ID or reaction SMARTS.",
                                },
                                "reactants": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "minItems": 1,
                                    "description": (
                                        "BB:<id>, exact building-block SMILES or INT:<step_number>."
                                    ),
                                },
                                "product_smiles": {"type": "string"},
                            },
                            "required": ["reaction_id", "reactants", "product_smiles"],
                            "additionalProperties": False,
                        },
                    },
                    "building_block": {
                        "type": "string",
                        "description": (
                            "Use only for direct-purchase zero-step submissions; provide BB:<id> "
                            "or exact building-block SMILES. Omit for reaction pathways."
                        ),
                    },
                    "pathway_text": {
                        "type": "string",
                        "description": (
                            "Natural-language explanation of the route: the overall strategy and "
                            "what each step does."
                        ),
                    },
                },
                "required": ["pathway_text"],
                "additionalProperties": False,
            },
        },
    },
]

RESPONSES_TOOL_SCHEMAS = [
    {
        "type": "function",
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "parameters": tool["function"]["parameters"],
    }
    for tool in TOOL_SCHEMAS
]

MESSAGES_TOOL_SCHEMAS = [
    {
        "name": tool["function"]["name"],
        "description": tool["function"]["description"],
        "input_schema": tool["function"]["parameters"],
    }
    for tool in TOOL_SCHEMAS
]

TOOL_NAMES = tuple(tool["function"]["name"] for tool in TOOL_SCHEMAS)
REQUIRED_TOOL_NAMES = frozenset({"submit_pathway"})

TOOL_ARGUMENT_SPECS: dict[str, dict[str, dict[str, Any]]] = {
    tool["function"]["name"]: tool["function"]["parameters"].get("properties", {})
    for tool in TOOL_SCHEMAS
}
TOOL_ARGUMENT_DEFAULTS: dict[str, dict[str, Any]] = {
    name: {key: spec["default"] for key, spec in specs.items() if "default" in spec}
    for name, specs in TOOL_ARGUMENT_SPECS.items()
}
TOOL_ARGUMENT_DEFAULTS["submit_pathway"]["steps"] = []
TOOL_ALLOWED_ARGUMENTS: dict[str, frozenset[str]] = {
    name: frozenset(specs) for name, specs in TOOL_ARGUMENT_SPECS.items()
}


def clamp_to_contract(tool_name: str, argument: str, value: Any) -> int:
    """Clamp a whole-number argument to the bounds its tool schema advertises.

    Anything else (a boolean, string, fraction, NaN or infinity) raises ``ValueError``
    instead of being coerced, so the caller sees the contract rather than a substituted value.
    """

    spec = TOOL_ARGUMENT_SPECS[tool_name][argument]
    whole = isinstance(value, int) or (isinstance(value, float) and value.is_integer())
    if isinstance(value, bool) or not whole:
        raise ValueError(
            f"{tool_name} argument '{argument}' must be a whole number from "
            f"{spec['minimum']} to {spec['maximum']}, got {value!r}"
        )
    return max(spec["minimum"], min(int(value), spec["maximum"]))


def normalize_excluded_tools(
    values: list[str] | tuple[str, ...] | set[str] | None,
) -> set[str]:
    """Validate and normalize a repeatable/comma-separated exclusion list."""

    raw_items: list[str] = []
    for value in values or []:
        raw_items.extend(part.strip() for part in str(value).split(","))
    excluded = {item for item in raw_items if item}
    unknown = sorted(excluded - set(TOOL_NAMES))
    if unknown:
        available = ", ".join(TOOL_NAMES)
        raise RuntimeError(
            f"Unknown excluded tool(s): {', '.join(unknown)}. Available tools: {available}"
        )
    required = sorted(excluded & REQUIRED_TOOL_NAMES)
    if required:
        raise RuntimeError(f"Cannot exclude required tool(s): {', '.join(required)}")
    return excluded


# Task rows may set these to override the runner's budgets and harness messages.
TASK_BUDGET_FIELDS = ("max_turns", "max_tool_calls", "max_model_len", "max_tokens")
# A budget of "inf" is not stated to the model; JSON has no infinity literal.
UNLIMITED_BUDGET = "inf"
TASK_FLAG_FIELDS = ("budget_feedback",)
TASK_OVERRIDE_FIELDS = (*TASK_BUDGET_FIELDS, *TASK_FLAG_FIELDS)


def task_budget_overrides(task: dict[str, Any]) -> dict[str, int | bool | str]:
    """Validate a task row's budget overrides, leaving out missing or null fields.

    Budgets must be positive integers or ``"inf"``, and the ``TASK_FLAG_FIELDS`` true or false.
    """

    overrides: dict[str, int | bool | str] = {}
    for name in TASK_OVERRIDE_FIELDS:
        value = task.get(name)
        if value is None:
            continue
        if name in TASK_FLAG_FIELDS:
            if type(value) is not bool:
                raise ValueError(f"Task {name} must be true or false, not {value!r}")
        elif value != UNLIMITED_BUDGET and (type(value) is not int or value <= 0):
            raise ValueError(
                f"Task budget {name} must be a positive integer or 'inf', not {value!r}"
            )
        overrides[name] = value
    return overrides


def task_excluded_tools(
    task: dict[str, Any], excluded_tools: set[str] | frozenset[str]
) -> set[str]:
    """Return the run's exclusions plus those the task row lists in ``excluded_tools``."""

    return normalize_excluded_tools(task.get("excluded_tools") or []) | set(excluded_tools)


def filter_chat_tool_schemas(excluded_tools: set[str]) -> list[dict[str, Any]]:
    """Return independent Chat schemas for the requested tool subset."""

    return deepcopy(
        [tool for tool in TOOL_SCHEMAS if tool["function"]["name"] not in excluded_tools]
    )


def filter_responses_tool_schemas(excluded_tools: set[str]) -> list[dict[str, Any]]:
    """Return independent Responses schemas for the requested tool subset."""

    return deepcopy([tool for tool in RESPONSES_TOOL_SCHEMAS if tool["name"] not in excluded_tools])


def filter_messages_tool_schemas(excluded_tools: set[str]) -> list[dict[str, Any]]:
    """Return independent Anthropic Messages schemas for the requested tool subset."""

    return deepcopy([tool for tool in MESSAGES_TOOL_SCHEMAS if tool["name"] not in excluded_tools])


def normalize_dispatch_arguments(
    name: str,
    arguments: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Apply the canonical defaults and ignore non-contract fields consistently."""

    allowed = TOOL_ALLOWED_ARGUMENTS.get(name)
    if allowed is None:
        return dict(arguments), {}, {}
    ignored = {key: value for key, value in arguments.items() if key not in allowed}
    normalized = {key: value for key, value in arguments.items() if key in allowed}
    applied: dict[str, Any] = {}
    for key, value in TOOL_ARGUMENT_DEFAULTS.get(name, {}).items():
        if key not in normalized:
            normalized[key] = deepcopy(value)
            applied[key] = deepcopy(value)
    return normalized, applied, ignored


def tool_schema_name(schema: dict[str, Any]) -> str | None:
    """Extract a tool name from Chat, Responses, or MCP schema shapes."""

    function = schema.get("function")
    if isinstance(function, dict) and function.get("name"):
        return str(function["name"])
    if schema.get("name"):
        return str(schema["name"])
    return None


__all__ = [
    "MESSAGES_TOOL_SCHEMAS",
    "REQUIRED_TOOL_NAMES",
    "RESPONSES_TOOL_SCHEMAS",
    "TOOL_NAMES",
    "TOOL_ARGUMENT_SPECS",
    "TOOL_SCHEMAS",
    "clamp_to_contract",
    "filter_chat_tool_schemas",
    "filter_messages_tool_schemas",
    "filter_responses_tool_schemas",
    "normalize_excluded_tools",
    "normalize_dispatch_arguments",
    "tool_schema_name",
]
