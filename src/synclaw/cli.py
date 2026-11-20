# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Command line interface for BioNeMo-SynClaw workflows."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from synclaw import __version__, paths
from synclaw.core import chem, db, toxicity
from synclaw.core.libraries import DEFAULT_REACTION_LIBRARIES
from synclaw.core.reactions import ingest_reaction_libraries, ingest_reaction_templates
from synclaw.evaluation.evaluate import evaluate_trace_file, write_summary_csv
from synclaw.evaluation.results import combine_summary_csvs
from synclaw.rollout import policy
from synclaw.runners import model_runner
from synclaw.tasks.datasets import generate_llm_taskset, parse_budget_set
from synclaw.tasks.emolecules import prepare_emolecules_building_blocks
from synclaw.tasks.matrix import (
    ATOM_MAP_POLICIES,
    DEFAULT_CHUNK_SIZE,
    MULTICOMPONENT_POLICIES,
    build_matrix_directory,
)
from synclaw.tasks.public_data import (
    PUBLIC_DATA_FILES,
    download_public_data,
    list_public_data_files,
)
from synclaw.tasks.targets import DEFAULT_POOL_CACHE_MAX, GenerationOptions, generate_targets
from synclaw.tools.contracts import TOOL_ARGUMENT_DEFAULTS
from synclaw.tools.synthesis import SynthesisTools


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    try:
        result = args.func(args)
    except RuntimeError as exc:
        parser.exit(1, f"error: {exc}\n")
    if result is not None:
        print_json(result)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synclaw")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    db_parser = sub.add_parser("db", help="Database operations")
    db_sub = db_parser.add_subparsers(dest="db_command")
    db_init = db_sub.add_parser("init", help="Initialize the SQLite database")
    db_init.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    db_init.set_defaults(func=_cmd_db_init)
    db_stats = db_sub.add_parser("stats", help="Show table counts")
    db_stats.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    db_stats.set_defaults(func=_cmd_db_stats)
    db_remove_reactions = db_sub.add_parser(
        "remove-reactions",
        help="Delete reaction-template libraries and their import records",
    )
    db_remove_reactions.add_argument("library", nargs="+")
    db_remove_reactions.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    db_remove_reactions.set_defaults(func=_cmd_db_remove_reactions)
    db_prepare_search = db_sub.add_parser(
        "prepare-search",
        help="Build the required indexed building-block search database",
    )
    db_prepare_search.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    db_prepare_search.add_argument(
        "--output",
        help="Atomically create a lean indexed chemistry DB at this path",
    )
    db_prepare_search.set_defaults(func=_cmd_db_prepare_search)

    ingest = sub.add_parser("ingest-bbs", help="Import building-block SMILES into SQLite")
    ingest.add_argument("source")
    ingest.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    ingest.set_defaults(func=_cmd_ingest_bbs)

    ingest_reactions = sub.add_parser("ingest-reactions", help="Import reaction templates")
    ingest_reactions.add_argument(
        "source", nargs="?", help="Custom template file; omit to import bundled source libraries"
    )
    ingest_reactions.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    ingest_reactions.add_argument("--library", help="Library name for a custom template file")
    ingest_reactions.add_argument(
        "--annotations",
        help="Optional JSONL file with curated name/description/search metadata for template IDs",
    )
    ingest_reactions.set_defaults(func=_cmd_ingest_reactions)

    taskset = sub.add_parser(
        "make-taskset", help="Convert a SMILES test set into LLM-friendly JSONL tasks"
    )
    taskset.add_argument("source")
    taskset.add_argument("--output", required=True)
    taskset.add_argument("--dataset", required=True)
    taskset.add_argument("--limit", type=int)
    taskset.add_argument(
        "--tool-exclusion-set",
        action="append",
        help=(
            "Comma-separated tools to hide from the model, or 'none'. Repeat to give each task one "
            "set drawn uniformly at random; default: every task has every tool."
        ),
    )
    taskset.add_argument("--tool-exclusion-seed", type=int, default=0)
    taskset.add_argument(
        "--budget-set",
        action="append",
        type=_budget_set,
        help=(
            "Comma-separated budget overrides such as 'max_turns=40,max_tool_calls=75', or 'none' "
            "for the runner's budgets. Fields: max_turns, max_tool_calls, max_model_len, max_tokens "
            "(each a positive integer, or inf to leave it unstated and, except the context, unlimited) "
            "and true|false for budget_feedback. "
            "Repeat to give each task one set drawn uniformly "
            "at random; default: every task uses the runner's budgets."
        ),
    )
    taskset.add_argument("--budget-seed", type=int, default=0)
    taskset.set_defaults(func=_cmd_make_taskset)

    targets = sub.add_parser(
        "generate-targets",
        help="Generate synthesis targets by forward reaction simulation over a reactant-template matrix directory",
    )
    targets.add_argument(
        "--matrix-dir",
        required=True,
        help="Matrix matrix directory; see 'Sourcing a reactant-template matrix directory' in the README",
    )
    targets.add_argument("--output", required=True, help="Destination TSV")
    targets.add_argument("--count", type=int, default=100_000, help="Unique targets to generate")
    targets.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    targets.add_argument("--seed", type=int, default=0)
    targets.add_argument("--min-steps", type=int, default=1)
    targets.add_argument("--max-steps", type=int, default=5)
    targets.add_argument("--max-num-atoms", type=int, default=80)
    targets.add_argument(
        "--sampling",
        choices=["template", "block"],
        default="template",
        help="Uniform over templates, or seeded from a uniformly random building block",
    )
    targets.add_argument("--libraries", help="Restrict to these source libraries, comma-separated")
    targets.add_argument(
        "--no-pathway",
        action="store_true",
        help="Emit targets without a route column; required if the file is fed to make-taskset",
    )
    targets.add_argument(
        "--with-intermediates",
        action="store_true",
        help="Emit each step's product inline, making the route replayable step by step",
    )
    targets.add_argument(
        "--allow-unscoreable",
        action="store_true",
        help="Keep targets whose standard InChI cannot be computed, which make-taskset rejects "
        "as impossible to match exactly (dative-bonded organometallics such as ferrocenes)",
    )
    targets.add_argument("--mw-min", type=float)
    targets.add_argument("--mw-max", type=float)
    targets.add_argument("--sa-min", type=float)
    targets.add_argument("--sa-max", type=float)
    targets.add_argument("--exclude", help="TSV/txt whose first column lists targets to skip")
    targets.add_argument("--pool-cache-max", type=int, default=DEFAULT_POOL_CACHE_MAX)
    targets.add_argument("--max-attempts-per-target", type=int, default=50)
    targets.set_defaults(func=_cmd_generate_targets)

    build_matrix = sub.add_parser(
        "build-matrix",
        help="Build the reactant-template matrix directory that generate-targets reads",
    )
    build_matrix.add_argument(
        "--building-blocks",
        required=True,
        help="SMILES-headed building-block file, optionally gzipped",
    )
    build_matrix.add_argument("--output", required=True, help="Matrix directory to write")
    build_matrix.add_argument(
        "--libraries",
        default=",".join(DEFAULT_REACTION_LIBRARIES),
        help="Template libraries to union",
    )
    build_matrix.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    build_matrix.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Building blocks per task"
    )
    build_matrix.add_argument(
        "--limit", type=int, help="Use only the first LIMIT blocks, for a quick pilot"
    )
    build_matrix.add_argument(
        "--multicomponent",
        choices=list(MULTICOMPONENT_POLICIES),
        default="drop",
        help="Salts and mixtures: drop them, keep the largest component, or keep them verbatim",
    )
    build_matrix.add_argument(
        "--atom-maps",
        choices=list(ATOM_MAP_POLICIES),
        default="drop",
        help="Building blocks whose SMILES carry atom map numbers: drop them, clear the maps, or keep them. "
        "Maps survive into the product, so a route using one yields an atom-mapped target",
    )
    build_matrix.add_argument(
        "--no-common-reagents", action="store_true", help="Skip the common-reagent block file"
    )
    build_matrix.set_defaults(func=_cmd_build_matrix)

    run_model = sub.add_parser(
        "run-model", help="Run a hosted model against one local synthesis task"
    )
    model_runner.add_arguments(run_model)
    run_model.set_defaults(func=_cmd_run_model)

    combine_results = sub.add_parser("combine-results", help="Combine result summary CSVs")
    combine_results.add_argument("inputs", nargs="+", help="CSV paths or glob patterns")
    combine_results.add_argument("--output", required=True)
    combine_results.set_defaults(func=_cmd_combine_results)

    public_data = sub.add_parser(
        "download-public-data", help="Download public reaction/template/example data"
    )
    public_data.add_argument(
        "--name",
        action="append",
        choices=[item.name for item in PUBLIC_DATA_FILES],
        help="Specific public asset to download; repeat to select multiple",
    )
    public_data.add_argument(
        "--dest-root", default=".", help="Repository root or alternate destination root"
    )
    public_data.add_argument("--force", action="store_true", help="Overwrite existing files")
    public_data.add_argument("--timeout", type=float, default=60.0)
    public_data.add_argument(
        "--list", action="store_true", help="List downloadable assets without downloading"
    )
    public_data.set_defaults(func=_cmd_download_public_data)

    emolecules = sub.add_parser(
        "prepare-emolecules-bbs",
        help="Download and normalize an eMolecules building-block release",
    )
    emolecules.add_argument(
        "--release",
        required=True,
        metavar="YYYY-MM-DD",
        help="Dated eMolecules release to download; releases are retired over time, so there is no default",
    )
    emolecules.add_argument(
        "--source-url",
        help="Override the release's parent.smi.gz URL",
    )
    emolecules.add_argument(
        "--dest-root",
        default=".",
        help="Repository root or alternate destination root",
    )
    emolecules.add_argument(
        "--multicomponent",
        choices=list(MULTICOMPONENT_POLICIES),
        default="major",
        help="Salts and mixtures: keep them as shipped, keep the largest component by heavy atoms, or drop them",
    )
    emolecules.add_argument("--force", action="store_true", help="Redownload and reconvert")
    emolecules.add_argument("--timeout", type=float, default=60.0)
    emolecules.set_defaults(func=_cmd_prepare_emolecules_bbs)

    tools = sub.add_parser("tool", help="Call local tool handlers without MCP")
    tools_sub = tools.add_subparsers(dest="tool_command")
    search = tools_sub.add_parser("search-building-blocks", help="Search building blocks")
    search.add_argument("query")
    search.add_argument(
        "--limit", type=int, default=TOOL_ARGUMENT_DEFAULTS["search_building_blocks"]["limit"]
    )
    search.add_argument(
        "--no-near", action="store_true", help="Disable similarity-ranked near matches"
    )
    search.add_argument(
        "--min-similarity", type=float, default=0.15, help="Minimum near-match similarity to return"
    )
    search.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    search.set_defaults(func=_cmd_tool_search)
    get = tools_sub.add_parser("get-bb", help="Fetch one building block")
    get.add_argument("bb_id", type=int)
    get.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    get.set_defaults(func=_cmd_tool_get_bb)
    validate = tools_sub.add_parser("validate", help="Validate SMILES")
    validate.add_argument("smiles")
    validate.set_defaults(func=_cmd_tool_validate)
    sim = tools_sub.add_parser("similarity", help="Compute similarity")
    sim.add_argument("smiles_a")
    sim.add_argument("smiles_b")
    sim.set_defaults(func=_cmd_tool_similarity)
    mces = tools_sub.add_parser("mces", help="Compute a Maximum Common Edge Subgraph")
    mces.add_argument("smiles_a")
    mces.add_argument("smiles_b")
    mces.set_defaults(func=_cmd_tool_mces)
    rxn_search = tools_sub.add_parser("search-reaction-templates", help="Search reaction templates")
    rxn_search.add_argument("query", nargs="?")
    rxn_search.add_argument(
        "--limit", type=int, default=TOOL_ARGUMENT_DEFAULTS["search_reaction_templates"]["limit"]
    )
    rxn_search.add_argument(
        "--library",
        action="append",
        help="Source library to include; repeat or comma-separate for a union",
    )
    rxn_search.add_argument(
        "--reactant-smiles",
        action="append",
        help="Reactant or precursor SMILES to match; repeat for multiple reactants",
    )
    rxn_search.add_argument(
        "--product-smiles",
        action="append",
        help="Product SMILES to match; repeat for multi-product templates",
    )
    rxn_search.add_argument("--run-forward", action="store_true")
    rxn_search.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    rxn_search.set_defaults(func=_cmd_tool_search_reaction_templates)
    rxn_get = tools_sub.add_parser("get-reaction", help="Fetch one reaction template")
    rxn_get.add_argument("reaction_id")
    rxn_get.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    rxn_get.set_defaults(func=_cmd_tool_get_reaction)
    rxn_sim = tools_sub.add_parser(
        "simulate-reaction", help="Run one reaction template and optionally check a claimed product"
    )
    rxn_sim.add_argument("--reaction-id", required=True)
    rxn_sim.add_argument(
        "--reactants-json", required=True, help="JSON list of BB refs, INT refs, or SMILES"
    )
    rxn_sim.add_argument(
        "--product-smiles", help="Optional claimed product to check against generated products"
    )
    rxn_sim.add_argument("--max-products", type=int, default=100)
    rxn_sim.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    rxn_sim.set_defaults(func=_cmd_tool_simulate_reaction)
    rxn_reverse = tools_sub.add_parser(
        "simulate-reverse-reaction",
        help="Run one reaction template backwards from a product to candidate reactants",
    )
    rxn_reverse.add_argument("--reaction-id", required=True)
    rxn_reverse.add_argument(
        "--product-smiles",
        required=True,
        action="append",
        help="Product SMILES; repeat for multi-product templates",
    )
    rxn_reverse.add_argument("--max-outcomes", type=int, default=50)
    rxn_reverse.add_argument(
        "--no-near", action="store_true", help="Disable similarity-ranked near BB matches"
    )
    rxn_reverse.add_argument("--min-similarity", type=float, default=0.15)
    rxn_reverse.add_argument(
        "--top-near-matches", type=int, default=3, help="Near BB matches returned per precursor"
    )
    rxn_reverse.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    rxn_reverse.set_defaults(func=_cmd_tool_simulate_reverse_reaction)
    submit = tools_sub.add_parser("submit-pathway", help="Validate a template-grounded pathway")
    submit.add_argument(
        "--target-smiles", help="Optional target used only for local similarity scoring"
    )
    submit.add_argument(
        "--steps-json", required=True, help="JSON list of structured reaction steps or @path"
    )
    submit.add_argument(
        "--building-block", help="BB:<id> or exact BB SMILES for zero-step direct purchase"
    )
    submit.add_argument("--pathway-text")
    submit.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    toxicity.add_toxicity_argument(submit)
    submit.set_defaults(func=_cmd_tool_submit)

    mcp = sub.add_parser("mcp", help="MCP server operations")
    mcp_sub = mcp.add_subparsers(dest="mcp_command")
    serve = mcp_sub.add_parser("serve", help="Serve MCP tools over stdio")
    serve.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    serve.add_argument(
        "--stage-db-to-node-local",
        action="store_true",
        help="Validate and atomically stage the prepared DB under SLURM_TMPDIR",
    )
    serve.add_argument(
        "--reaction-library",
        nargs="+",
        default=policy.DEFAULT_REACTION_LIBRARY,
        help="Harness-controlled reaction-template library restriction",
    )
    serve.add_argument(
        "--max-concurrent-executions",
        type=int,
        default=64,
        help="Maximum concurrent chemistry tool executions per MCP server",
    )
    toxicity.add_toxicity_argument(serve)
    serve.set_defaults(func=_cmd_mcp_serve)

    eval_parser = sub.add_parser(
        "eval-trace", help="Import and score one or more trace JSONL files"
    )
    eval_parser.add_argument("traces", nargs="+")
    eval_parser.add_argument("--db", default=str(paths.DEFAULT_DB_PATH))
    eval_parser.add_argument("--model", default="unknown")
    eval_parser.add_argument("--prompt-version")
    eval_parser.add_argument("--target-smiles")
    eval_parser.add_argument(
        "--stop-condition",
        type=policy.parse_stop_condition,
        default=policy.DEFAULT_STOP_CONDITION,
        help="The run's stop condition: 'exact' (default), 'model' or a similarity from 0 to 1",
    )
    eval_parser.add_argument(
        "--reaction-library",
        action="append",
        help="Reaction library the run was limited to; repeat or comma-separate for a union",
    )
    eval_parser.add_argument("--output")
    eval_parser.set_defaults(func=_cmd_eval_trace)

    return parser


def print_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _cmd_db_prepare_search(args: argparse.Namespace) -> dict[str, Any]:
    return db.prepare_building_block_search(args.db, output_path=args.output)


def _cmd_db_remove_reactions(args: argparse.Namespace) -> dict[str, Any]:
    with db.session(args.db) as conn:
        removed = db.remove_reaction_libraries(conn, args.library)
    return {"db": args.db, "libraries": args.library, "removed_templates": removed}


def _cmd_db_init(args: argparse.Namespace) -> dict[str, Any]:
    db.init_db(args.db)
    return {"db": args.db, "initialized": True}


def _cmd_db_stats(args: argparse.Namespace) -> dict[str, Any]:
    return {"db": args.db, "counts": db.table_counts(args.db), "chem_backend": chem.backend_name()}


def _cmd_ingest_bbs(args: argparse.Namespace) -> dict[str, Any]:
    return db.ingest_building_blocks(args.source, args.db)


def _cmd_ingest_reactions(args: argparse.Namespace) -> dict[str, Any]:
    if args.source is None:
        if args.library or args.annotations:
            raise RuntimeError(
                "--library and --annotations require a custom template file; omit them to import bundled sources"
            )
        return ingest_reaction_libraries(args.db)
    return ingest_reaction_templates(
        args.source,
        db_path=args.db,
        library=args.library or "default",
        annotations_path=args.annotations,
    )


def _cmd_make_taskset(args: argparse.Namespace) -> dict[str, Any]:
    return generate_llm_taskset(
        source_path=args.source,
        output_path=args.output,
        dataset=args.dataset,
        limit=args.limit,
        tool_exclusion_sets=[
            [] if value.strip() == "none" else [value]
            for value in args.tool_exclusion_set or ["none"]
        ],
        tool_exclusion_seed=args.tool_exclusion_seed,
        budget_sets=args.budget_set or [{}],
        budget_seed=args.budget_seed,
    )


def _budget_set(text: str) -> dict[str, int | bool | str]:
    try:
        return parse_budget_set(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _cmd_build_matrix(args: argparse.Namespace) -> dict[str, Any]:
    return build_matrix_directory(
        building_blocks=args.building_blocks,
        output=args.output,
        libraries=[name for name in args.libraries.split(",") if name.strip()],
        workers=args.workers,
        chunk_size=args.chunk_size,
        limit=args.limit,
        multicomponent=args.multicomponent,
        atom_maps=args.atom_maps,
        common_reagents=not args.no_common_reagents,
    )


def _cmd_generate_targets(args: argparse.Namespace) -> dict[str, Any]:
    libraries = tuple(name.strip() for name in (args.libraries or "").split(",") if name.strip())
    options = GenerationOptions(
        count=args.count,
        workers=args.workers,
        seed=args.seed,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        max_num_atoms=args.max_num_atoms,
        sampling=args.sampling,
        libraries=libraries,
        with_pathway=not args.no_pathway,
        with_intermediates=args.with_intermediates,
        mw_min=args.mw_min,
        mw_max=args.mw_max,
        sa_min=args.sa_min,
        sa_max=args.sa_max,
        require_scoreable=not args.allow_unscoreable,
        exclude=args.exclude,
        pool_cache_max=args.pool_cache_max,
        max_attempts_per_target=args.max_attempts_per_target,
    )
    return generate_targets(args.matrix_dir, args.output, options)


def _cmd_run_model(args: argparse.Namespace) -> dict[str, Any]:
    return model_runner.run_model_from_args(args)


def _cmd_combine_results(args: argparse.Namespace) -> dict[str, Any]:
    return combine_summary_csvs(args.inputs, args.output)


def _cmd_download_public_data(args: argparse.Namespace) -> dict[str, Any]:
    if args.list:
        return {"files": list_public_data_files()}
    return download_public_data(
        dest_root=args.dest_root, names=args.name, force=args.force, timeout=args.timeout
    )


def _cmd_prepare_emolecules_bbs(args: argparse.Namespace) -> dict[str, Any]:
    def report(event: dict[str, Any]) -> None:
        print(json.dumps(event, sort_keys=True), file=sys.stderr, flush=True)

    return prepare_emolecules_building_blocks(
        dest_root=args.dest_root,
        release=args.release,
        source_url=args.source_url,
        multicomponent=args.multicomponent,
        force=args.force,
        timeout=args.timeout,
        progress=report,
    )


def _cmd_tool_search(args: argparse.Namespace) -> dict[str, Any]:
    return SynthesisTools(args.db).search_building_blocks(
        args.query,
        args.limit,
        include_near=not args.no_near,
        min_similarity=args.min_similarity,
    )


def _cmd_tool_get_bb(args: argparse.Namespace) -> dict[str, Any]:
    return SynthesisTools(args.db).get_building_block(args.bb_id)


def _cmd_tool_validate(args: argparse.Namespace) -> dict[str, Any]:
    return chem.validate_smiles(args.smiles)


def _cmd_tool_similarity(args: argparse.Namespace) -> dict[str, Any]:
    return chem.similarity(args.smiles_a, args.smiles_b)


def _cmd_tool_mces(args: argparse.Namespace) -> dict[str, Any]:
    return chem.mces(args.smiles_a, args.smiles_b)


def _cmd_tool_search_reaction_templates(args: argparse.Namespace) -> dict[str, Any]:
    product_smiles = args.product_smiles
    if isinstance(product_smiles, list) and len(product_smiles) == 1:
        product_smiles = product_smiles[0]
    return SynthesisTools(args.db).search_reaction_templates(
        query=args.query,
        reactant_smiles=args.reactant_smiles,
        product_smiles=product_smiles,
        limit=args.limit,
        library=args.library,
        run_forward=args.run_forward,
    )


def _cmd_tool_get_reaction(args: argparse.Namespace) -> dict[str, Any]:
    return SynthesisTools(args.db).get_reaction_template(args.reaction_id)


def _cmd_tool_simulate_reaction(args: argparse.Namespace) -> dict[str, Any]:
    reactants = _load_json_argument(args.reactants_json)
    if not isinstance(reactants, list):
        raise RuntimeError("--reactants-json must decode to a JSON list")
    return SynthesisTools(args.db).simulate_reaction(
        reaction_id=args.reaction_id,
        reactants=[str(item) for item in reactants],
        product_smiles=args.product_smiles,
        max_products=args.max_products,
    )


def _cmd_tool_simulate_reverse_reaction(args: argparse.Namespace) -> dict[str, Any]:
    product_smiles = (
        args.product_smiles[0] if len(args.product_smiles) == 1 else args.product_smiles
    )
    return SynthesisTools(args.db).simulate_reverse_reaction(
        reaction_id=args.reaction_id,
        product_smiles=product_smiles,
        max_outcomes=args.max_outcomes,
        include_near=not args.no_near,
        min_similarity=args.min_similarity,
        top_near_matches=args.top_near_matches,
    )


def _cmd_tool_submit(args: argparse.Namespace) -> dict[str, Any]:
    steps = _load_json_argument(args.steps_json)
    if not isinstance(steps, list):
        raise RuntimeError("--steps-json must decode to a JSON list")
    return SynthesisTools(args.db, toxicity=toxicity.config_from_args(args)).submit_pathway(
        steps=steps,
        building_block=args.building_block,
        pathway_text=args.pathway_text,
        target_smiles=args.target_smiles,
    )


def _cmd_mcp_serve(args: argparse.Namespace) -> None:
    from synclaw.runners.mcp_server import serve

    database_path = (
        db.stage_prepared_search_database(args.db) if args.stage_db_to_node_local else args.db
    )
    serve(
        database_path,
        reaction_library=args.reaction_library,
        max_concurrent_executions=args.max_concurrent_executions,
        toxicity=toxicity.config_from_args(args),
    )
    return None


def _cmd_eval_trace(args: argparse.Namespace) -> dict[str, Any]:
    tool_config = {"stop_condition": args.stop_condition}
    if args.reaction_library:
        tool_config["reaction_library"] = args.reaction_library
    summaries = [
        evaluate_trace_file(
            Path(trace),
            db_path=args.db,
            model=args.model,
            prompt_version=args.prompt_version,
            target_smiles=args.target_smiles,
            tool_config=tool_config,
        )
        for trace in args.traces
    ]
    if args.output:
        write_summary_csv(summaries, args.output)
    return {"summaries": summaries, "output": args.output}


def _load_json_argument(value: str) -> Any:
    if value.startswith("@"):
        return json.loads(Path(value[1:]).read_text(encoding="utf-8"))
    return json.loads(value)
