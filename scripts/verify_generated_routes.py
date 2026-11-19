#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay generated routes through the reaction engine and report what reproduces.

Reads a TSV written by `synclaw generate-targets --with-intermediates`, where each reaction token
is followed by the product recorded for that step, and re-runs every step through
`_run_reaction_with_permutations` -- the function behind `simulate_reaction`. A step
passes when the recorded product is among the ones the engine generates.

Routes produced by `tasks.targets` are executed by that same engine, so they reproduce by
construction; a non-zero failure count means the file came from somewhere else or no longer
matches the matrix directory it names. `--diagnose` retries each failure over every reactant
order, separating "the engine would find it in another order" (it stops at the first productive
order) from "no order produces it".

Replay alone cannot see a template that changed or disappeared from the installed package, because
it executes the SMARTS stored in the matrix directory -- the old definition replays against itself
perfectly. That is reported separately, under `catalog_drift`: reaction ids the matrix has and the
package does not, ids whose SMARTS no longer agree, and how many checked routes cite one. A route
citing a dropped template replays here and is still unusable, since `search_reaction_templates`
will never return that id to an agent.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
from pathlib import Path
from typing import Any

from synclaw.core.libraries import bundled_reaction_template_dir
from synclaw.core.reactions import _rdkit_reaction_modules
from synclaw.tasks.targets import (
    NO_PRODUCTS_REASON,
    REACTION_FILE,
    REACTION_ID_FILE,
    replay_route,
    route_records_products,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tsv", type=Path, required=True, help="Generated TSV with intermediates")
    parser.add_argument("--matrix-dir", type=Path, required=True, help="MatrixDir the TSV was generated from")
    parser.add_argument("--limit", type=int, help="Check a random sample of this many routes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Classify each failure by retrying over every reactant order",
    )
    return parser.parse_args()


def installed_templates() -> dict[str, str]:
    """Template id -> SMARTS as the installed package has them, which is what an agent can reach."""
    root = bundled_reaction_template_dir()
    out: dict[str, str] = {}
    for path in sorted(root.glob("*/templates.jsonl")):
        with open(path) as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    out.setdefault(record["reaction_id"], record["smarts"].strip())
    return out


def main() -> int:
    args = parse_args()
    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    with open(args.matrix_dir / REACTION_FILE) as handle:
        smarts = [line.strip() for line in handle if line.strip()]
    with open(args.matrix_dir / REACTION_ID_FILE) as handle:
        reaction_ids = [line.strip() for line in handle if line.strip()]
    smarts_by_id = dict(zip(reaction_ids, smarts))

    installed = installed_templates()
    missing_from_catalog = {tid for tid in reaction_ids if tid not in installed}
    changed_in_catalog = {
        tid for tid in reaction_ids if tid in installed and installed[tid] != smarts_by_id[tid]
    }
    drifted = missing_from_catalog | changed_in_catalog

    compiled: dict[str, Any] = {}

    def reaction_for(reaction_id: str) -> Any:
        if reaction_id not in compiled:
            reaction = AllChem.ReactionFromSmarts(smarts_by_id[reaction_id])
            rdChemReactions.ChemicalReaction.Initialize(reaction)
            compiled[reaction_id] = reaction
        return compiled[reaction_id]

    def every_order(reaction: Any, mols: list[Any]) -> set[str]:
        found: set[str] = set()
        for order in itertools.permutations(range(len(mols))):
            for product_tuple in reaction.RunReactants(tuple(mols[i] for i in order)):
                for product in product_tuple:
                    if product is None:
                        continue
                    try:
                        Chem.SanitizeMol(product)
                    except Exception:  # noqa: BLE001 - an unsanitizable product does not count
                        continue
                    found.add(Chem.MolToSmiles(product, canonical=True))
        return found

    rows = [line.rstrip("\n").split("\t") for line in open(args.tsv) if line.strip()]
    if any(len(row) < 2 for row in rows):
        raise SystemExit("every row needs a pathway column; regenerate with a route column")
    # Default output would fail every multi-step route for want of recorded products, which reads
    # as broken chemistry rather than the wrong file.
    if not all(route_records_products(row[1], smarts_by_id) for row in rows):
        raise SystemExit(f"{args.tsv}: {NO_PRODUCTS_REASON}")
    if args.limit is not None and args.limit < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.limit)

    routes = steps = steps_ok = routes_ok = routes_citing_drift = 0
    reasons: collections.Counter = collections.Counter()
    for product, pathway, *_ in rows:
        routes += 1
        if drifted.intersection(pathway.split(";")):
            routes_citing_drift += 1
        outcome = replay_route(pathway, product, smarts_by_id, compiled=compiled)
        steps += outcome["steps"]
        steps_ok += outcome["steps_reproduced"]
        if outcome["ok"]:
            routes_ok += 1
        diagnosed = 0
        if args.diagnose:
            for failure in outcome["failures"]:
                mols = [Chem.MolFromSmiles(s) for s in failure["reactants"]]
                if all(m is not None for m in mols) and failure["expected"] in every_order(
                    reaction_for(failure["reaction_id"]), mols
                ):
                    reasons["only from a later reactant order"] += 1
                    diagnosed += 1
        for reason in outcome["reasons"]:
            if reason == "not reproduced" and diagnosed:
                diagnosed -= 1  # already counted under the permutation bucket
                continue
            reasons[reason] += 1

    print(
        json.dumps(
            {
                "routes": routes,
                "routes_fully_reproduced": routes_ok,
                "route_rate": round(routes_ok / routes, 4) if routes else None,
                "reaction_steps": steps,
                "steps_reproduced": steps_ok,
                "step_rate": round(steps_ok / steps, 4) if steps else None,
                "failures": dict(reasons),
                "catalog_drift": {
                    "installed_templates": len(installed),
                    "matrix_templates": len(reaction_ids),
                    "missing_from_catalog": sorted(missing_from_catalog),
                    "changed_in_catalog": sorted(changed_in_catalog),
                    "routes_citing_drift": routes_citing_drift,
                    "route_drift_rate": round(routes_citing_drift / routes, 4) if routes else None,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
