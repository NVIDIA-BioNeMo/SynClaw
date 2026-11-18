# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLM-facing synthesis tool implementations."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path
from typing import Any

from synclaw.core import chem, db, fingerprint_index, reactions
from synclaw.core.libraries import LibrarySelection, normalize_libraries
from synclaw.core.toxicity import BLOCK_KEY, ToxicityConfig, ToxicityScreen, refusal_message
from synclaw.tools.contracts import clamp_to_contract

REVERSE_RANK_LIMIT = 25
REVERSE_RANK_MAX_OUTCOMES = 10
HIDDEN_TARGET_EVENT = "target_hidden_from_stock"


class SynthesisTools:
    """Tool handlers shared by the MCP server, CLI, and tests."""

    def __init__(
        self,
        db_path: str | Path = db.DEFAULT_DB_PATH,
        *,
        reaction_libraries: LibrarySelection = None,
        toxicity: ToxicityConfig | None = None,
        hidden_target_smiles: str | None = None,
        hidden_building_block_ids: frozenset[int] = frozenset(),
        hidden_similarity: float | None = None,
        _initialize_database: bool = True,
    ):
        """Create tool handlers.

        ``reaction_libraries`` is a hard boundary: when set, template search,
        lookup, simulation, and pathway submission only accept templates in
        those libraries. Leave it empty to allow every imported library.

        ``toxicity`` configures the optional watchlist screen, which is off unless
        the config (or ``SYNCLAW_TOXICITY_SCREEN=on``) enables it. The screen is built on
        first use, so tools that never screen do not read the settings or the watchlist;
        when enabled, a missing watchlist raises ``WatchlistUnavailable`` then. Touch
        ``screen`` to fail early.

        ``hidden_target_smiles`` treats that molecule as out of stock: every building
        block that exact-match scoring would count as it is left out of search,
        grounding and submission. Explicit ``hidden_building_block_ids`` join that set.
        ``hidden_similarity`` also excludes constitutional Morgan/Tanimoto threshold matches.
        The combined set is computed once and cached for the lifetime of this task's tools.
        """
        self.db_path = Path(db_path)
        self.reaction_libraries = normalize_libraries(reaction_libraries)
        self._toxicity = toxicity
        self.hidden_target_smiles = hidden_target_smiles
        self._hidden_ids = frozenset(hidden_building_block_ids)
        if hidden_similarity is not None and (
            not hidden_target_smiles or not 0 <= hidden_similarity <= 1
        ):
            raise RuntimeError("Stock similarity cutoff requires a target and a value in [0, 1].")
        self.hidden_similarity = hidden_similarity
        if _initialize_database:
            db.init_db(self.db_path)

    @cached_property
    def hidden_building_block_ids(self) -> frozenset[int]:
        with db.session(self.db_path) as conn:
            if any(db.get_building_block(conn, bb_id) is None for bb_id in self._hidden_ids):
                raise RuntimeError("Hidden building-block IDs must exist in this database.")
            target_ids = frozenset()
            if self.hidden_target_smiles:
                target_ids = db.building_block_ids_identical_to(conn, self.hidden_target_smiles)
            if self.hidden_similarity is not None:
                db.require_prepared_building_block_search(self.db_path)
                index = fingerprint_index.load_fingerprint_index(conn, self.db_path)
                target_ids |= {
                    bb_id for bb_id, _ in index.top_k(
                        chem.constitutional_morgan_fingerprint(self.hidden_target_smiles),
                        None, min_similarity=self.hidden_similarity,
                    )
                }
            return self._hidden_ids | target_ids

    @cached_property
    def screen(self) -> ToxicityScreen:
        return ToxicityScreen(self._toxicity)

    def search_building_blocks(
        self,
        query: str,
        limit: int = 10,
        include_near: bool = True,
        min_similarity: float = 0.15,
    ) -> dict[str, Any]:
        """Look up a SMILES in stock: the exact match first, then near matches.

        ``*`` browses the catalog. Hidden building blocks never appear.
        """
        limit = clamp_to_contract("search_building_blocks", "limit", limit)
        validation = chem.validate_smiles(query)
        with db.session(self.db_path) as conn:
            results = db.search_building_blocks(
                conn,
                query=query,
                limit=limit,
                include_near=include_near,
                min_similarity=min_similarity,
                hidden_ids=self.hidden_building_block_ids,
            )
        exact_count = sum(1 for item in results if item.get("match_type") == "exact")
        near_count = sum(1 for item in results if item.get("match_type") == "near")
        return {
            "query": query,
            "query_canonical_smiles": validation.get("canonical_smiles") if validation["valid"] else None,
            "limit": limit,
            "include_near": include_near,
            "min_similarity": min_similarity,
            "count": len(results),
            "exact_match_count": exact_count,
            "near_match_count": near_count,
            "submission_guidance": (
                "For submit_pathway, prefer each result's submit_reference such as BB:<id>. "
                "Returned raw/canonical SMILES are also accepted for that exact building block. "
                "Near matches are purchasable alternatives, not proof that the original query SMILES is in the library."
            ),
            "results": results,
            "chem_backend": chem.backend_name(),
        }

    def get_building_block(self, bb_id: int) -> dict[str, Any]:
        with db.session(self.db_path) as conn:
            result = db.get_building_block(
                conn, int(bb_id), hidden_ids=self.hidden_building_block_ids
            )
        if result is None:
            return {"found": False, "bb_id": int(bb_id), "building_block": None}
        return {"found": True, "bb_id": int(bb_id), "building_block": result}

    def similarity(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        return chem.similarity(smiles_a, smiles_b)

    def mces(self, smiles_a: str, smiles_b: str) -> dict[str, Any]:
        return chem.mces(smiles_a, smiles_b)

    def search_reaction_templates(
        self,
        query: str | None = None,
        limit: int = 10,
        library: LibrarySelection = None,
        reactant_smiles: list[str] | None = None,
        product_smiles: str | list[str] | None = None,
        run_forward: bool = False,
        try_permutations: bool = True,
        template_scan_limit: int = 5000,
    ) -> dict[str, Any]:
        """Search templates by text, by reactant/product structure, or both.

        With neither, return an overview of the reaction families. Given both, the two
        searches run independently and return side by side. A configured library
        selection overrides ``library``, so a caller cannot widen it.
        """
        library = self.reaction_libraries or library
        text_query = (query or "").strip()
        has_text_query = bool(text_query and text_query != "*")
        has_molecule_query = bool(reactant_smiles or product_smiles)
        result_limit = clamp_to_contract("search_reaction_templates", "limit", limit)
        if not has_text_query and not has_molecule_query:
            # With nothing to search for, list the reaction families to search by.
            with db.session(self.db_path) as conn:
                overview = db.reaction_template_library_overview(conn, library=library)
            families = overview.get("families") or []
            return {
                "valid": True,
                "issues": [],
                "search_mode": "overview",
                "library": library,
                "template_count": overview["template_count"],
                "family_count": overview["family_count"],
                **(
                    {
                        "families_note": (
                            f"The {len(families)} largest of {overview['family_count']} families; "
                            "search by text to find others."
                        )
                    }
                    if overview["family_count"] > len(families)
                    else {}
                ),
                "families": [
                    {
                        "family": family["family"],
                        "template_count": family["count"],
                        "search_terms": list(family.get("query_hints") or []),
                        **({"description": family["description"]} if family.get("description") else {}),
                    }
                    for family in families
                ],
                "less_common_transformations": [
                    {
                        "name": template.get("name") or template.get("reaction_family") or "unusual transformation",
                        "search_terms": list(template.get("query_hints") or [])[:3],
                    }
                    for template in overview.get("notable_templates") or []
                ],
            }
        with db.session(self.db_path) as conn:
            text_results = (
                reactions.search_reaction_templates(conn, query=text_query, limit=result_limit, library=library)
                if has_text_query
                else []
            )
            if not has_molecule_query:
                return {
                    "valid": True,
                    "issues": [],
                    "query": query,
                    "search_mode": "text",
                    "limit": result_limit,
                    "library": library,
                    "count": len(text_results),
                    "results": text_results,
                }
            # Rank the whole structural match pool before truncating to the requested limit.
            molecule_results = reactions.search_reaction_templates_structural_matches(
                conn,
                reactant_smiles=reactant_smiles,
                product_smiles=product_smiles,
                limit=max(result_limit, min(max(1, int(template_scan_limit)), 50000)),
                library=library,
                run_forward=run_forward,
                try_permutations=try_permutations,
                template_scan_limit=template_scan_limit,
            )
            if product_smiles and molecule_results.get("valid"):
                self._rank_template_search_by_reverse_usefulness(conn, molecule_results, product_smiles)
            else:
                molecule_results["reverse_ranking"] = {"applied": False, "reason": "product_smiles was not supplied"}
        matches = list(molecule_results.pop("results", None) or [])[:result_limit]
        molecule_results.pop("count", None)
        result = {**molecule_results, "query": query, "limit": result_limit}
        if has_text_query:
            # Text and molecule constraints are searched independently and returned side by side.
            return {
                **result,
                "search_mode": "text_and_molecule",
                "structural_results": matches,
                "structural_count": len(matches),
                "text_results": text_results,
                "text_count": len(text_results),
            }
        return {**result, "search_mode": "molecule", "results": matches, "count": len(matches)}

    def get_reaction_template(self, reaction_id: str) -> dict[str, Any]:
        with db.session(self.db_path) as conn:
            template, issue = reactions.lookup_reaction_template(
                conn, reaction_id, library=self.reaction_libraries
            )
        if template is None:
            return {"found": False, "reaction_id": reaction_id, "template": None, "issues": [issue]}
        return {"found": True, "reaction_id": reaction_id, "template": template}

    def simulate_reaction(
        self,
        reaction_id: str,
        reactants: list[str],
        product_smiles: str | None = None,
        max_products: int = 100,
        try_permutations: bool = True,
    ) -> dict[str, Any]:
        """Run a template forward and check an optional claimed product.

        Reactants may be ``BB:<id>`` or any SMILES; unlike in submit_pathway, they need
        not be in stock. A reaction SMARTS counts only if a library template reproduces it.
        """
        with db.session(self.db_path) as conn:
            hidden_ids = self.hidden_building_block_ids
            resolved = [
                _resolve_reactant_ref(conn, ref, hidden_ids=hidden_ids) for ref in reactants
            ]
            issues = [item["issue"] for item in resolved if item.get("issue")]
            if issues:
                return {
                    "valid": False,
                    "reaction_id": reaction_id,
                    "issues": issues,
                    "resolved_reactants": resolved,
                    "product_smiles": product_smiles,
                    "product_matched": False if product_smiles else None,
                    "generated_products": [],
                    "generated_product_sets": [],
                    "generated_product_count": 0,
                    "generated_product_set_count": 0,
                    "execution_backend": "rdkit",
                }
            result = reactions.simulate_reaction(
                conn,
                reaction_id=reaction_id,
                reactant_smiles=[item["smiles"] for item in resolved],
                product_smiles=product_smiles,
                max_products=clamp_to_contract("simulate_reaction", "max_products", max_products),
                try_permutations=try_permutations,
                library=self.reaction_libraries,
            )
            result["resolved_reactants"] = resolved
            return result

    def simulate_reverse_reaction(
        self,
        reaction_id: str,
        product_smiles: str | list[str],
        max_outcomes: int = 50,
        include_near: bool = True,
        min_similarity: float = 0.15,
        top_near_matches: int = 3,
    ) -> dict[str, Any]:
        """Run a template backwards from product(s) to candidate reactant sets.

        Stock matches for the precursors are planning evidence only; an unmatched
        precursor may still be made as an intermediate.
        """
        top_near_matches = clamp_to_contract("simulate_reverse_reaction", "top_near_matches", top_near_matches)
        with db.session(self.db_path) as conn:
            result = reactions.simulate_reverse_reaction(
                conn,
                reaction_id=reaction_id,
                product_smiles=product_smiles,
                max_outcomes=clamp_to_contract("simulate_reverse_reaction", "max_outcomes", max_outcomes),
                library=self.reaction_libraries,
            )
            if result.get("reaction_executed"):
                self._add_reverse_building_block_matches(
                    conn,
                    result,
                    include_near=include_near,
                    min_similarity=min_similarity,
                    top_near_matches=top_near_matches,
                )
        result["include_near"] = include_near
        result["min_similarity"] = min_similarity
        result["top_near_matches"] = top_near_matches
        _compact_reverse_reaction_result(result)
        return result

    def _rank_template_search_by_reverse_usefulness(
        self,
        conn,
        result: dict[str, Any],
        product_smiles: str | list[str],
    ) -> None:
        """Re-rank ``result["results"]`` in place by how well each template's reverse precursors are in stock."""
        matches = list(result.get("results") or [])
        enriched_count = min(len(matches), REVERSE_RANK_LIMIT)
        enriched = matches[:enriched_count]
        remaining = matches[enriched_count:]
        for match in enriched:
            reverse_result = reactions.simulate_reverse_reaction(
                conn,
                reaction_id=str(match.get("template_id") or ""),
                product_smiles=product_smiles,
                max_outcomes=REVERSE_RANK_MAX_OUTCOMES,
                library=self.reaction_libraries,
            )
            if reverse_result.get("reaction_executed"):
                # Ranking only counts exact stock matches.
                self._add_reverse_building_block_matches(conn, reverse_result)
            match["reverse_usefulness"] = _reverse_usefulness_summary(reverse_result)

        enriched.sort(
            key=lambda item: (
                -float(item.get("reverse_usefulness", {}).get("score", 0.0)),
                *reactions.structural_rank_key(item),
            )
        )
        result["results"] = enriched + remaining
        result["count"] = len(result["results"])
        result["reverse_ranking"] = {
            "applied": True,
            "ranking_note": (
                "Product-side structural matches were re-ranked using shallow reverse simulation "
                "and building-block grounding evidence."
            ),
            "enriched_result_count": enriched_count,
        }

    def _add_reverse_building_block_matches(
        self,
        conn,
        result: dict[str, Any],
        *,
        include_near: bool = False,
        # The near-match settings below do nothing unless include_near is True.
        min_similarity: float = 0.0,
        top_near_matches: int = 0,
    ) -> None:
        """Annotate a reverse-simulation ``result`` in place with building-block grounding for each candidate set."""
        exact_match_sets = 0
        exact_match_count = 0
        near_match_count = 0
        match_cache: dict[str, tuple[bool, list[dict[str, Any]]]] = {}
        for candidate in result.get("candidate_reactant_sets") or []:
            reactant_details = []
            for smiles in candidate.get("reactants") or []:
                smiles_text = str(smiles)
                cached = match_cache.get(smiles_text)
                if cached is None:
                    valid_smiles = bool(chem.validate_smiles(smiles_text)["valid"])
                    matches = (
                        db.search_building_blocks(
                            conn,
                            query=smiles_text,
                            limit=10,
                            include_near=True,
                            min_similarity=min_similarity,
                            hidden_ids=self.hidden_building_block_ids,
                        )
                        if include_near and valid_smiles
                        # Only exact hits are used without near matches; skip the catalog scan.
                        else db.exact_building_block_matches(
                            conn, smiles_text, hidden_ids=self.hidden_building_block_ids
                        )
                    )
                    match_cache[smiles_text] = (valid_smiles, matches)
                else:
                    valid_smiles, matches = cached
                exact_matches = [match for match in matches if match.get("match_type") == "exact"]
                near_matches = [match for match in matches if match.get("match_type") == "near"]
                exact_match_count += len(exact_matches)
                near_match_count += len(near_matches[:top_near_matches])
                reactant_details.append(
                    {
                        "smiles": smiles_text,
                        "valid_smiles": valid_smiles,
                        "has_exact_building_block_match": bool(exact_matches),
                        "exact_building_block_matches": [_compact_building_block_match(match) for match in exact_matches],
                        "top_near_building_block_matches": [
                            _compact_building_block_match(match) for match in near_matches[:top_near_matches]
                        ],
                        "grounding_note": (
                            "Exact BB match: can be submitted as a starting reactant."
                            if exact_matches
                            else "No exact BB match: treat as an intermediate unless you find a separate purchasable route."
                        ),
                    }
                )
            candidate["reactant_details"] = reactant_details
            candidate["precursor_grounding"] = [
                {
                    "smiles": detail["smiles"],
                    "has_exact_building_block_match": detail["has_exact_building_block_match"],
                    "exact_submit_references": [
                        match["submit_reference"] for match in detail["exact_building_block_matches"]
                    ],
                    "top_near_submit_references": [
                        match["submit_reference"] for match in detail["top_near_building_block_matches"]
                    ],
                }
                for detail in reactant_details
            ]
            candidate["all_reactants_have_exact_building_block_match"] = bool(reactant_details) and all(
                detail["has_exact_building_block_match"] for detail in reactant_details
            )
            candidate["all_precursors_grounded"] = candidate["all_reactants_have_exact_building_block_match"]
            candidate["grounding_summary"] = (
                "All reverse precursors exactly match purchasable building blocks."
                if candidate["all_precursors_grounded"]
                else "One or more reverse precursors lack exact BB matches; use them only as intermediates."
            )
            if candidate["all_reactants_have_exact_building_block_match"]:
                exact_match_sets += 1
        if exact_match_sets:
            recommended_next_step = (
                "A fully grounded reverse candidate exists. Verify it forward with simulate_reaction before submitting."
            )
        elif result.get("candidate_reactant_sets"):
            recommended_next_step = (
                "No fully grounded reverse candidate was found. Search exact building blocks for useful fragments, "
                "or treat ungrounded reverse precursors as intermediates in a longer route."
            )
        else:
            recommended_next_step = (
                "RDKit did not produce any valid reverse precursor sets for this template/product pair, "
                "though this does not prove the template cannot work forward."
            )
        result["reverse_planning_summary"] = {
            "candidate_sets": len(result.get("candidate_reactant_sets") or []),
            "fully_grounded_candidate_sets": exact_match_sets,
            "has_fully_grounded_candidate": exact_match_sets > 0,
            "returned_top_near_matches_per_precursor": top_near_matches if include_near else 0,
            "recommended_next_step": recommended_next_step,
        }
        result["building_block_match_summary"] = {
            "candidate_sets_with_all_reactants_exactly_matched": exact_match_sets,
            "exact_match_count": exact_match_count,
            "returned_near_match_count": near_match_count,
            "note": (
                "Building-block matches are planning evidence only. Reverse precursors may be "
                "intermediates; only exact matches are submit-ready starting reactants."
            ),
        }

    def submit_pathway(
        self,
        steps: list[dict[str, Any]] | None = None,
        building_block: str | None = None,
        pathway_text: str | None = None,
        target_smiles: str | None = None,
    ) -> dict[str, Any]:
        step_type_issue = None
        if steps is None:
            steps = []
        elif not isinstance(steps, list):
            step_type_issue = "steps must be a JSON array"
            steps = []
        target = ("target", target_smiles)
        block, refusal = self.screen.check_or_refuse("submit_pathway", [target])
        if refusal:
            return {**refusal, "accepted": False}
        target_validation = chem.validate_smiles(target_smiles) if target_smiles else None
        issues = []
        if step_type_issue:
            issues.append(step_type_issue)
        if target_validation is not None and not target_validation["valid"]:
            issues.append(f"target_smiles invalid: {target_validation['error']}")
        if not steps:
            def purchase(extra_issues: list[str]) -> dict[str, Any]:
                return _direct_purchase_submission(
                    self.db_path,
                    building_block=building_block,
                    pathway_text=pathway_text,
                    target_smiles=target_smiles,
                    target_validation=target_validation,
                    issues=[*issues, *extra_issues],
                    hidden_ids=self.hidden_building_block_ids,
                )

            result = purchase([])
            purchased = [("product", smiles) for smiles in result["actual_final_products"]]
            block = self.screen.check([target, *purchased])
            if block and block["refused"]:
                # Rescore with the refusal as an issue so a rejected purchase carries no similarity.
                result = {**purchase([refusal_message(block)]), "refused": True}
            return {**result, BLOCK_KEY: block}
        if building_block:
            issues.append("building_block is only valid for zero-step direct-purchase submissions")

        intermediates: dict[str, str] = {}
        step_results = []
        with db.session(self.db_path) as conn:
            for index, step in enumerate(steps, start=1):
                reaction_id = str(step.get("reaction_id") or step.get("template_id") or "")
                product = str(step.get("product_smiles") or "")
                refs = step.get("reactants") or step.get("reactant_refs") or []
                if not reaction_id:
                    result = {"valid": False, "issues": [f"step {index}: reaction_id is required"]}
                elif not isinstance(refs, list) or not refs:
                    result = {"valid": False, "issues": [f"step {index}: reactants must be a non-empty list"]}
                elif not product:
                    result = {"valid": False, "issues": [f"step {index}: product_smiles is required"]}
                else:
                    resolved = [
                        _resolve_reactant_ref(
                            conn,
                            str(ref),
                            intermediates,
                            require_building_block=True,
                            hidden_ids=self.hidden_building_block_ids,
                        )
                        for ref in refs
                    ]
                    resolve_issues = [item["issue"] for item in resolved if item.get("issue")]
                    if resolve_issues:
                        result = {
                            "valid": False,
                            "reaction_id": reaction_id,
                            "issues": [f"step {index}: {issue}" for issue in resolve_issues],
                            "resolved_reactants": resolved,
                        }
                    else:
                        result = reactions.simulate_reaction(
                            conn,
                            reaction_id=reaction_id,
                            reactant_smiles=[item["smiles"] for item in resolved],
                            product_smiles=product,
                            library=self.reaction_libraries,
                        )
                        result["resolved_reactants"] = resolved
                result["step_index"] = index
                result["submitted_step"] = step
                step_results.append(result)
                if result.get("valid") is False:
                    issues.extend(result.get("issues", []))
                elif result.get("valid") is None:
                    issues.extend(result.get("issues", []))
                actual_step_product = _actual_step_product(result)
                if actual_step_product:
                    result["actual_product_smiles"] = actual_step_product
                    intermediates[f"INT:{index}"] = actual_step_product

        # INT:k counts as used when a step after k lists it. Position alone excludes forward
        # and self references, and a failure in either step does not hide the use.
        used_intermediates = {
            str(ref)
            for position, step in enumerate(steps, start=1)
            if isinstance(refs := step.get("reactants") or step.get("reactant_refs"), list)
            for ref in refs
            if str(ref) in {f"INT:{earlier}" for earlier in range(1, position)}
        }
        issues.extend(
            f"step {index}: INT:{index} is not used by a later step; every step must feed the final product"
            for index in range(1, len(steps))
            if f"INT:{index}" not in used_intermediates
        )

        submitted_final = steps[-1].get("product_smiles") if steps else None
        submitted_final_validation = chem.validate_smiles(submitted_final) if submitted_final else None
        if not submitted_final_validation or not submitted_final_validation["valid"]:
            issues.append(
                "final step product_smiles invalid: "
                f"{submitted_final_validation['error'] if submitted_final_validation else 'missing'}"
            )

        # ReaSyn checks the whole generated product state, not just the chosen
        # outcome. BioNeMo-SynClaw exposes all components as route candidates, so check the
        # existing generated products plus the selected product if output was capped.
        products = [
            ("product" if index == len(step_results) else "intermediate", smiles)
            for index, result in enumerate(step_results, start=1)
            for smiles in dict.fromkeys([
                *(result.get("generated_products") or []), result.get("actual_product_smiles"),
            ])
        ]
        block = self.screen.check([target, *products])
        if block and block["refused"]:
            issues.append(refusal_message(block))

        actual_final_candidates = final_product_candidates(step_results)
        all_steps_valid = bool(step_results) and all(result.get("valid") is True for result in step_results)
        grounding = _summarize_reactant_grounding(step_results)
        accepted = not issues and all_steps_valid
        scoring = score_pathway_against_target(
            actual_final_candidates,
            target_smiles=target_smiles,
            target_validation=target_validation,
            preferred_smiles=(
                submitted_final_validation["canonical_smiles"]
                if submitted_final_validation and submitted_final_validation["valid"]
                else None
            ),
            accepted=accepted,
        )

        error_message = None
        if not accepted:
            error_message = _submission_error_message(
                issues=issues,
                step_results=step_results,
                grounding=grounding,
                actual_final_smiles=scoring["actual_final_smiles"],
            )
        return {
            BLOCK_KEY: block,
            "refused": bool(block and block["refused"]),
            "accepted": accepted,
            "issues": issues,
            "error_message": error_message,
            **scoring,
            "submitted_final_smiles": submitted_final,
            "submitted_final": submitted_final_validation,
            "actual_final_products": actual_final_candidates,
            "pathway_text": pathway_text,
            "steps": step_results,
            "reaction_validation": "valid" if all_steps_valid else "invalid",
            "building_block_grounded": grounding["grounded"],
            "referenced_building_block_ids": grounding["building_block_ids"],
            "ungrounded_reactants": grounding["ungrounded_reactants"],
            "chem_backend": chem.backend_name(),
        }


def hidden_target_event(
    target_smiles: str, hidden_ids: frozenset[int], *, hide_target: bool = True,
    similarity_threshold: float | None = None,
) -> dict[str, Any]:
    """Record stock exclusions, including whether the target was explicitly hidden."""

    return {
        "event_type": HIDDEN_TARGET_EVENT,
        "target_smiles": target_smiles,
        "payload": {
            "hidden_building_block_ids": sorted(hidden_ids),
            "target_hidden_from_stock": hide_target,
            "constitutional_similarity_threshold": similarity_threshold,
        },
    }


def hidden_target_summary_fields(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise target hiding for run summaries and result CSVs.

    ``hidden_building_block_ids`` includes fixed exclusions and target matches, if enabled.
    """

    for event in events:
        if event.get("event_type") == HIDDEN_TARGET_EVENT:
            ids = (event.get("payload") or {}).get("hidden_building_block_ids") or []
            return {
                "target_hidden_from_stock": event["payload"].get("target_hidden_from_stock", True),
                "hidden_building_block_ids": json.dumps(ids),
            }
    return {"target_hidden_from_stock": False, "hidden_building_block_ids": None}


def _direct_purchase_submission(
    db_path: str | Path,
    *,
    building_block: str | None,
    pathway_text: str | None,
    target_smiles: str | None,
    target_validation: dict[str, Any] | None,
    issues: list[Any],
    hidden_ids: frozenset[int],
) -> dict[str, Any]:
    resolved: dict[str, Any] | None = None
    if not building_block:
        issues.append("zero-step submission requires building_block")
    else:
        with db.session(db_path) as conn:
            resolved = _resolve_reactant_ref(
                conn,
                str(building_block),
                intermediates={},
                require_building_block=True,
                hidden_ids=hidden_ids,
            )
        if resolved.get("issue"):
            issues.append(f"building_block: {resolved['issue']}")

    block = resolved.get("building_block") if isinstance(resolved, dict) else None
    final_smiles = str(block["canonical_smiles"]) if isinstance(block, dict) and block.get("canonical_smiles") else None
    final_validation = chem.validate_smiles(final_smiles) if final_smiles else None
    referenced_ids = [int(block["id"])] if isinstance(block, dict) and block.get("id") is not None else []
    accepted = not issues and bool(final_validation and final_validation["valid"]) and bool(referenced_ids)
    scoring = score_pathway_against_target(
        [final_smiles] if final_smiles else [],
        target_smiles=target_smiles,
        target_validation=target_validation,
        preferred_smiles=final_smiles,
        accepted=accepted,
    )
    ungrounded = []
    if building_block and not referenced_ids:
        ungrounded.append(
            {
                "ref": building_block,
                "smiles": resolved.get("smiles") if isinstance(resolved, dict) else None,
                "source": resolved.get("source") if isinstance(resolved, dict) else None,
                "issue": resolved.get("issue") if isinstance(resolved, dict) else "not grounded to a building block",
            }
        )
    grounding = {
        "grounded": bool(referenced_ids) and not ungrounded,
        "building_block_ids": referenced_ids,
        "ungrounded_reactants": ungrounded,
    }
    error_message = None
    if not accepted:
        issue_text = "; ".join(_unique_strings(issues)) if issues else "building_block is not grounded"
        error_message = (
            f"The zero-step direct-purchase submission was rejected: {issue_text}. "
            "Submit steps: [] only with building_block set to BB:<id> or an exact building-block SMILES."
        )
    direct_building_block = None
    if isinstance(block, dict):
        direct_building_block = {
            "id": block.get("id"),
            "submit_reference": f"BB:{block.get('id')}",
            "smiles": block.get("smiles"),
            "canonical_smiles": block.get("canonical_smiles"),
        }
    return {
        "accepted": accepted,
        "issues": issues,
        "error_message": error_message,
        **scoring,
        "submitted_final_smiles": final_smiles,
        "submitted_final": final_validation,
        "actual_final_products": [final_smiles] if final_smiles else [],
        "pathway_text": pathway_text,
        "steps": [],
        "pathway_type": "direct_purchase",
        "direct_building_block": direct_building_block,
        "direct_building_block_ref": building_block,
        "reaction_validation": "valid" if accepted else "invalid",
        "building_block_grounded": grounding["grounded"],
        "referenced_building_block_ids": grounding["building_block_ids"],
        "ungrounded_reactants": grounding["ungrounded_reactants"],
        "chem_backend": chem.backend_name(),
    }


def _actual_step_product(step_result: dict[str, Any]) -> str | None:
    if step_result.get("valid") is not True:
        return None
    product = step_result.get("product")
    if isinstance(product, dict) and product.get("valid") and product.get("canonical_smiles"):
        return str(product["canonical_smiles"])
    generated = _valid_generated_products(step_result.get("generated_products"))
    return generated[0] if generated else None


def final_product_candidates(step_results: list[dict[str, Any]]) -> list[str]:
    """The validated products of a pathway's final step, from which its final product is chosen."""

    if not step_results:
        return []
    final_step = step_results[-1]
    generated = _valid_generated_products(final_step.get("generated_products"))
    if generated:
        return generated
    actual_step_product = _actual_step_product(final_step)
    if actual_step_product:
        return [actual_step_product]
    return []


def _valid_generated_products(products: Any) -> list[str]:
    if not isinstance(products, list):
        return []
    valid_products = []
    for product in products:
        validation = chem.validate_smiles(str(product))
        if validation["valid"]:
            valid_products.append(str(validation["canonical_smiles"]))
    return sorted(set(valid_products))


def score_pathway_against_target(
    candidates: list[str],
    *,
    target_smiles: str | None,
    target_validation: dict[str, Any] | None,
    preferred_smiles: str | None,
    accepted: bool,
) -> dict[str, Any]:
    """Choose the final product of a submission and score it against the target.

    Shared by in-tool scoring, where `submit_pathway` is handed the target at
    dispatch, and post-hoc scoring of a targetless submission, where the tool host
    never sees the target. Only an accepted route is ranked or scored; a rejected
    one keeps its product for diagnosis but carries no metrics.
    """

    final_smiles = select_final_product(
        candidates,
        # Ranking candidates against the target only pays off for a route that
        # will be scored; a rejected one discards the similarity anyway.
        target_smiles=target_smiles if accepted else None,
        preferred_smiles=preferred_smiles,
    )
    final_validation = chem.validate_smiles(final_smiles) if final_smiles else None
    # Callers that validate their own candidates already guarantee these; a host
    # scoring an untrusted submission payload does not.
    # The target and its validation are passed separately so a caller can reuse a
    # validation it already computed; guard on both so the two cannot disagree.
    scorable = (
        accepted
        and target_smiles is not None
        and bool(target_validation and target_validation["valid"])
        and bool(final_validation and final_validation["valid"])
    )
    similarity: dict[str, Any] = chem.similarity(target_smiles, final_smiles) if scorable else {}
    return {
        "target": target_validation,
        "final_smiles": final_smiles,
        "final": final_validation,
        "actual_final_smiles": final_smiles,
        "similarity_to_target": similarity.get("similarity"),
        "identical_to_target": (
            chem.molecules_identical(target_smiles, final_smiles) if scorable else None
        ),
        "constitutional_similarity_to_target": similarity.get("constitutional_similarity"),
        "stereo_aware_similarity_to_target": similarity.get("stereo_aware_similarity"),
    }


def select_final_product(
    candidates: list[str],
    *,
    target_smiles: str | None,
    preferred_smiles: str | None,
) -> str | None:
    """Choose among validated products, prioritizing identity over similarity."""

    if not candidates:
        return None
    if not target_smiles:
        return preferred_smiles if preferred_smiles in candidates else candidates[0]
    # An exact generated product must outrank a fingerprint collision or a
    # high-scoring analogue, including when InChI normalizes a tautomer.
    exact_candidates = [
        product for product in candidates
        if chem.compare_molecular_identity(target_smiles, product)["identical"]
    ]
    if exact_candidates:
        candidates = exact_candidates
    target_validation = chem.validate_smiles(target_smiles)
    if not target_validation["valid"]:
        return candidates[0]
    return max(candidates, key=lambda product: _final_product_rank(product, target_smiles))


def _final_product_rank(product_smiles: str, target_smiles: str) -> tuple[float, float, str]:
    result = chem.similarity(target_smiles, product_smiles)
    stereo = result.get("stereo_aware_similarity")
    constitutional = result.get("constitutional_similarity")
    return (
        float(stereo) if stereo is not None else -1.0,
        float(constitutional) if constitutional is not None else -1.0,
        product_smiles,
    )


def _summarize_reactant_grounding(step_results: list[dict[str, Any]]) -> dict[str, Any]:
    building_block_ids: set[int] = set()
    ungrounded: list[dict[str, Any]] = []
    for step in step_results:
        for reactant in step.get("resolved_reactants") or []:
            source = reactant.get("source")
            if source == "intermediate":
                continue
            block = reactant.get("building_block")
            if block is not None:
                building_block_ids.add(int(block["id"]))
                continue
            ungrounded.append(
                {
                    "step_index": step.get("step_index"),
                    "ref": reactant.get("ref"),
                    "smiles": reactant.get("smiles"),
                    "source": source,
                    "issue": reactant.get("issue") or "reactant is not grounded to a building block",
                }
            )
    return {
        "grounded": not ungrounded,
        "building_block_ids": sorted(building_block_ids),
        "ungrounded_reactants": ungrounded,
    }


def _submission_error_message(
    *,
    issues: list[Any],
    step_results: list[dict[str, Any]],
    grounding: dict[str, Any],
    actual_final_smiles: str | None,
) -> str:
    parts: list[str] = []
    if issues:
        issue_text = "; ".join(_unique_strings(issues))
        parts.append(f"The submitted pathway was rejected: {issue_text}.")
    invalid_steps = [
        str(step.get("step_index"))
        for step in step_results
        if step.get("valid") is not True and step.get("step_index") is not None
    ]
    if invalid_steps:
        parts.append(
            "Repair the invalid reaction step(s): "
            + ", ".join(invalid_steps)
            + ". Each product_smiles must be generated by its reaction_id from the same reactants."
        )
    if not grounding.get("grounded"):
        refs = []
        for item in grounding.get("ungrounded_reactants") or []:
            if isinstance(item, dict):
                ref = item.get("ref") or item.get("smiles") or item.get("issue")
                if ref is not None:
                    refs.append(str(ref))
        refs_text = ", ".join(_unique_strings(refs)) if refs else "one or more reactants"
        parts.append(
            "Every starting reactant in submit_pathway must be a purchasable building block. "
            f"Ungrounded starting reactant(s): {refs_text}. These molecules are not directly "
            "purchasable in the current database. Either synthesize them as earlier intermediates "
            "and reference them as INT:<step_number>, or pivot to a route whose starting reactants "
            "are BB:<id> references or exact building-block SMILES."
        )
    if not actual_final_smiles:
        parts.append(
            "RDKit did not produce a valid final product for the last step, so similarity cannot be scored. "
            "Correct the reaction template/reactant count/reactants and resubmit a route that generates a product."
        )
    if not parts:
        parts.append("The submitted pathway was rejected. Inspect the per-step validation results and resubmit.")
    return " ".join(parts)


def _unique_strings(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for value in values:
        text = str(value)
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def _compact_building_block_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": match.get("id"),
        "submit_reference": match.get("submit_reference"),
        "match_type": match.get("match_type"),
        "smiles": match.get("smiles"),
        "canonical_smiles": match.get("canonical_smiles"),
        "similarity": match.get("similarity"),
        "submission_note": match.get("submission_note"),
    }


def _reverse_usefulness_summary(reverse_result: dict[str, Any]) -> dict[str, Any]:
    candidates = list(reverse_result.get("candidate_reactant_sets") or [])
    best_candidate = _best_reverse_candidate(candidates)
    exact_precursors = 0
    ungrounded_precursors = 0
    exact_refs: list[str] = []
    best_reactants: list[str] = []
    if best_candidate:
        best_reactants = [str(smiles) for smiles in best_candidate.get("reactants") or []]
        for detail in best_candidate.get("precursor_grounding") or []:
            exact_submit_refs = [str(ref) for ref in detail.get("exact_submit_references") or []]
            if exact_submit_refs:
                exact_precursors += 1
                exact_refs.extend(exact_submit_refs)
            else:
                ungrounded_precursors += 1
    planning = reverse_result.get("reverse_planning_summary") or {}
    fully_grounded_sets = int(planning.get("fully_grounded_candidate_sets") or 0)
    candidate_count = int(reverse_result.get("candidate_count") or len(candidates))
    reverse_executed = bool(reverse_result.get("reaction_executed"))
    score = _reverse_usefulness_score(
        reverse_executed=reverse_executed,
        candidate_sets=candidate_count,
        fully_grounded_candidate_sets=fully_grounded_sets,
        exact_precursors=exact_precursors,
        ungrounded_precursors=ungrounded_precursors,
        issues=reverse_result.get("issues") or [],
    )
    return {
        "reverse_executed": reverse_executed,
        "candidate_sets": candidate_count,
        "fully_grounded_candidate_sets": fully_grounded_sets,
        "precursors_with_exact_bb_matches": exact_precursors,
        "best_candidate_reactants": best_reactants,
        "best_candidate_exact_submit_references": exact_refs,
        "recommended_next_step": planning.get("recommended_next_step"),
        "score": score,
        "issues": reverse_result.get("issues") or [],
    }


def _best_reverse_candidate(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return max(candidates, key=_reverse_candidate_score)


def _reverse_candidate_score(candidate: dict[str, Any]) -> tuple[int, int, int]:
    details = candidate.get("precursor_grounding") or []
    exact = sum(1 for detail in details if detail.get("exact_submit_references"))
    return int(bool(candidate.get("all_precursors_grounded"))), exact, exact - len(details)


def _reverse_usefulness_score(
    *,
    reverse_executed: bool,
    candidate_sets: int,
    fully_grounded_candidate_sets: int,
    exact_precursors: int,
    ungrounded_precursors: int,
    issues: list[Any],
) -> float:
    score = 0.0
    if fully_grounded_candidate_sets:
        score += 100.0
    score += 20.0 * exact_precursors
    if candidate_sets:
        score += 10.0
    score -= 5.0 * ungrounded_precursors
    if not reverse_executed or not candidate_sets or issues:
        score -= 20.0
    return score


def _compact_reverse_reaction_result(result: dict[str, Any]) -> None:
    template = result.pop("template", None)
    if isinstance(template, dict):
        result["template_summary"] = {
            "template_id": template.get("template_id"),
            "reaction_family": template.get("reaction_family"),
            "description": template.get("description"),
            "reactant_roles": template.get("reactant_roles"),
            "query_hints": template.get("query_hints"),
            "introduces_unmapped_product_atoms": template.get("introduces_unmapped_product_atoms"),
        }
    products = result.pop("products", None)
    if isinstance(products, list):
        product_errors = [product.get("error") for product in products if isinstance(product, dict) and product.get("error")]
        canonical = [
            product.get("canonical_smiles")
            for product in products
            if isinstance(product, dict) and product.get("canonical_smiles")
        ]
        result["product_valid"] = bool(products) and not product_errors and len(canonical) == len(products)
        result["product_canonical_smiles"] = canonical[0] if len(canonical) == 1 else canonical
        if product_errors:
            result["product_error"] = "; ".join(str(error) for error in product_errors)
    product = result.pop("product", None)
    if isinstance(product, dict) and "product_valid" not in result:
        result["product_valid"] = product.get("valid")
        result["product_canonical_smiles"] = product.get("canonical_smiles")
        if product.get("error"):
            result["product_error"] = product.get("error")


def _resolve_reactant_ref(
    conn,
    ref: str,
    intermediates: dict[str, str] | None = None,
    require_building_block: bool = False,
    *,
    hidden_ids: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    text = str(ref)
    if text.startswith("BB:"):
        try:
            bb_id = int(text.split(":", 1)[1])
        except ValueError:
            return {"ref": ref, "issue": f"invalid building block reference: {ref}"}
        block = db.get_building_block(conn, bb_id, hidden_ids=hidden_ids)
        if block is None:
            return {"ref": ref, "issue": f"unknown building block reference: {ref}"}
        return {"ref": ref, "smiles": block["canonical_smiles"], "source": "building_block", "building_block": block}
    if text.startswith("INT:"):
        if intermediates is None or text not in intermediates:
            return {"ref": ref, "issue": f"unknown intermediate reference: {ref}"}
        return {"ref": ref, "smiles": intermediates[text], "source": "intermediate"}
    validation = chem.validate_smiles(text)
    if not validation["valid"]:
        return {"ref": ref, "issue": f"invalid reactant SMILES/reference: {ref}"}
    canonical = str(validation["canonical_smiles"])
    if not require_building_block:
        return {"ref": ref, "smiles": canonical, "source": "smiles"}
    block = db.get_building_block_by_canonical_smiles(conn, canonical, hidden_ids=hidden_ids)
    if block is None:
        return {
            "ref": ref,
            "smiles": canonical,
            "source": "smiles",
            "issue": (
                f"raw reactant SMILES is not an exact building-block match: {canonical}. "
                "Submit BB:<id> or an exact building-block SMILES."
            ),
        }
    return {"ref": ref, "smiles": block["canonical_smiles"], "source": "building_block_smiles", "building_block": block}
