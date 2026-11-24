# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic reaction-template library and deterministic validation hooks."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import re
import sqlite3
import tempfile
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from synclaw import __version__
from synclaw.core import chem, db
from synclaw.core.libraries import (
    BUNDLED_REACTION_SOURCE,
    bundled_reaction_catalog,
    bundled_reaction_library_versions,
    bundled_reaction_template_dir,
)

SMARTS_ARROW = ">>"
# Every reactant order is tried, so a model-written SMARTS is held to the largest library arity.
MAX_SMARTS_REACTANTS = 4
TEMPLATE_KEYS = ("smarts", "reaction_smarts", "template", "template_smarts")
ID_KEYS = ("reaction_id",)
NAME_KEYS = ("name", "title", "reaction_name")
DESCRIPTION_KEYS = ("description", "desc", "natural_language", "notes")
TAG_KEYS = ("tags", "tag", "class", "reaction_class")
ANNOTATION_METADATA_KEYS = (
    "family",
    "reaction_family",
    "family_metadata",
    "search_terms",
    "search_exclusions",
    "reactant_roles",
    "retro_planning_notes",
    "execution_caveats",
    "overview",
    "replace_tags",
)


# Where an installed package, which ships no template files, fetches them from.
CATALOG_BASE_URL = "https://raw.githubusercontent.com/NVIDIA-BioNeMo/SynClaw"


def ingest_reaction_libraries(
    db_path: str | Path, *, catalog_ref: str | None = None
) -> dict[str, Any]:
    """Install every bundled source and renumber the database's templates R001 onward.

    Existing rows stored under bundled library names are replaced in the same
    transaction, so a reimport never leaves stale templates behind. Drop sources
    a run should not use with ``db.remove_reaction_libraries``.

    A source checkout reads its own template files. An installed package, or a call with
    ``catalog_ref`` (a tag, branch or commit), downloads them from the repository; the release
    tag for this version is the default. Every file is checked against the SHA-256 in the
    manifest, so a ref that holds different templates is refused, not installed.

    Raises:
        RuntimeError: A file could not be downloaded or does not match the manifest.
    """
    catalog = bundled_reaction_catalog()
    template_dir = bundled_reaction_template_dir()
    if catalog_ref is None and template_dir.is_dir():
        return _install_catalog(db_path, catalog, template_dir)
    with tempfile.TemporaryDirectory() as tmp:
        _download_catalog(catalog, Path(tmp), catalog_ref or f"v{__version__}")
        return _install_catalog(db_path, catalog, Path(tmp))


def _download_catalog(catalog: dict[str, Any], destination: Path, ref: str) -> None:
    for entry in (catalog["families"], *catalog["sources"].values()):
        url = f"{CATALOG_BASE_URL}/{ref}/data/reaction_templates/{entry['file']}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data = response.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RuntimeError(
                f"Could not download {url}: {exc}. Check the network, or pass --catalog-ref."
            ) from exc
        if hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise RuntimeError(
                f"{url} does not match its SHA-256 in reaction_libraries.json. Pass --catalog-ref "
                "with the tag or commit this package was built from."
            )
        path = destination / entry["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _install_catalog(
    db_path: str | Path, catalog: dict[str, Any], template_dir: Path
) -> dict[str, Any]:
    libraries = list(catalog["sources"])
    families = json.loads(_verified_bundled_file(template_dir, catalog["families"]).read_text())
    records = []
    for library, source in catalog["sources"].items():
        for row in db.iter_jsonl(_verified_bundled_file(template_dir, source)):
            if row["family"] not in families:
                raise ValueError(
                    f"{row['reaction_id']}: family {row['family']!r} is missing from families.json"
                )
            records.append((library, {**row, "family_metadata": families[row["family"]]}))
    if len({row["smarts"] for _, row in records}) != len(records):
        raise ValueError("Bundled reaction sources contain duplicated SMARTS")
    unparseable = [
        row["reaction_id"] for _, row in records if not _is_reaction_smarts_template(row["smarts"])
    ]
    if unparseable:
        raise ValueError(f"Bundled templates with SMARTS RDKit cannot run: {unparseable}")
    db.init_db(db_path)
    stats = {"libraries": libraries, "records": len(records), "inserted": 0, "replaced": 0}
    versions = bundled_reaction_library_versions()
    with db.session(db_path) as conn:
        stats["replaced"] = db.remove_reaction_libraries(conn, libraries)
        for index, (library, row) in enumerate(records):
            normalized = normalize_template_record(row, index=index, library=library)
            db.upsert_reaction_template(conn, **normalized)
            stats["inserted"] += 1
        db.number_reaction_templates(conn)
        for library, source in catalog["sources"].items():
            db.replace_reaction_library(
                conn,
                library,
                source=BUNDLED_REACTION_SOURCE,
                catalog_sha256=versions[library],
                template_count=int(source["templates"]),
            )
    return stats


def _verified_bundled_file(template_dir: Path, entry: dict[str, Any]) -> Path:
    path = template_dir / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError(f"{path} does not match its SHA-256 in reaction_libraries.json")
    return path


def ingest_reaction_templates(
    source_path: str | Path,
    db_path: str | Path,
    library: str = "default",
    annotations_path: str | Path | None = None,
) -> dict[str, Any]:
    """Import reaction templates from TXT, JSONL, JSON, or CSV."""

    db.init_db(db_path)
    source = Path(source_path)
    annotations = (
        load_template_annotations(annotations_path, library=library) if annotations_path else {}
    )
    stats: dict[str, Any] = {
        "source_file": str(source),
        "library": library,
        "annotations_file": str(annotations_path) if annotations_path else None,
        "records": 0,
        "inserted": 0,
        "updated": 0,
        "annotated": 0,
        "invalid": 0,
        "replaced": 0,
    }
    records = list(read_template_records(source))
    start_index = 0 if source.suffix.lower() not in (".jsonl", ".json", ".csv") else 1
    digest = hashlib.sha256(source.read_bytes())
    if annotations_path:
        digest.update(Path(annotations_path).read_bytes())
    with db.session(db_path) as conn:
        # Reimporting a library replaces it, so templates dropped from the
        # source file do not linger in searches or the family overview.
        stats["replaced"] = db.remove_reaction_libraries(conn, library)
        for index, record in enumerate(records, start=start_index):
            stats["records"] += 1
            normalized = normalize_template_record(record, index=index, library=library)
            if not _is_reaction_smarts_template(normalized["smarts"]):
                stats["invalid"] += 1
                continue
            annotation = annotations.get((library, normalized["reaction_id"]))
            if annotation is not None:
                normalized = apply_template_annotation(
                    normalized,
                    annotation,
                    replace_existing_tags=True,
                )
                stats["annotated"] += 1
            stored = db.get_reaction_template_by_smarts(conn, normalized["smarts"])
            if stored is not None:
                raise ValueError(
                    f"{normalized['reaction_id']}: its SMARTS is already stored as {stored['reaction_id']}; "
                    "store each SMARTS in one library only"
                )
            inserted = db.upsert_reaction_template(conn, **normalized)
            if inserted:
                stats["inserted"] += 1
            else:
                stats["updated"] += 1
        # Raising inside the session rolls the whole import back, so an import
        # either applies its annotations or leaves the database untouched.
        if annotations_path and not stats["annotated"] and stats["inserted"] + stats["updated"]:
            raise ValueError(
                f"No template in {source} matched an annotation in {annotations_path} "
                f"for library {library!r}. Check that the annotations record the same "
                "library and template IDs, or omit --annotations."
            )
        db.replace_reaction_library(
            conn,
            library,
            source=str(source.resolve()),
            catalog_sha256=digest.hexdigest(),
            template_count=stats["inserted"] + stats["updated"],
        )
        db.number_reaction_templates(conn)
    return stats


def load_template_annotations(
    annotations_path: str | Path,
    *,
    library: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load explicit model-facing metadata overlays for reaction templates."""

    annotations: dict[tuple[str, str], dict[str, Any]] = {}
    family_metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for line_number, record in enumerate(db.iter_jsonl(annotations_path), start=1):
        reaction_id = _first_value(record, ID_KEYS)
        if reaction_id in (None, ""):
            raise ValueError(f"{annotations_path}:{line_number}: annotation requires reaction_id")
        annotation_library = str(record.get("library") or library)
        normalized_id = _normalize_reaction_id(reaction_id, index=0, library=annotation_library)
        key = (annotation_library, normalized_id)
        if key in annotations:
            raise ValueError(
                f"{annotations_path}:{line_number}: duplicate annotation for {normalized_id}"
            )
        _validate_template_annotation(record, source=f"{annotations_path}:{line_number}")
        family = str(record.get("family") or record.get("reaction_family") or "")
        metadata = record.get("family_metadata")
        if family and isinstance(metadata, dict):
            family_key = (annotation_library, family)
            previous = family_metadata.get(family_key)
            if previous is not None and previous != metadata:
                raise ValueError(
                    f"{annotations_path}:{line_number}: inconsistent family_metadata for {family}"
                )
            family_metadata[family_key] = metadata
        annotations[key] = record
    return annotations


def _validate_template_annotation(annotation: dict[str, Any], *, source: str) -> None:
    replace_tags = annotation.get("replace_tags")
    if replace_tags is not None and not isinstance(replace_tags, bool):
        raise ValueError(f"{source}: replace_tags must be a boolean")
    for key in ("search_terms", "search_exclusions"):
        if key in annotation:
            _validate_text_list(annotation[key], source=source, field=key)
    for key in ("retro_planning_notes", "execution_caveats"):
        if key in annotation:
            value = annotation[key]
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError(f"{source}: {key} must not be empty")
            else:
                _validate_text_list(value, source=source, field=key)

    family_metadata = annotation.get("family_metadata")
    if family_metadata is not None:
        if not isinstance(family_metadata, dict):
            raise ValueError(f"{source}: family_metadata must be an object")
        description = family_metadata.get("description")
        if description is not None and not isinstance(description, str):
            raise ValueError(f"{source}: family_metadata.description must be a string")
        if "search_terms" in family_metadata:
            _validate_text_list(
                family_metadata["search_terms"],
                source=source,
                field="family_metadata.search_terms",
            )

    reactant_roles = annotation.get("reactant_roles")
    if reactant_roles is not None:
        if not isinstance(reactant_roles, list):
            raise ValueError(f"{source}: reactant_roles must be a list")
        positions: set[int] = set()
        for role in reactant_roles:
            if not isinstance(role, dict):
                raise ValueError(f"{source}: each reactant role must be an object")
            position = role.get("position")
            if isinstance(position, bool) or not isinstance(position, int) or position < 1:
                raise ValueError(f"{source}: reactant role position must be a positive integer")
            if position in positions:
                raise ValueError(f"{source}: duplicate reactant role position {position}")
            positions.add(position)
            if not isinstance(role.get("role"), str) or not str(role["role"]).strip():
                raise ValueError(f"{source}: reactant role text must be a non-empty string")

    overview = annotation.get("overview")
    if overview is not None:
        if not isinstance(overview, dict):
            raise ValueError(f"{source}: overview must be an object")
        priority = overview.get("priority", 0)
        if isinstance(priority, bool) or not isinstance(priority, int) or priority < 0:
            raise ValueError(f"{source}: overview.priority must be a non-negative integer")
        category = overview.get("category")
        if priority > 0 and (not isinstance(category, str) or not category.strip()):
            raise ValueError(f"{source}: prioritized overview entries require a category")


def _validate_text_list(value: Any, *, source: str, field: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{source}: {field} must be a list of non-empty strings")


def apply_template_annotation(
    normalized: dict[str, Any],
    annotation: dict[str, Any],
    *,
    replace_existing_tags: bool,
) -> dict[str, Any]:
    """Apply curated model-facing metadata to a normalized template record."""

    value = dict(normalized)
    reactant_roles = annotation.get("reactant_roles")
    if reactant_roles:
        reactant_count, _ = db.reaction_template_arity(str(value.get("smarts") or ""))
        positions = sorted(int(role["position"]) for role in reactant_roles)
        expected_positions = list(range(1, reactant_count + 1))
        if positions != expected_positions:
            raise ValueError(
                f"{value.get('reaction_id')}: reactant role positions {positions} "
                f"do not match template positions {expected_positions}"
            )
    name = annotation.get("name") or annotation.get("title")
    if name not in (None, ""):
        value["name"] = str(name)
    description = _first_value(annotation, DESCRIPTION_KEYS)
    if description not in (None, ""):
        value["description"] = str(description)

    explicit_replace_tags = annotation.get("replace_tags")
    replace_tags = (
        explicit_replace_tags if isinstance(explicit_replace_tags, bool) else replace_existing_tags
    )
    tags = [] if replace_tags else _normalize_tags(_json_list(value.get("tags_json")))
    family = annotation.get("family") or annotation.get("reaction_family")
    if family not in (None, ""):
        tags.append(str(family))
    tags.extend(_normalize_tags(annotation.get("tags")))
    tags.extend(_normalize_tags(annotation.get("search_terms")))
    value["tags_json"] = json.dumps(_dedupe_texts(tags), sort_keys=True)

    metadata = _json_object(value.get("metadata_json"))
    metadata["annotation"] = annotation
    value["metadata_json"] = json.dumps(metadata, sort_keys=True)
    return value


def read_template_records(source: Path) -> list[dict[str, Any]]:
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        return list(db.iter_jsonl(source))
    if suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if isinstance(value, list):
            return [record for record in value if isinstance(record, dict)]
        if isinstance(value, dict):
            for key in ("templates", "records", "reactions"):
                if isinstance(value.get(key), list):
                    return [record for record in value[key] if isinstance(record, dict)]
            return [value]
        raise ValueError(f"{source}: JSON root must be an object or list")
    if suffix == ".csv":
        with source.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    records = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            records.append({"smarts": line, "line_number": line_number})
    return records


def _inline_template_annotation(record: dict[str, Any], *, source: str) -> dict[str, Any]:
    annotation: dict[str, Any] = {}
    for key in ANNOTATION_METADATA_KEYS:
        value = record.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        annotation[key] = _decode_inline_annotation_value(key, value, source=source)
    return annotation


def _decode_inline_annotation_value(key: str, value: Any, *, source: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if key == "replace_tags":
        return text.lower() == "true" if text.lower() in {"true", "false"} else text
    if key in {"search_terms", "search_exclusions", "retro_planning_notes", "execution_caveats"}:
        if text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        return _normalize_tags(text) if key in {"search_terms", "search_exclusions"} else text
    if key in {"family_metadata", "reactant_roles", "overview"}:
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source}: {key} must contain valid JSON") from exc
    return text


def normalize_template_record(record: dict[str, Any], index: int, library: str) -> dict[str, Any]:
    smarts = _first_value(record, TEMPLATE_KEYS)
    reaction_id = _normalize_reaction_id(
        _first_value(record, ID_KEYS), index=index, library=library
    )
    name = _first_value(record, NAME_KEYS)
    description = _first_value(record, DESCRIPTION_KEYS)
    tags = _normalize_tags(_first_value(record, TAG_KEYS))
    if not name:
        name = "Unnamed reaction template"
    valid = _is_reaction_smarts_template(str(smarts or ""))
    if not description and valid:
        description = _infer_description(str(smarts))
    # Keep the record's own metadata (e.g. provenance) and nest any other unused fields,
    # so neither can be mistaken for the curated annotation stored below.
    consumed = {
        *TEMPLATE_KEYS,
        *ID_KEYS,
        *NAME_KEYS,
        *DESCRIPTION_KEYS,
        *TAG_KEYS,
        *ANNOTATION_METADATA_KEYS,
        "library",
    }
    source_metadata = record.get("metadata")
    nest_metadata = not isinstance(source_metadata, dict) or "annotation" in source_metadata
    metadata: dict[str, Any] = {} if nest_metadata else dict(source_metadata)
    source_fields = {
        key: value
        for key, value in record.items()
        if key not in consumed and (key != "metadata" or nest_metadata)
    }
    if source_fields:
        metadata["source_fields"] = source_fields
    # Import numbers every template R001 onward; this keeps the ID the source used.
    metadata["source_id"] = str(reaction_id)
    value = {
        "reaction_id": str(reaction_id),
        "library": library,
        "name": str(name),
        "description": str(description),
        "smarts": str(smarts or "").strip(),
        "tags_json": json.dumps(tags, sort_keys=True),
        "metadata_json": json.dumps(metadata, sort_keys=True),
    }
    if not valid:
        return value
    inline_annotation = _inline_template_annotation(record, source=str(reaction_id))
    if inline_annotation:
        _validate_template_annotation(inline_annotation, source=str(reaction_id))
    return (
        apply_template_annotation(
            value,
            inline_annotation,
            replace_existing_tags=False,
        )
        if inline_annotation
        else value
    )


def _is_reaction_smarts_template(smarts: str) -> bool:
    if SMARTS_ARROW not in smarts:
        return False
    try:
        reactants, products = db.reaction_template_arity(smarts)
    except ValueError:
        return False
    return reactants > 0 and products > 0


def _normalize_reaction_id(value: Any, *, index: int, library: str) -> str:
    if value in (None, ""):
        return f"{library}:R{index:03d}"
    text = str(value)
    if text.isdigit():
        return f"{library}:R{int(text):03d}"
    if ":" not in text:
        return f"{library}:{text}"
    return text


def search_reaction_templates_structural_matches(
    conn: sqlite3.Connection,
    reactant_smiles: list[str] | None = None,
    product_smiles: str | list[str] | None = None,
    *,
    run_forward: bool,
) -> dict[str, Any]:
    """Rank every template whose SMARTS roles structurally match the molecules."""

    reactant_inputs = _normalize_smiles_inputs(reactant_smiles)
    product_inputs = _normalize_smiles_inputs(product_smiles)
    if not reactant_inputs and not product_inputs:
        return {
            "valid": False,
            "issues": ["Provide at least one reactant_smiles or product_smiles value."],
            "reactants": [],
            "products": [],
            "count": 0,
            "results": [],
            "run_forward": run_forward,
        }

    reactant_validations = [chem.validate_smiles(smiles) for smiles in reactant_inputs]
    product_validations = [chem.validate_smiles(smiles) for smiles in product_inputs]
    issues = _smiles_validation_issues("reactant", reactant_validations)
    issues.extend(_smiles_validation_issues("product", product_validations))
    if issues:
        return {
            "valid": False,
            "issues": issues,
            "reactants": reactant_validations,
            "products": product_validations,
            "count": 0,
            "results": [],
            "run_forward": run_forward,
        }

    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    reactant_mols = [Chem.MolFromSmiles(v["canonical_smiles"]) for v in reactant_validations]
    product_mols = [Chem.MolFromSmiles(v["canonical_smiles"]) for v in product_validations]
    requested_product_smiles = [str(v["canonical_smiles"]) for v in product_validations]
    templates = db.list_reaction_templates(conn)
    matches: list[dict[str, Any]] = []
    execution_issues: list[str] = []

    for template in templates:
        try:
            reaction = AllChem.ReactionFromSmarts(template["smarts"])
            rdChemReactions.ChemicalReaction.Initialize(reaction)
        except Exception as exc:
            execution_issues.append(
                f"{template.get('reaction_id')}: RDKit could not parse template: {exc}"
            )
            continue

        reactant_templates = [
            reaction.GetReactantTemplate(index)
            for index in range(reaction.GetNumReactantTemplates())
        ]
        product_templates = [
            reaction.GetProductTemplate(index) for index in range(reaction.GetNumProductTemplates())
        ]
        reactant_match = _match_molecules_to_reaction_roles(reactant_mols, reactant_templates)
        product_match = _match_molecules_to_reaction_roles(product_mols, product_templates)
        if not reactant_match["matched"] or not product_match["matched"]:
            continue

        forward_result = None
        forward_summary = None
        if (
            run_forward
            and reactant_inputs
            and len(reactant_inputs) == reaction.GetNumReactantTemplates()
        ):
            forward_result = _simulate_template(
                template,
                str(template["reaction_id"]),
                [str(v["canonical_smiles"]) for v in reactant_validations],
                None,
                max_products=None,
            )
            generated_products = forward_result.get("generated_products") or []
            generated_sets = forward_result.get("generated_product_sets") or []
            requested_set = set(requested_product_smiles)
            forward_summary = {
                "reaction_executed": forward_result.get("reaction_executed"),
                "valid": forward_result.get("valid"),
                "generated_products": generated_products,
                "generated_product_sets": generated_sets,
                "generated_product_count": forward_result.get("generated_product_count", 0),
                "generated_product_set_count": forward_result.get("generated_product_set_count", 0),
                "skipped_invalid_product_set_count": forward_result.get(
                    "skipped_invalid_product_set_count",
                    0,
                ),
                "requested_products_generated": {
                    smiles: smiles in generated_products for smiles in requested_product_smiles
                },
                "requested_product_set_generated": (
                    any(requested_set.issubset(set(product_set)) for product_set in generated_sets)
                    if requested_set
                    else None
                ),
                "issues": forward_result.get("issues") or [],
            }

        score = _structure_template_match_score(
            reactant_inputs=reactant_inputs,
            product_inputs=product_inputs,
            template=template,
            forward_summary=forward_summary,
        )
        match = dict(template)
        match["structure_match"] = {
            "score": score,
            "reactant_match": reactant_match["matched"],
            "product_match": product_match["matched"],
            "reactant_mappings": reactant_match["mappings"],
            "product_mappings": product_match["mappings"],
            "provided_reactant_count": len(reactant_inputs),
            "provided_product_count": len(product_inputs),
            "template_reactant_count": reaction.GetNumReactantTemplates(),
            "template_product_count": reaction.GetNumProductTemplates(),
            "partial_reactant_query": bool(reactant_inputs)
            and len(reactant_inputs) < reaction.GetNumReactantTemplates(),
            "partial_product_query": bool(product_inputs)
            and len(product_inputs) < reaction.GetNumProductTemplates(),
            "forward_simulation": forward_summary,
        }
        matches.append(match)

    matches.sort(key=structural_rank_key)
    result = {
        "valid": True,
        "issues": [],
        "reactants": reactant_validations,
        "products": product_validations,
        "run_forward": run_forward,
        "scanned_template_count": len(templates),
        "matched_template_count": len(matches),
        "skipped_template_issue_count": len(execution_issues),
        "skipped_template_issues": execution_issues,
        "count": len(matches),
        "results": matches,
    }
    return result


def structural_rank_key(match: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -float(match.get("structure_match", {}).get("score", 0.0)),
        match.get("reaction_id", ""),
    )


def lookup_reaction_template(
    conn: sqlite3.Connection, reaction_id: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Resolve a template ID, or explain why it is unavailable."""

    template = db.get_reaction_template(conn, reaction_id)
    if template is not None:
        return template, None
    return None, f"Unknown reaction template: {reaction_id}"


def simulate_reaction(
    conn: sqlite3.Connection,
    reaction_id: str,
    reactant_smiles: list[str],
    product_smiles: str | None = None,
    *,
    max_products: int,
) -> dict[str, Any]:
    """Run a reaction template and optionally check a claimed product.

    ``reaction_id`` may instead be a reaction SMARTS, which is grounded to a library template
    that gives the same result on these reactants.
    """

    if is_reaction_smarts(reaction_id):
        return _simulate_reaction_smarts(
            conn,
            str(reaction_id),
            reactant_smiles,
            product_smiles,
            max_products=max_products,
        )
    template, issue = lookup_reaction_template(conn, reaction_id)
    if template is None:
        return {"valid": False, "reaction_id": reaction_id, "issues": [issue]}
    return _simulate_template(
        template,
        reaction_id,
        reactant_smiles,
        product_smiles,
        max_products=max_products,
    )


def is_reaction_smarts(reaction_id: Any) -> bool:
    return SMARTS_ARROW in str(reaction_id)


def _simulate_reaction_smarts(
    conn: sqlite3.Connection,
    smarts: str,
    reactant_smiles: list[str],
    product_smiles: str | None,
    *,
    max_products: int,
) -> dict[str, Any]:
    """Run the library template that reproduces a reaction SMARTS on these reactants.

    The SMARTS must generate the claimed product, or any product when none is claimed. A
    library template must then generate the claimed product, or every product the SMARTS
    generates. A template with exactly this SMARTS is tried first, then the others in
    structural rank order.
    """

    proposed = (
        _simulate_template(
            {"smarts": smarts},
            smarts,
            reactant_smiles,
            product_smiles,
            max_products=None,
        )
        if len(reactant_smiles) <= MAX_SMARTS_REACTANTS
        else {}
    )
    product = proposed.get("product")
    if not proposed:
        issues = [f"A reaction SMARTS may take at most {MAX_SMARTS_REACTANTS} reactants."]
    elif not proposed.get("reactants_valid", True) or (
        product is not None and not product["valid"]
    ):
        # Bad input SMILES, not the SMARTS, is the problem to report.
        issues = [issue for issue in proposed.get("issues") or [] if " invalid: " in issue]
    elif not proposed.get("reaction_executed"):
        issues = [f"reaction SMARTS: {item}" for item in proposed.get("issues") or []]
    elif not proposed.get("generated_products"):
        issues = ["The reaction SMARTS generated no products from these reactants."]
    elif product is not None and not proposed.get("product_matched"):
        issues = ["The reaction SMARTS does not generate the claimed product from these reactants."]
    else:
        claimed = product["canonical_smiles"] if product is not None else None
        wanted = {claimed} if claimed else set(proposed["generated_products"])
        reactants = [item["canonical_smiles"] for item in proposed["reactants"]]
        for template in _smarts_grounding_candidates(conn, smarts, reactants, claimed):
            result = _simulate_template(
                template,
                str(template["reaction_id"]),
                reactant_smiles,
                product_smiles,
                max_products=None,
            )
            if wanted <= set(result["generated_products"]):
                result["generated_products"] = result["generated_products"][:max_products]
                result["generated_product_sets"] = result["generated_product_sets"][:max_products]
                result["reaction_smarts"] = smarts
                return result
        issues = [
            "No library template generates "
            + ("the claimed product" if claimed else "the same products")
            + " from these reactants, so the reaction SMARTS cannot be used."
        ]
    return {
        "valid": False,
        "reaction_id": smarts,
        "issues": issues,
        "reactants": proposed.get("reactants"),
        "product": product,
        "product_smiles": product_smiles,
        "product_matched": False if product_smiles not in (None, "") else None,
        "generated_products": [],
        "generated_product_sets": [],
        "generated_product_count": 0,
        "generated_product_set_count": 0,
        "reaction_executed": False,
    }


def _smarts_grounding_candidates(
    conn: sqlite3.Connection,
    smarts: str,
    reactant_smiles: list[str],
    product_smiles: str | None,
) -> Iterator[dict[str, Any]]:
    exact = db.get_reaction_template_by_smarts(conn, smarts)
    if exact is not None:
        yield exact
    # Role matching alone is cheap; the caller simulates candidates only until one fits.
    matches = search_reaction_templates_structural_matches(
        conn,
        reactant_smiles=reactant_smiles,
        product_smiles=product_smiles,
        run_forward=False,
    )
    for match in matches.get("results") or []:
        if exact is None or match["reaction_id"] != exact["reaction_id"]:
            yield {key: value for key, value in match.items() if key != "structure_match"}


def _simulate_template(
    template: dict[str, Any],
    reaction_id: str,
    reactant_smiles: list[str],
    product_smiles: str | None,
    *,
    max_products: int | None,
) -> dict[str, Any]:
    product_issues = []
    product_validation = None
    if product_smiles not in (None, ""):
        product_validation = chem.validate_smiles(str(product_smiles))
        if not product_validation["valid"]:
            product_issues.append(f"product invalid: {product_validation['error']}")
    reactant_issues = []
    reactant_validations = [chem.validate_smiles(smiles) for smiles in reactant_smiles]
    for index, validation in enumerate(reactant_validations, start=1):
        if not validation["valid"]:
            reactant_issues.append(f"reactant {index} invalid: {validation['error']}")
    if reactant_issues:
        return {
            "valid": False,
            "reaction_id": reaction_id,
            "template": template,
            "issues": [*product_issues, *reactant_issues],
            "reactants": reactant_validations,
            "product": product_validation,
            "product_smiles": product_smiles,
            "product_matched": False if product_validation is not None else None,
            "generated_products": [],
            "generated_product_sets": [],
            "generated_product_count": 0,
            "generated_product_set_count": 0,
            "reaction_executed": False,
            "reactants_valid": all(validation["valid"] for validation in reactant_validations),
        }

    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    try:
        reaction = AllChem.ReactionFromSmarts(template["smarts"])
        rdChemReactions.ChemicalReaction.Initialize(reaction)
        expected_reactants = reaction.GetNumReactantTemplates()
        provided_reactants = len(reactant_smiles)
        if expected_reactants != provided_reactants:
            return {
                "valid": False,
                "reaction_id": reaction_id,
                "template": template,
                "issues": [
                    *product_issues,
                    "Reactant count mismatch: "
                    f"template expects {expected_reactants} reactant(s), "
                    f"but {provided_reactants} were provided.",
                ],
                "reactants": reactant_validations,
                "product": product_validation,
                "product_smiles": product_smiles,
                "product_matched": False if product_validation is not None else None,
                "generated_products": [],
                "generated_product_sets": [],
                "generated_product_count": 0,
                "generated_product_set_count": 0,
                "reaction_executed": False,
                "reactants_valid": True,
                "expected_reactant_count": expected_reactants,
                "provided_reactant_count": provided_reactants,
            }
        reactant_mols = [Chem.MolFromSmiles(v["canonical_smiles"]) for v in reactant_validations]
        (
            generated,
            generated_product_sets,
            productive_orders,
            attempted_orders,
            skipped_invalid_product_sets,
        ) = _run_reaction_with_permutations(
            reaction,
            reactant_mols,
            Chem,
        )
    except Exception as exc:
        return {
            "valid": False,
            "reaction_id": reaction_id,
            "template": template,
            "issues": [*product_issues, f"RDKit reaction execution failed: {exc}"],
            "reactants": reactant_validations,
            "product": product_validation,
            "product_smiles": product_smiles,
            "product_matched": False if product_validation is not None else None,
            "generated_products": [],
            "generated_product_sets": [],
            "generated_product_count": 0,
            "generated_product_set_count": 0,
            "reaction_executed": False,
            "reactants_valid": True,
            "expected_reactant_count": template.get("reactant_count"),
            "provided_reactant_count": len(reactant_smiles),
        }

    issues = list(product_issues)
    if not generated:
        issues.append(
            "No products generated for the provided order or alternate reactant orders. "
            "Inspect reactant roles and try a different template or reactants."
            if len(reactant_smiles) > 1
            else "No products generated. Inspect the reactant role and try a different template or reactant."
        )
    used_order = (
        productive_orders[0] if productive_orders else list(range(1, len(reactant_smiles) + 1))
    )
    product_matched = None
    if product_validation is not None and product_validation["valid"]:
        claimed = product_validation["canonical_smiles"]
        product_matched = claimed in generated
        if not product_matched:
            issues.append("Claimed product was not generated by this template/reactant set")
    elif product_validation is not None:
        product_matched = False
    return {
        "valid": bool(product_matched) if product_validation is not None else True,
        "reaction_id": reaction_id,
        "template": template,
        "issues": issues,
        "reactants": reactant_validations,
        "product": product_validation,
        "product_smiles": product_smiles,
        "product_matched": product_matched,
        "generated_products": generated[:max_products],
        "generated_product_sets": generated_product_sets[:max_products],
        "generated_product_count": len(generated),
        "generated_product_set_count": len(generated_product_sets),
        "skipped_invalid_product_set_count": skipped_invalid_product_sets,
        "reaction_executed": True,
        "reactants_valid": True,
        "expected_reactant_count": reaction.GetNumReactantTemplates(),
        "provided_reactant_count": len(reactant_smiles),
        "attempted_reactant_orders": attempted_orders,
        "used_reactant_order": used_order,
        "reactant_order_permuted": used_order != list(range(1, len(reactant_smiles) + 1)),
        "productive_reactant_orders": productive_orders,
    }


def simulate_reverse_reaction(
    conn: sqlite3.Connection,
    reaction_id: str,
    product_smiles: str | list[str],
    *,
    max_outcomes: int,
) -> dict[str, Any]:
    """Run a reaction template backwards from product molecule(s) to candidate reactant sets."""

    template, issue = lookup_reaction_template(conn, reaction_id)
    if template is None:
        return {"valid": False, "reaction_id": reaction_id, "issues": [issue]}

    product_inputs = _normalize_smiles_inputs(product_smiles)
    product_validations = [chem.validate_smiles(smiles) for smiles in product_inputs]
    product_issues = _smiles_validation_issues("product", product_validations)
    if product_issues:
        return {
            "valid": False,
            "reaction_id": reaction_id,
            "template": template,
            "issues": product_issues,
            "products": product_validations,
            "product": product_validations[0] if len(product_validations) == 1 else None,
            "product_smiles": product_smiles,
            "candidate_reactant_sets": [],
            "candidate_count": 0,
            "returned_candidate_count": 0,
            "reaction_executed": False,
        }

    reactants, products = db.split_reaction_smarts(template["smarts"])
    expected_reactants, expected_products = db.reaction_template_arity(str(template["smarts"]))
    if len(product_inputs) != expected_products:
        return {
            "valid": False,
            "reaction_id": reaction_id,
            "template": template,
            "issues": [
                "Product count mismatch for reverse simulation: "
                f"template has {expected_products} product pattern(s), "
                f"but {len(product_inputs)} product SMILES value(s) were provided. "
                "For multi-product templates, provide all product components as product_smiles."
            ],
            "products": product_validations,
            "product": product_validations[0] if len(product_validations) == 1 else None,
            "product_smiles": product_smiles,
            "candidate_reactant_sets": [],
            "candidate_count": 0,
            "returned_candidate_count": 0,
            "reaction_executed": False,
            "product_template_count": expected_products,
            "provided_product_count": len(product_inputs),
        }

    reverse_smarts = f"{products}>>{reactants}"
    warnings = []
    if template.get("introduces_unmapped_product_atoms"):
        warnings.append(
            "This forward template introduces reagent-supplied product atoms. Reverse results may be "
            "abstract synthons and may omit implicit reagents."
        )

    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    try:
        reaction = AllChem.ReactionFromSmarts(reverse_smarts)
        rdChemReactions.ChemicalReaction.Initialize(reaction)
        product_mols = [Chem.MolFromSmiles(v["canonical_smiles"]) for v in product_validations]
        candidate_sets, productive_orders, attempted_orders, skipped_invalid_sets = (
            _run_reverse_reaction(
                reaction,
                product_mols,
                Chem,
                max_outcomes=max_outcomes,
            )
        )
    except Exception as exc:
        return {
            "valid": False,
            "reaction_id": reaction_id,
            "template": template,
            "issues": [f"RDKit reverse reaction execution failed: {exc}"],
            "warnings": warnings,
            "products": product_validations,
            "product": product_validations[0] if len(product_validations) == 1 else None,
            "product_smiles": product_smiles,
            "candidate_reactant_sets": [],
            "candidate_count": 0,
            "returned_candidate_count": 0,
            "reaction_executed": False,
            "reverse_smarts": reverse_smarts,
        }

    issues = []
    if not candidate_sets:
        issues.append(
            "RDKit did not produce any valid reverse precursor sets for this template/product pair, "
            "though this does not prove the template cannot work forward."
        )
    return {
        "valid": True,
        "reaction_id": reaction_id,
        "template": template,
        "issues": issues,
        "warnings": warnings,
        "products": product_validations,
        "product": product_validations[0] if len(product_validations) == 1 else None,
        "product_smiles": product_smiles,
        "candidate_reactant_sets": candidate_sets,
        "candidate_count": len(candidate_sets),
        "returned_candidate_count": len(candidate_sets),
        "skipped_invalid_candidate_set_count": skipped_invalid_sets,
        "reaction_executed": True,
        "reverse_smarts": reverse_smarts,
        "expected_product_count": expected_products,
        "provided_product_count": len(product_inputs),
        "generated_reactant_count": expected_reactants,
        "attempted_product_orders": attempted_orders,
        "used_product_order": productive_orders[0]
        if productive_orders
        else list(range(1, len(product_inputs) + 1)),
        "product_order_permuted": bool(productive_orders)
        and productive_orders[0] != list(range(1, len(product_inputs) + 1)),
        "productive_product_orders": productive_orders,
        "template_introduces_unmapped_product_atoms": bool(
            template.get("introduces_unmapped_product_atoms")
        ),
    }


def _run_reaction_with_permutations(
    reaction: Any,
    reactant_mols: list[Any],
    Chem: Any,
) -> tuple[list[str], list[list[str]], list[list[int]], int, int]:
    identity = tuple(range(len(reactant_mols)))
    orders = [identity]
    if len(reactant_mols) > 1:
        orders.extend(order for order in itertools.permutations(identity) if order != identity)

    attempted = 0
    productive_orders: list[list[int]] = []
    generated: set[str] = set()
    generated_product_sets: list[list[str]] = []
    seen_product_sets: set[tuple[str, ...]] = set()
    skipped_invalid_product_sets = 0
    for order in orders:
        attempted += 1
        order_product_sets = reaction.RunReactants(tuple(reactant_mols[index] for index in order))
        for product_tuple in order_product_sets:
            product_set = _canonicalize_sanitized_molecule_tuple(product_tuple, Chem)
            if product_set is None:
                skipped_invalid_product_sets += 1
                continue
            if not product_set or product_set in seen_product_sets:
                continue
            seen_product_sets.add(product_set)
            generated_product_sets.append(list(product_set))
            generated.update(product_set)
        if generated:
            productive_orders.append([index + 1 for index in order])
            break
    return (
        sorted(generated),
        generated_product_sets,
        productive_orders,
        attempted,
        skipped_invalid_product_sets,
    )


def _run_reverse_reaction(
    reaction: Any,
    product_mols: list[Any],
    Chem: Any,
    *,
    max_outcomes: int,
) -> tuple[list[dict[str, Any]], list[list[int]], int, int]:
    generated: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    identity = tuple(range(len(product_mols)))
    orders = [identity]
    if len(product_mols) > 1:
        orders.extend(order for order in itertools.permutations(identity) if order != identity)
    attempted = 0
    skipped_invalid_sets = 0
    productive_orders: list[list[int]] = []
    for order in orders:
        attempted += 1
        for reactant_tuple in reaction.RunReactants(tuple(product_mols[index] for index in order)):
            reactants = _canonicalize_sanitized_molecule_tuple(reactant_tuple, Chem)
            if reactants is None:
                skipped_invalid_sets += 1
                continue
            if not reactants or reactants in seen:
                continue
            seen.add(reactants)
            generated.append(
                {
                    "reactants": list(reactants),
                    "reactant_count": len(reactants),
                }
            )
            if len(generated) >= max_outcomes:
                break
        if generated:
            productive_orders.append([index + 1 for index in order])
            break
    return generated, productive_orders, attempted, skipped_invalid_sets


def _canonicalize_sanitized_molecule_tuple(molecules: Any, Chem: Any) -> tuple[str, ...] | None:
    smiles = []
    for molecule in molecules:
        if molecule is None:
            continue
        try:
            Chem.SanitizeMol(molecule)
        except Exception:
            return None
        smiles.append(Chem.MolToSmiles(molecule, canonical=True))
    return tuple(smiles)


def _normalize_smiles_inputs(values: str | list[str] | None) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        if values == "":
            return []
        return [values]
    return [str(value) for value in values if str(value)]


def _smiles_validation_issues(label: str, validations: list[dict[str, Any]]) -> list[str]:
    issues = []
    for index, validation in enumerate(validations, start=1):
        if not validation["valid"]:
            prefix = f"{label} invalid" if len(validations) == 1 else f"{label} {index} invalid"
            issues.append(f"{prefix}: {validation['error']}")
    return issues


def _match_molecules_to_reaction_roles(
    mols: list[Any],
    template_mols: list[Any],
) -> dict[str, Any]:
    if not mols:
        return {"matched": True, "mappings": []}
    if len(mols) > len(template_mols):
        return {"matched": False, "mappings": []}
    candidate_positions: list[list[int]] = []
    for mol in mols:
        positions = [
            index
            for index, template_mol in enumerate(template_mols)
            if mol is not None and template_mol is not None and mol.HasSubstructMatch(template_mol)
        ]
        if not positions:
            return {"matched": False, "mappings": []}
        candidate_positions.append(positions)
    for assignment in itertools.product(*candidate_positions):
        if len(set(assignment)) != len(assignment):
            continue
        return {
            "matched": True,
            "mappings": [
                {
                    "query_index": query_index + 1,
                    "template_position": template_index + 1,
                }
                for query_index, template_index in enumerate(assignment)
            ],
        }
    return {"matched": False, "mappings": []}


def _structure_template_match_score(
    *,
    reactant_inputs: list[str],
    product_inputs: list[str],
    template: dict[str, Any],
    forward_summary: dict[str, Any] | None,
) -> float:
    # Every match covers all supplied molecules, so only complete role coverage and
    # forward-simulation evidence distinguish templates within one search.
    score = 0.0
    if reactant_inputs and len(reactant_inputs) == int(template.get("reactant_count") or 0):
        score += 10.0
    if product_inputs and len(product_inputs) == int(template.get("product_count") or 0):
        score += 10.0
    if forward_summary:
        if forward_summary.get("reaction_executed"):
            score += 5.0
        requested = forward_summary.get("requested_products_generated")
        if isinstance(requested, dict) and requested:
            score += 25.0 * (
                sum(1 for matched in requested.values() if matched) / max(1, len(requested))
            )
        if forward_summary.get("requested_product_set_generated"):
            score += 20.0
    return score


def _rdkit_reaction_modules():
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdChemReactions

    return Chem, AllChem, rdChemReactions


def _first_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    lower_map = {str(key).lower(): value for key, value in record.items()}
    for key in keys:
        value = lower_map.get(key)
        if value not in (None, ""):
            return value
    return None


def _normalize_tags(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in re.split(r"[,;|]", str(value)) if part.strip()]


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if not isinstance(value, str):
        return {}
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if not isinstance(value, str):
        return []
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return []
    return decoded if isinstance(decoded, list) else []


def _dedupe_texts(values: list[str]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        text = value.strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        result.append(text)
    return result


def _infer_description(smarts: str) -> str:
    arity, product_count = db.reaction_template_arity(smarts)
    return f"Template with {arity} reactant pattern(s) and {product_count} product pattern(s)."
