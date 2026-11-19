# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source memberships, union selection, and migration of the bundled catalog."""

import hashlib
import json
import re
from typing import Any

import pytest
from rdkit.Chem import rdChemReactions

from synclaw import paths
from synclaw.cli import build_parser
from synclaw.core import db, libraries
from synclaw.core.libraries import DEFAULT_REACTION_LIBRARIES, normalize_libraries
from synclaw.core.reactions import ingest_reaction_libraries, ingest_reaction_templates
from synclaw.rollout.loop import call_local_tool
from synclaw.rollout.prompts import validate_model_run_database
from synclaw.tools.synthesis import SynthesisTools

ASSETS = paths.DATA_DIR / "reaction_templates"


def libraries_json() -> dict[str, Any]:
    return libraries.bundled_reaction_catalog()


def catalog_families() -> dict[str, dict[str, Any]]:
    return json.loads((ASSETS / libraries_json()["families"]["file"]).read_text())


def catalog_records() -> list[dict[str, Any]]:
    catalog = libraries_json()
    return [
        row
        for source in catalog["sources"].values()
        for row in db.iter_jsonl(ASSETS / source["file"])
    ]


def test_source_catalog_has_unique_smarts_provenance_and_consistent_manifest():
    records = catalog_records()
    catalog = libraries_json()
    ids = {r["reaction_id"] for r in records}
    assert len(ids) == len(records) == len({r["smarts"] for r in records}) == 501
    for name, source in catalog["sources"].items():
        file = ASSETS / source["file"]
        assert hashlib.sha256(file.read_bytes()).hexdigest() == source["sha256"]
        assert len(list(db.iter_jsonl(file))) == source["templates"]
        assert all(r["library"] == name for r in db.iter_jsonl(file))
    # SynFormer keeps its original line numbers; line R033 duplicated R032 and is not stored.
    assert {r["reaction_id"] for r in records if r["library"] == "synformer"} == {
        f"synformer:R{i:03d}" for i in range(115) if i != 33
    }
    families_file = ASSETS / catalog["families"]["file"]
    assert hashlib.sha256(families_file.read_bytes()).hexdigest() == catalog["families"]["sha256"]
    families = catalog_families()
    assert {row["family"] for row in records} == set(families)
    for family in families.values():
        assert family["description"].strip() and family["search_terms"]
    for row in records:
        assert row["metadata"]["provenance"]
        rxn = rdChemReactions.ReactionFromSmarts(row["smarts"])
        assert rxn is not None
        assert len(row["reactant_roles"]) == rxn.GetNumReactantTemplates()


@pytest.fixture
def catalog_db(tmp_path):
    path = tmp_path / "catalog.sqlite"
    stats = ingest_reaction_libraries(path)
    assert stats["inserted"] == stats["records"] == 501
    return path


def test_imported_default_union_needs_no_runtime_annotation_files(catalog_db, monkeypatch):
    def unexpected_file_read(*args, **kwargs):
        raise AssertionError("Runtime annotation lookup must use the database")

    monkeypatch.setattr(db, "iter_jsonl", unexpected_file_read)
    with db.session(catalog_db) as conn:
        rows = db.list_reaction_templates(conn, library=DEFAULT_REACTION_LIBRARIES)
    assert len(rows) == 501
    assert all(row["metadata"]["annotation"]["family_metadata"] for row in rows)
    overview = call_local_tool(SynthesisTools(catalog_db), "search_reaction_templates", {})
    assert overview["template_count"] == 501
    assert overview["families_note"] == (
        f"The 8 largest of {overview['family_count']} families; search by text to find others."
    )
    for source in DEFAULT_REACTION_LIBRARIES:
        reaction_id = next(row["reaction_id"] for row in rows if row["library"] == source)
        result = call_local_tool(
            SynthesisTools(catalog_db),
            "search_reaction_templates",
            {"query": reaction_id},
        )
        assert result["library"] == DEFAULT_REACTION_LIBRARIES
        assert reaction_id in {row["reaction_id"] for row in result["results"]}


def test_raw_synformer_rows_require_explicit_annotated_import(tmp_path):
    database = tmp_path / "raw.sqlite"
    source = tmp_path / "raw.txt"
    template = next(row for row in catalog_records() if row["reaction_id"] == "synformer:R000")
    source.write_text(template["smarts"] + "\n")
    ingest_reaction_templates(source, database, library="synformer")
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("C\n")
    db.ingest_building_blocks(blocks, database)
    db.prepare_building_block_search(database)

    with db.session(database) as conn:
        row = db.get_reaction_template(conn, "synformer:R000")
    assert row["reaction_family"] == ""
    assert "annotation" not in row["metadata"]
    with pytest.raises(ValueError, match="missing family_metadata.*synclaw ingest-reactions --db"):
        validate_model_run_database(database, library="synformer")

    ingest_reaction_libraries(database)
    validate_model_run_database(database, library=DEFAULT_REACTION_LIBRARIES)
    with db.session(database) as conn:
        row = db.get_reaction_template(conn, "synformer:R000")
    assert row["reaction_family"] == template["family"]
    assert row["family_metadata"] == catalog_families()[template["family"]]


@pytest.mark.parametrize(
    ("selection", "count"),
    [
        (["synformer"], 114),
        (["paroutes"], 176),
        (["ord"], 203),
        (["nvidia"], 8),
        (["synformer", "paroutes"], 290),
        (["synformer", "ord"], 317),
        (["paroutes", "ord"], 379),
        (["ord", "nvidia"], 211),
        (list(DEFAULT_REACTION_LIBRARIES), 501),
    ],
)
def test_unions_drive_search_and_the_family_overview(catalog_db, selection, count):
    with db.session(catalog_db) as conn:
        db.require_reaction_library(conn, selection)
        rows = db.list_reaction_templates(conn, library=selection)
    assert len(rows) == len({r["smarts"] for r in rows}) == count
    expected = {r["reaction_id"] for r in catalog_records() if r["library"] in selection}
    assert {r["reaction_id"] for r in rows} == expected
    overview = SynthesisTools(catalog_db, reaction_libraries=selection).search_reaction_templates()
    assert overview["template_count"] == count


def test_reimport_preserves_stock_and_replaces_raw_synformer_rows(tmp_path):
    database = tmp_path / "stock.sqlite"
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("CN\nCC(=O)O\nCCO\n")
    db.ingest_building_blocks(blocks, database)
    db.prepare_building_block_search(database)
    synformer = [r for r in catalog_records() if r["library"] == "synformer"]
    source = tmp_path / "raw.txt"
    source.write_text("\n".join(r["smarts"] for r in synformer) + "\n")
    ingest_reaction_templates(source, database, library="synformer")
    custom = tmp_path / "custom.txt"
    custom.write_text("[C:1]>>[C:1]\n")
    ingest_reaction_templates(custom, database, library="custom")
    with db.session(database) as conn:
        stock = [tuple(r) for r in conn.execute("SELECT * FROM building_blocks ORDER BY id")]
        custom_before = [
            tuple(r)
            for r in conn.execute("SELECT * FROM reaction_templates WHERE library = 'custom'")
        ]
    stats = ingest_reaction_libraries(database)
    assert stats["replaced"] == 114
    again = ingest_reaction_libraries(database)
    assert again["replaced"] == again["inserted"] == 501
    with db.session(database) as conn:
        assert [
            tuple(r) for r in conn.execute("SELECT * FROM building_blocks ORDER BY id")
        ] == stock
        assert [
            tuple(r)
            for r in conn.execute("SELECT * FROM reaction_templates WHERE library = 'custom'")
        ] == custom_before
        assert len(db.list_reaction_templates(conn, library=DEFAULT_REACTION_LIBRARIES)) == 501
        assert db.get_reaction_template(conn, "synformer:R033") is None
    db.require_prepared_building_block_search(database)


def test_bundled_templates_execute(catalog_db):
    tools = SynthesisTools(catalog_db)
    for reaction_id in ("synformer:R008", "nvidia:R000"):
        assert tools.simulate_reaction(
            reaction_id=reaction_id,
            reactants=["CN", "CC(=O)O"],
            product_smiles="CNC(C)=O",
        )["product_matched"]
    products = tools.simulate_reaction(reaction_id="ord:R253", reactants=["CC(=O)O", "CNC"])
    assert any(len(ps) == 2 for ps in products["generated_product_sets"])


def test_union_text_and_structural_search_covers_only_selected_libraries(tmp_path):
    records = [
        {
            "reaction_id": "P",
            "library": "paroutes",
            "smarts": "[C:1][O:2]>>[C:1]=[O:2]",
            "name": "Alcohol oxidation",
            "token_aliases": {"dehydrogenation": ["oxidation"]},
        },
        {
            "reaction_id": "O",
            "library": "ord",
            "smarts": "[C:1][O:2]>>[C:1][O:2]",
            "name": "Alcohol identity",
        },
        {
            "reaction_id": "S",
            "library": "synformer",
            "smarts": "[C:1][N:2]>>[C:1][N:2]",
            "name": "Amine identity",
        },
    ]
    database = tmp_path / "fixture.sqlite"
    for row in records:
        source = tmp_path / f"{row['library']}.jsonl"
        source.write_text(json.dumps(row) + "\n")
        ingest_reaction_templates(source, database, library=row["library"])
    tools = SynthesisTools(database)
    hits = tools.search_reaction_templates("alcohol", library=["ord"])["results"]
    assert [r["reaction_id"] for r in hits] == ["ord:O"]
    assert tools.search_reaction_templates("dehydrogenation", library=["ord"])["results"] == []
    for selection in (["paroutes", "ord"], ["ord", "ord", "paroutes"]):
        hits = tools.search_reaction_templates("alcohol", library=selection)["results"]
        assert sorted(r["reaction_id"] for r in hits) == ["ord:O", "paroutes:P"]
        hits = tools.search_reaction_templates("dehydrogenation", library=selection)["results"]
        assert [r["reaction_id"] for r in hits] == ["paroutes:P"]
        hits = tools.search_reaction_templates(reactant_smiles=["CCO"], library=selection)[
            "results"
        ]
        assert sorted(r["reaction_id"] for r in hits) == ["ord:O", "paroutes:P"]
        both = tools.search_reaction_templates(
            "oxidation", reactant_smiles=["CCO"], library=selection
        )
        assert sorted(r["reaction_id"] for r in both["structural_results"]) == [
            "ord:O",
            "paroutes:P",
        ]
        assert [r["reaction_id"] for r in both["text_results"]] == ["paroutes:P"]
    with db.session(database) as conn:
        with pytest.raises(RuntimeError, match="'absent'.*available:"):
            db.require_reaction_library(conn, ["ord", "absent"])


def test_cli_imports_catalog_and_selects_union(tmp_path):
    args = build_parser().parse_args(["ingest-reactions", "--db", str(tmp_path / "catalog.sqlite")])
    assert args.func(args)["records"] == 501
    args = build_parser().parse_args(["mcp", "serve", "--reaction-library", "synformer", "ord"])
    assert args.reaction_library == ["synformer", "ord"]
    assert normalize_libraries(["ord,paroutes", "ord", " synformer "]) == (
        "ord",
        "paroutes",
        "synformer",
    )
    with pytest.raises(ValueError, match="Reaction libraries"):
        normalize_libraries([123])
    args = build_parser().parse_args(["ingest-reactions", "--library", "ord"])
    with pytest.raises(RuntimeError, match="require a custom template file"):
        args.func(args)


def test_bundled_import_records_catalog_and_detects_stale_or_untracked_libraries(catalog_db):
    with db.session(catalog_db) as conn:
        records = db.reaction_library_catalog(conn)
        assert set(records) == set(DEFAULT_REACTION_LIBRARIES)
        assert records["ord"]["template_count"] == 203
        assert records["nvidia"]["template_count"] == 8
        db.require_reaction_library(conn, None)
        conn.execute(
            "UPDATE reaction_library_catalog SET catalog_sha256 = 'old' WHERE library = 'ord'"
        )
        with pytest.raises(RuntimeError, match="'ord' is out of date"):
            db.require_reaction_library(conn, ["ord"])
        # Each source is pinned separately, so other selections stay usable.
        db.require_reaction_library(conn, ["synformer", "paroutes", "nvidia"])
    ingest_reaction_libraries(catalog_db)
    with db.session(catalog_db) as conn:
        db.require_reaction_library(conn, ["ord"])
        # Rows imported before tracking (e.g. the old rxn-template import) have no record.
        conn.execute(
            "INSERT INTO reaction_templates(reaction_id, library, name, description, smarts, created_at, updated_at) "
            "VALUES ('rxn-template:R000', 'rxn-template', '', '', '[C:1]>>[C:1]', '', '')"
        )
        db.require_reaction_library(conn, DEFAULT_REACTION_LIBRARIES)
        with pytest.raises(
            RuntimeError, match="'rxn-template' has no import record.*remove-reactions"
        ):
            db.require_reaction_library(conn, None)
        assert db.remove_reaction_libraries(conn, "rxn-template") == 1
        db.require_reaction_library(conn, None)


def test_staleness_is_checked_per_source_and_hints_fit_each_library(catalog_db):
    with db.session(catalog_db) as conn:
        db.require_reaction_library(conn, DEFAULT_REACTION_LIBRARIES)
        conn.execute(
            "UPDATE reaction_library_catalog SET catalog_sha256 = 'old' WHERE library = 'paroutes'"
        )
        with pytest.raises(
            RuntimeError, match="'paroutes' is out of date.*rebuild the prepared database"
        ):
            db.require_reaction_library(conn, DEFAULT_REACTION_LIBRARIES)
        db.require_reaction_library(conn, ["synformer", "ord", "nvidia"])
        assert db.remove_reaction_libraries(conn, "nvidia") == 8
        # Each missing library gets the fix that installs it: the bundled import or a custom one.
        with pytest.raises(RuntimeError, match="'nvidia', 'mylib' has no templates") as missing:
            db.require_reaction_library(conn, ["nvidia", "mylib"])
        assert "import the bundled sources" in str(missing.value)
        assert "--library mylib" in str(missing.value) and "--library nvidia" not in str(
            missing.value
        )
        # A bundled source this version no longer ships cannot be fixed by reimporting.
        conn.execute(
            "UPDATE reaction_library_catalog SET library = 'retired' WHERE library = 'ord'"
        )
        conn.execute("UPDATE reaction_templates SET library = 'retired' WHERE library = 'ord'")
        with pytest.raises(
            RuntimeError, match="'retired' was imported as a bundled source.*remove-reactions"
        ):
            db.require_reaction_library(conn, ["retired"])


def test_custom_reimport_replaces_only_that_library(catalog_db, tmp_path):
    source = tmp_path / "custom.txt"
    source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n[C:1][N:2]>>[C:1]=[N:2]\n")
    ingest_reaction_templates(source, catalog_db, library="custom")
    source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n")
    stats = ingest_reaction_templates(source, catalog_db, library="custom")
    assert stats["replaced"] == 2
    with db.session(catalog_db) as conn:
        assert len(db.list_reaction_templates(conn, library="custom")) == 1
        assert len(db.list_reaction_templates(conn, library=DEFAULT_REACTION_LIBRARIES)) == 501
        record = db.reaction_library_catalog(conn)["custom"]
    assert record["template_count"] == 1
    assert record["catalog_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_lean_search_snapshot_keeps_library_records(catalog_db, tmp_path):
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("CN\n")
    db.ingest_building_blocks(blocks, catalog_db)
    lean = tmp_path / "lean.sqlite"
    db.prepare_building_block_search(catalog_db, output_path=lean)
    validate_model_run_database(lean, library=DEFAULT_REACTION_LIBRARIES)
    with db.session(lean) as conn:
        assert set(db.reaction_library_catalog(conn)) == set(DEFAULT_REACTION_LIBRARIES)


def test_library_selection_is_a_hard_boundary_for_template_execution(catalog_db, tmp_path):
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("CN\nCC(=O)O\n")
    db.ingest_building_blocks(blocks, catalog_db)
    tools = SynthesisTools(catalog_db, reaction_libraries=["synformer"])
    outside = "outside this run's reaction libraries (synformer)"

    lookup = tools.get_reaction_template("nvidia:R000")
    assert not lookup["found"]
    assert outside in lookup["issues"][0]
    forward = tools.simulate_reaction(
        reaction_id="nvidia:R000", reactants=["CN", "CC(=O)O"], product_smiles="CNC(C)=O"
    )
    assert not forward["valid"] and outside in forward["issues"][0]
    reverse = tools.simulate_reverse_reaction(reaction_id="nvidia:R000", product_smiles="CNC(C)=O")
    assert not reverse["valid"] and outside in reverse["issues"][0]
    submission = tools.submit_pathway(
        steps=[
            {
                "reaction_id": "nvidia:R000",
                "reactants": ["CN", "CC(=O)O"],
                "product_smiles": "CNC(C)=O",
            }
        ],
        target_smiles="CNC(C)=O",
    )
    assert any(outside in issue for issue in submission["issues"])
    # A caller-supplied library cannot widen the configured selection.
    search = tools.search_reaction_templates("nvidia:R000", library=["nvidia"])
    assert {row["library"] for row in search["results"]} <= {"synformer"}

    # Templates inside the selection still run.
    assert tools.get_reaction_template("synformer:R008")["found"]
    assert tools.simulate_reaction(
        reaction_id="synformer:R008",
        reactants=["CN", "CC(=O)O"],
        product_smiles="CNC(C)=O",
    )["product_matched"]
    unknown = SynthesisTools(catalog_db).simulate_reaction(reaction_id="nope:R1", reactants=["CN"])
    assert unknown["issues"] == ["Unknown reaction template: nope:R1"]


def test_ord_templates_trace_to_uspto_grants_and_nvidia_templates_are_separate() -> None:
    records = catalog_records()
    ord_members = [r for r in records if r["library"] == "ord"]
    assert len(ord_members) == 203
    for record in ord_members:
        sources = record["metadata"]["provenance"]
        assert [p["source_id"] for p in sources] == ["uspto-grants-lowe-2017"], record[
            "reaction_id"
        ]
        assert re.fullmatch(r"US\w+:\d+", sources[0]["source_record_id"]), record["reaction_id"]
    nvidia = [r for r in records if r["library"] == "nvidia"]
    assert [r["reaction_id"] for r in nvidia] == [f"nvidia:R{i:03d}" for i in range(8)]
    for record in nvidia:
        # Authored, not extracted; later entries cite recorded reactions the template reproduces.
        authored, *examples = record["metadata"]["provenance"]
        assert authored == {"source_id": "nvidia-authored"}
        assert 1 <= len(examples) <= 5, record["reaction_id"]
        for example in examples:
            assert example["relation"] == "example"
            assert example["source_id"] == "uspto-grants-lowe-2017"
            assert re.fullmatch(r"US\w+:\d+", example["source_record_id"]), record["reaction_id"]


def test_custom_record_fields_cannot_pose_as_curated_annotations(tmp_path):
    source = tmp_path / "custom.jsonl"
    rows = [
        {
            "reaction_id": "X1",
            "smarts": "[C:1][OH]>>[C:1]Cl",
            "annotation": {"family": "bogus", "search_terms": ["fake"]},
        },
        {"reaction_id": "X2", "smarts": "[C:1][NH2]>>[C:1]Cl", "metadata": "free-text note"},
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    database = tmp_path / "custom.sqlite"
    ingest_reaction_templates(source, database, library="custom")
    with db.session(database) as conn:
        first = db.get_reaction_template(conn, "custom:X1")
        second = db.get_reaction_template(conn, "custom:X2")
    assert first["reaction_family"] == "" and "fake" not in first["query_hints"]
    assert first["metadata"]["source_fields"]["annotation"]["family"] == "bogus"
    assert second["metadata"]["source_fields"]["metadata"] == "free-text note"
