# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundled catalog, its import, template numbering and staleness checks."""

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from typing import Any

import pytest
from rdkit.Chem import rdChemReactions

from synclaw import paths
from synclaw.cli import build_parser
from synclaw.core import db, libraries
from synclaw.core.libraries import normalize_libraries
from synclaw.core.reactions import ingest_reaction_libraries, ingest_reaction_templates
from synclaw.rollout.loop import call_local_tool
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


def _numbered(count):
    return [f"R{number:03d}" for number in range(1, count + 1)]


def test_imported_catalog_needs_no_runtime_annotation_files(bundled_db, monkeypatch):
    def unexpected_file_read(*args, **kwargs):
        raise AssertionError("Runtime annotation lookup must use the database")

    monkeypatch.setattr(db, "iter_jsonl", unexpected_file_read)
    with db.session(bundled_db) as conn:
        rows = db.list_reaction_templates(conn)
    assert len(rows) == 501
    assert all(row["family_metadata"] for row in rows)
    tools = SynthesisTools(bundled_db)
    overview = call_local_tool(tools, "search_reaction_templates", {})
    assert overview["template_count"] == 501
    assert overview["families_note"] == (
        f"The 8 largest of {overview['family_count']} families; search by text to find others."
    )
    reaction_id = rows[0]["reaction_id"]
    result = call_local_tool(tools, "search_reaction_templates", {"query": reaction_id})
    assert result["results"][0]["reaction_id"] == reaction_id


def test_templates_are_numbered_across_libraries_and_renumbered_on_removal(catalog_db):
    with db.session(catalog_db) as conn:
        before = db.list_reaction_templates(conn)
        libraries = [
            library
            for (library,) in conn.execute(
                "SELECT library FROM reaction_templates ORDER BY reaction_id"
            )
        ]
        db.remove_reaction_libraries(conn, ["paroutes", "nvidia"])
        db.require_reaction_templates(conn)
        after = db.list_reaction_templates(conn)
    assert [row["reaction_id"] for row in before] == _numbered(501)
    # IDs follow no library: the first twenty already mix several.
    assert len(set(libraries[:20])) > 1
    assert [row["reaction_id"] for row in after] == _numbered(317)
    # The remaining templates keep their relative order.
    kept = {r["smarts"] for r in catalog_records() if r["library"] in ("synformer", "ord")}
    assert [row["smarts"] for row in after] == [r["smarts"] for r in before if r["smarts"] in kept]
    assert SynthesisTools(catalog_db).search_reaction_templates()["template_count"] == 317


def test_templates_are_addressed_only_by_numbered_ids(bundled_db, bundled_ids):
    tools = SynthesisTools(bundled_db)
    template = tools.get_reaction_template(bundled_ids["synformer:R008"])["template"]
    # Tool results, which MCP clients receive unfiltered, carry no library or source ID.
    assert not {"library", "metadata", "metadata_json", "tags_json"} & template.keys()
    unknown = tools.simulate_reaction(reaction_id="synformer:R008", reactants=["CN"])
    assert unknown["issues"] == ["Unknown reaction template: synformer:R008"]


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
        assert db.get_reaction_template(conn, "R001")["reaction_family"] == ""
    with pytest.raises(ValueError, match="missing family_metadata.*synclaw ingest-reactions --db"):
        db.validate_model_run_database(database)

    ingest_reaction_libraries(database)
    db.validate_model_run_database(database)
    with db.session(database) as conn:
        row = db.get_reaction_template_by_smarts(conn, template["smarts"])
    assert row["reaction_family"] == template["family"]
    assert row["family_metadata"] == catalog_families()[template["family"]]


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
    # Renumbering changes only reaction_id, so compare everything else.
    custom_query = (
        "SELECT name, description, smarts, tags_json, metadata_json, created_at, updated_at "
        "FROM reaction_templates WHERE library = 'custom'"
    )
    with db.session(database) as conn:
        stock = [tuple(r) for r in conn.execute("SELECT * FROM building_blocks ORDER BY id")]
        custom_before = [tuple(r) for r in conn.execute(custom_query)]
    stats = ingest_reaction_libraries(database)
    assert stats["replaced"] == 114
    again = ingest_reaction_libraries(database)
    assert again["replaced"] == again["inserted"] == 501
    with db.session(database) as conn:
        assert [
            tuple(r) for r in conn.execute("SELECT * FROM building_blocks ORDER BY id")
        ] == stock
        assert [tuple(r) for r in conn.execute(custom_query)] == custom_before
        assert [r["reaction_id"] for r in db.list_reaction_templates(conn)] == _numbered(502)
    db.require_prepared_building_block_search(database)


def test_bundled_templates_execute(bundled_db, bundled_ids):
    tools = SynthesisTools(bundled_db)
    for source_id in ("synformer:R008", "nvidia:R000"):
        assert tools.simulate_reaction(
            reaction_id=bundled_ids[source_id],
            reactants=["CN", "CC(=O)O"],
            product_smiles="CNC(C)=O",
        )["product_matched"]
    products = tools.simulate_reaction(
        reaction_id=bundled_ids["ord:R253"], reactants=["CC(=O)O", "CNC"]
    )
    assert any(len(ps) == 2 for ps in products["generated_product_sets"])


def test_cli_imports_the_bundled_catalog(tmp_path):
    args = build_parser().parse_args(["ingest-reactions", "--db", str(tmp_path / "catalog.sqlite")])
    assert args.func(args)["records"] == 501
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
        assert set(records) == set(libraries_json()["sources"])
        assert records["ord"]["template_count"] == 203
        assert records["nvidia"]["template_count"] == 8
        db.require_reaction_templates(conn)
        conn.execute(
            "UPDATE reaction_library_catalog SET catalog_sha256 = 'old' WHERE library = 'ord'"
        )
        with pytest.raises(
            RuntimeError, match="'ord' is out of date.*rebuild the prepared database"
        ):
            db.require_reaction_templates(conn)
    ingest_reaction_libraries(catalog_db)
    with db.session(catalog_db) as conn:
        db.require_reaction_templates(conn)
        conn.execute(
            "INSERT INTO reaction_templates(reaction_id, library, name, description, smarts, created_at, updated_at) "
            "VALUES ('untracked:R000', 'untracked', '', '', '[C:1]>>[C:1]', '', '')"
        )
        with pytest.raises(
            RuntimeError, match="'untracked' has no import record.*remove-reactions"
        ):
            db.require_reaction_templates(conn)
        assert db.remove_reaction_libraries(conn, "untracked") == 1
        db.require_reaction_templates(conn)
        # A bundled source this version no longer ships cannot be fixed by reimporting.
        conn.execute(
            "UPDATE reaction_library_catalog SET library = 'retired' WHERE library = 'ord'"
        )
        conn.execute("UPDATE reaction_templates SET library = 'retired' WHERE library = 'ord'")
        with pytest.raises(
            RuntimeError, match="'retired' was imported as a bundled source.*remove-reactions"
        ):
            db.require_reaction_templates(conn)


def test_templates_not_numbered_by_this_version_fail_loudly(catalog_db):
    with db.session(catalog_db) as conn:
        conn.execute("UPDATE reaction_templates SET reaction_id = 'synformer:R000' WHERE id = 1")
        with pytest.raises(RuntimeError, match="not numbered R001 onward.*ingest-reactions"):
            db.require_reaction_templates(conn)


def test_custom_reimport_replaces_only_that_library(catalog_db, tmp_path):
    source = tmp_path / "custom.txt"
    source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n[C:1][N:2]>>[C:1]=[N:2]\n")
    ingest_reaction_templates(source, catalog_db, library="custom")
    source.write_text("[C:1][O:2]>>[C:1]=[O:2]\n")
    stats = ingest_reaction_templates(source, catalog_db, library="custom")
    assert stats["replaced"] == 2
    with db.session(catalog_db) as conn:
        counts = dict(conn.execute("SELECT library, COUNT(*) FROM reaction_templates GROUP BY 1"))
        db.require_reaction_templates(conn)
        record = db.reaction_library_catalog(conn)["custom"]
    assert counts == {"custom": 1, "nvidia": 8, "ord": 203, "paroutes": 176, "synformer": 114}
    assert record["template_count"] == 1
    assert record["catalog_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()


def test_lean_search_snapshot_keeps_library_records(catalog_db, tmp_path):
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("CN\n")
    db.ingest_building_blocks(blocks, catalog_db)
    lean = tmp_path / "lean.sqlite"
    db.prepare_building_block_search(catalog_db, output_path=lean)
    db.validate_model_run_database(lean)
    with db.session(lean) as conn:
        assert set(db.reaction_library_catalog(conn)) == set(libraries_json()["sources"])


def test_a_smarts_already_in_another_library_is_rejected(tmp_path):
    for name in ("first", "second"):
        (tmp_path / f"{name}.txt").write_text("[C:1][O:2]>>[C:1]=[O:2]\n")
    database = tmp_path / "test.sqlite"
    ingest_reaction_templates(tmp_path / "first.txt", database, library="first")
    with pytest.raises(ValueError, match="already stored as R001"):
        ingest_reaction_templates(tmp_path / "second.txt", database, library="second")


def test_a_lean_snapshot_of_a_database_from_before_numbering_says_to_reingest(tmp_path):
    old = tmp_path / "old.sqlite"
    columns = "reaction_id, library, name, description, smarts, created_at, updated_at"
    with closing(sqlite3.connect(old)) as conn:
        conn.execute(f"CREATE TABLE reaction_templates ({columns})")
        conn.executemany(
            f"INSERT INTO reaction_templates ({columns}) VALUES (?, ?, 'n', 'd', ?, 't', 't')",
            [("synformer:R000", "first", "C>>C"), ("synformer:R000", "second", "N>>N")],
        )
        conn.commit()
    with pytest.raises(RuntimeError, match="ingest-reactions --db"):
        db.prepare_building_block_search(old, output_path=tmp_path / "lean.sqlite")


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
        stored = {
            metadata["source_id"]: (reaction_id, metadata)
            for reaction_id, metadata in (
                (row[0], json.loads(row[1]))
                for row in conn.execute("SELECT reaction_id, metadata_json FROM reaction_templates")
            )
        }
        first = db.get_reaction_template(conn, stored["custom:X1"][0])
    assert first["reaction_family"] == "" and "fake" not in first["query_hints"]
    assert stored["custom:X1"][1]["source_fields"]["annotation"]["family"] == "bogus"
    assert stored["custom:X2"][1]["source_fields"]["metadata"] == "free-text note"
