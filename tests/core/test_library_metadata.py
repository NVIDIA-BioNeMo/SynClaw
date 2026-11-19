# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Guard library metadata validation and the generated family overview."""

import json

import pytest

from synclaw.core import db
from synclaw.core.reactions import ingest_reaction_templates
from synclaw.rollout.policy import DEFAULT_REACTION_LIBRARY
from synclaw.rollout.prompts import validate_model_run_database
from synclaw.tools.synthesis import SynthesisTools


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"description": "Scope."},
        {"search_terms": ["family hint"]},
        {"description": " ", "search_terms": ["family hint"]},
        {"description": "Scope.", "search_terms": []},
    ],
)
def test_incomplete_family_metadata_fails_the_family_overview(tmp_path, metadata):
    records = [
        {
            "reaction_id": f"R{i:03d}",
            "smarts": "[C:1]>>[C:1]",
            "family": f"family {i}",
            "family_metadata": {"description": "Fixture family.", "search_terms": ["family hint"]},
        }
        for i in range(9)
    ]
    # The ninth family would be omitted from the displayed top eight, but must
    # still be validated. Personal template hints cannot fill the missing field.
    records[-1].pop("family_metadata")
    records[-1]["search_terms"] = ["misleading individual template hint"]
    if metadata is not None:
        records[-1]["family_metadata"] = metadata
    source = tmp_path / "templates.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    database = tmp_path / "bad.sqlite"
    ingest_reaction_templates(source, database, library="demo")
    overview = SynthesisTools(database, reaction_libraries="demo").search_reaction_templates
    with pytest.raises(ValueError, match="family 8.*missing family_metadata"):
        overview()
    with pytest.raises(ValueError, match="ingest-reactions --annotations"):
        overview()


def test_missing_reaction_library_fails_loudly_and_lists_what_exists(tmp_path):
    """A requested library with no templates must not degrade to empty search results."""
    db_path = tmp_path / "synclaw.sqlite"
    db.init_db(db_path)
    source = tmp_path / "templates.txt"
    source.write_text("[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]\n", encoding="utf-8")
    ingest_reaction_templates(source, db_path=db_path, library="synformer")

    with db.session(db_path) as conn:
        db.require_reaction_library(conn, "synformer")
        with pytest.raises(RuntimeError, match="available: 'synformer'"):
            db.require_reaction_library(conn, DEFAULT_REACTION_LIBRARY)


def test_annotations_that_match_nothing_fail_the_import_and_roll_it_back(tmp_path):
    """A library/ID mismatch must fail the import and roll it back, not report annotated: 0."""
    source = tmp_path / "templates.jsonl"
    source.write_text(
        json.dumps(
            {
                "reaction_id": "R001",
                "smarts": "[C:1]>>[C:1]",
                "family": "fam",
                "family_metadata": {"description": "Fixture family.", "search_terms": ["hint"]},
            }
        )
        + "\n"
    )
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "reaction_id": "R001",
                "library": "other-library",
                "name": "Amide coupling",
            }
        )
        + "\n"
    )
    database = tmp_path / "synclaw.sqlite"

    with pytest.raises(ValueError, match="matched an annotation"):
        ingest_reaction_templates(
            source,
            database,
            library="demo",
            annotations_path=annotations,
        )
    with db.session(database) as conn:
        assert conn.execute("SELECT COUNT(*) FROM reaction_templates").fetchone()[0] == 0


def test_incomplete_library_metadata_fails_before_the_run_starts(tmp_path):
    """The same error the prompt raises, reported once up front with the library named."""
    source = tmp_path / "templates.jsonl"
    source.write_text(
        json.dumps(
            {
                "reaction_id": "R001",
                "smarts": "[C:1]>>[C:1]",
                "family": "fam",
            }
        )
        + "\n"
    )
    database = tmp_path / "synclaw.sqlite"
    ingest_reaction_templates(source, database, library="demo")
    blocks = tmp_path / "blocks.smi"
    blocks.write_text("C\n")
    db.ingest_building_blocks(blocks, database)
    db.prepare_building_block_search(database)

    with pytest.raises(ValueError, match="family 'fam'.*missing family_metadata"):
        validate_model_run_database(database, library="demo")
