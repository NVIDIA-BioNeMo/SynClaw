# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The building-block file readers and the salt policy applied at import."""

import gzip
import sqlite3
from contextlib import closing

import pytest
from rdkit import Chem

from synclaw.core import db
from synclaw.core.structures import detect_format, open_structures

SMILES = ["CCO", "CCN.Cl"]


def _sdf(smiles: list[str]) -> str:
    return "".join(
        f"{Chem.MolToMolBlock(Chem.MolFromSmiles(s))}>  <EMOL_VERSION_ID>\n{i}\n\n$$$$\n"
        for i, s in enumerate(smiles, start=1)
    )


@pytest.fixture
def files(tmp_path):
    contents = {
        "text": "SMILES\nCCO\n\nCCN.Cl\n",
        "smi": "isosmiles version_id\nCCO 1\nCCN.Cl 2\n",
        "sdf": _sdf(SMILES),
    }
    paths = {}
    for name, text in contents.items():
        paths[name] = tmp_path / f"{name}.txt"
        paths[name].write_text(text)
    paths["gz"] = tmp_path / "text.txt.gz"
    with gzip.open(paths["gz"], "wt") as handle:
        handle.write(contents["text"])
    return paths


@pytest.mark.parametrize(
    "name, ids",
    [("text", ["", ""]), ("gz", ["", ""]), ("smi", ["1", "2"]), ("sdf", ["1", "2"])],
)
def test_every_format_is_detected_and_reads_the_same_molecules(files, name, ids):
    with open_structures(files[name]) as (_, _, rows):
        read = [(source_id, Chem.MolToSmiles(mol)) for _, source_id, mol, _ in rows]

    assert read == list(zip(ids, ["CCO", "CCN.Cl"], strict=True))


def test_an_sdf_is_recognised_by_its_content_and_a_wrong_choice_is_loud(tmp_path, files):
    misnamed = tmp_path / "catalog.txt"
    misnamed.write_text(files["sdf"].read_text())
    assert detect_format(misnamed) == "sdf"
    assert detect_format(files["text"]) == "smiles"

    stats = db.ingest_building_blocks(misnamed, tmp_path / "bb.sqlite")
    assert stats["inserted"] == 2
    with pytest.raises(RuntimeError, match="unreadable"):
        db.ingest_building_blocks(misnamed, tmp_path / "wrong.sqlite", file_format="smiles")


@pytest.mark.parametrize(
    "policy, expected", [("keep", {"CCO", "CCN.Cl"}), ("major", {"CCO", "CCN"}), ("drop", {"CCO"})]
)
def test_ingest_applies_the_salt_policy_whatever_the_format(tmp_path, files, policy, expected):
    stats = db.ingest_building_blocks(files["sdf"], tmp_path / "bb.sqlite", multicomponent=policy)

    with closing(sqlite3.connect(tmp_path / "bb.sqlite")) as conn:
        stored = {row[0] for row in conn.execute("SELECT canonical_smiles FROM building_blocks")}
    assert stored == expected
    assert stats["multicomponent_dropped"] == (1 if policy == "drop" else 0)
