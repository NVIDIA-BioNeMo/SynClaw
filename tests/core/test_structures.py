# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The building-block file readers and the salt policy applied at import."""

import gzip
import sqlite3
from contextlib import closing

import pytest
from rdkit import Chem

from synclaw.core import db
from synclaw.core.structures import open_structures

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
    "name, file_format, ids",
    [
        ("text", "text", ["", ""]),
        ("gz", "text", ["", ""]),
        ("smi", "smi", ["1", "2"]),
        ("sdf", "sdf", ["1", "2"]),
    ],
)
def test_every_format_reads_the_same_molecules(files, name, file_format, ids):
    with open_structures(files[name], file_format) as (_, _, rows):
        read = [(source_id, Chem.MolToSmiles(mol)) for _, source_id, mol, _ in rows]

    assert read == list(zip(ids, ["CCO", "CCN.Cl"], strict=True))


@pytest.mark.parametrize(
    "policy, expected", [("keep", {"CCO", "CCN.Cl"}), ("major", {"CCO", "CCN"}), ("drop", {"CCO"})]
)
def test_ingest_applies_the_salt_policy_whatever_the_format(tmp_path, files, policy, expected):
    stats = db.ingest_building_blocks(
        files["sdf"], tmp_path / "bb.sqlite", file_format="sdf", multicomponent=policy
    )

    with closing(sqlite3.connect(tmp_path / "bb.sqlite")) as conn:
        stored = {row[0] for row in conn.execute("SELECT canonical_smiles FROM building_blocks")}
    assert stored == expected
    assert stats["multicomponent_dropped"] == (1 if policy == "drop" else 0)
