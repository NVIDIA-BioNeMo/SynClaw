# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from synclaw.core.libraries import bundled_reaction_catalog
from synclaw.tasks.matrix import (
    COLUMN_FILE,
    DROPPED_TEMPLATE_FILE,
    META_FILE,
    build_matrix_directory,
    compute_columns,
    load_template_records,
    reactant_queries,
    write_templates,
)
from synclaw.tasks.targets import (
    BUILDING_BLOCK_FILE,
    MATRIX_FILE,
    OFFSET_FILE,
    REACTION_FILE,
    REACTION_ID_FILE,
    GenerationOptions,
    generate_targets,
    line_offsets,
    load_matrix_dir,
    read_smiles_file,
)

# Blocks chosen to exercise each policy: a plain amine, an acid chloride, a nitroarene, a
# hydrochloride salt, an atom-mapped entry, a duplicate of the first, and an unparseable row.
CATALOG = [
    "CCN",
    "CC(=O)Cl",
    "O=[N+]([O-])c1ccccc1",
    "CCN.Cl",
    "[CH3:7]CO",
    "CCN",
    "not-a-molecule",
]


@pytest.fixture
def catalog(tmp_path: Path) -> Path:
    path = tmp_path / "catalog.smi.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("isosmiles parent_id\n")
        for index, smiles in enumerate(CATALOG):
            handle.write(f"{smiles} {index}\n")
    return path


def build(catalog: Path, directory: Path, **kwargs) -> dict:
    options = dict(libraries=["synformer"], workers=1, common_reagents=False)
    options.update(kwargs)
    return build_matrix_directory(building_blocks=catalog, output=directory, **options)


def test_a_built_directory_loads_and_its_files_agree(catalog: Path, tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    meta = build(catalog, directory)

    loaded = load_matrix_dir(directory)
    assert loaded.matrix.shape == (loaded.num_blocks, loaded.num_templates)
    assert loaded.num_blocks == meta["building_blocks"]["written"]
    assert loaded.num_templates == meta["templates"]["kept"]
    assert json.loads((directory / META_FILE).read_text())["matrix"]["fortran_order"] is True
    assert (directory / COLUMN_FILE).exists()


def test_every_matrix_bit_is_a_substructure_match(catalog: Path, tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    build(catalog, directory)
    loaded = load_matrix_dir(directory)

    smarts = [line.strip() for line in open(directory / REACTION_FILE) if line.strip()]
    blocks = read_smiles_file(directory / BUILDING_BLOCK_FILE)
    matrix = np.asarray(loaded.matrix)

    for column, template_smarts in enumerate(smarts):
        reaction = AllChem.ReactionFromSmarts(template_smarts)
        rdChemReactions.ChemicalReaction.Initialize(reaction)
        queries = [
            reaction.GetReactantTemplate(i) for i in range(reaction.GetNumReactantTemplates())
        ]
        for row, smiles in enumerate(blocks):
            mol = Chem.MolFromSmiles(smiles)
            expected = 0
            for position, query in enumerate(queries):
                if mol.HasSubstructMatch(query):
                    expected |= 1 << position
            assert matrix[row, column] == expected, (smiles, template_smarts)


def test_reactant_queries_outlive_the_reaction_they_came_from() -> None:
    """A borrowed reactant template dangles once its reaction is collected, and matching segfaults."""
    import gc

    records = load_template_records(["synformer"])
    queries = [reactant_queries(record["smarts"]) for record in records]
    gc.collect()  # every ChemicalReaction built above is unreachable by now
    probe = Chem.MolFromSmiles("CCN")
    assert sum(probe.HasSubstructMatch(q) for group in queries for q in group) >= 0


def test_reactant_queries_match_a_smarts_round_trip() -> None:
    """The reactant template is used directly; a round trip through SMARTS must not change it."""
    records = load_template_records(["synformer"])
    checked = 0
    for record in records:
        reaction = AllChem.ReactionFromSmarts(record["smarts"])
        rdChemReactions.ChemicalReaction.Initialize(reaction)
        probes = [
            Chem.MolFromSmiles(s) for s in ("CCN", "CC(=O)Cl", "O=[N+]([O-])c1ccccc1", "OCCO")
        ]
        for direct, template in zip(
            reactant_queries(record["smarts"]),
            [reaction.GetReactantTemplate(i) for i in range(reaction.GetNumReactantTemplates())],
        ):
            round_tripped = Chem.MolFromSmarts(Chem.MolToSmarts(template))
            for probe in probes:
                assert probe.HasSubstructMatch(direct) == probe.HasSubstructMatch(round_tripped)
                checked += 1
    assert checked > 0


def test_salts_are_dropped_by_default_and_kept_on_request(catalog: Path, tmp_path: Path) -> None:
    dropped = build(catalog, tmp_path / "drop")
    assert dropped["building_blocks"]["multicomponent_dropped"] == 1
    assert "CCN.Cl" not in (tmp_path / "drop" / BUILDING_BLOCK_FILE).read_text()

    major = build(catalog, tmp_path / "major", multicomponent="major")
    assert major["building_blocks"]["multicomponent_dropped"] == 0
    # The salt reduces to CCN, which is already present, so it becomes a duplicate instead.
    assert major["building_blocks"]["duplicates"] > dropped["building_blocks"]["duplicates"]

    build(catalog, tmp_path / "keep", multicomponent="keep")
    assert "CCN.Cl" in (tmp_path / "keep" / BUILDING_BLOCK_FILE).read_text()


def test_atom_mapped_entries_are_dropped_by_default_and_can_be_stripped(
    catalog: Path, tmp_path: Path
) -> None:
    dropped = build(catalog, tmp_path / "drop")
    assert dropped["building_blocks"]["atom_mapped_dropped"] == 1
    assert ":7" not in (tmp_path / "drop" / BUILDING_BLOCK_FILE).read_text()

    stripped = build(catalog, tmp_path / "strip", atom_maps="strip")
    assert stripped["building_blocks"]["atom_mapped_dropped"] == 0
    text = (tmp_path / "strip" / BUILDING_BLOCK_FILE).read_text()
    assert ":7" not in text and "CCO" in text


def test_invalid_rows_and_duplicates_are_counted_not_written(catalog: Path, tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    stats = build(catalog, directory)["building_blocks"]
    assert stats["invalid"] == 1
    assert stats["duplicates"] == 1
    assert stats["rows_read"] == len(CATALOG)
    written = [line.strip() for line in open(directory / BUILDING_BLOCK_FILE)][1:]
    assert len(written) == stats["written"] == len(set(written))


def test_offsets_address_each_block_line(catalog: Path, tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    build(catalog, directory)
    path = directory / BUILDING_BLOCK_FILE
    blocks = read_smiles_file(path)
    stored = np.load(directory / OFFSET_FILE)
    assert np.array_equal(stored, line_offsets(path, skip_header=True))
    raw = path.read_bytes()
    for index, smiles in enumerate(blocks):
        assert raw[stored[index] : stored[index + 1]].decode().strip() == smiles


def test_templates_the_engine_cannot_run_are_dropped_with_a_reason(tmp_path: Path) -> None:
    records = [
        {"reaction_id": "test:ok", "library": "test", "smarts": "[C:1][OH]>>[C:1]O"},
        {
            "reaction_id": "test:four",
            "library": "test",
            "smarts": "[C:1].[C:2].[C:3].[C:4]>>[C:1][C:2][C:3][C:4]",
        },
        {"reaction_id": "test:broken", "library": "test", "smarts": "not a reaction"},
    ]
    stats = write_templates(records, tmp_path)
    assert stats["kept"] == 1 and stats["dropped"] == 2
    assert [line.strip() for line in open(tmp_path / REACTION_ID_FILE)] == ["test:ok"]
    reasons = dict(
        line.rstrip("\n").split("\t")
        for line in open(tmp_path / DROPPED_TEMPLATE_FILE).readlines()[1:]
    )
    assert "4 reactants" in reasons["test:four"]
    assert reasons["test:broken"]


def test_templates_come_from_the_installed_package(catalog: Path, tmp_path: Path) -> None:
    """A directory built here cannot cite a template the package does not have."""
    directory = tmp_path / "matrix"
    build(catalog, directory)
    known = {
        record["reaction_id"] for record in load_template_records(["synformer", "paroutes", "ord"])
    }
    built = {line.strip() for line in open(directory / REACTION_ID_FILE) if line.strip()}
    assert built and built <= known
    assert set(bundled_reaction_catalog()["sources"]) >= {"synformer"}


def test_seed_reactions_need_every_position_filled() -> None:
    # One 2-reactant template: position 0 only, then both positions across two rows.
    matrix = np.array([[0b01], [0b01]], dtype=np.uint8)
    assert compute_columns(matrix, [2])["seed_reaction_indices"].tolist() == []
    matrix = np.array([[0b01], [0b10]], dtype=np.uint8)
    assert compute_columns(matrix, [2])["seed_reaction_indices"].tolist() == [0]


def test_a_built_directory_generates_targets(catalog: Path, tmp_path: Path) -> None:
    directory = tmp_path / "matrix"
    build(catalog, directory)
    output = tmp_path / "targets.tsv"
    summary = generate_targets(
        directory, output, GenerationOptions(count=5, max_steps=1, workers=1, seed=3)
    )
    assert summary["targets"] > 0
    rows = [line.rstrip("\n").split("\t") for line in open(output) if line.strip()]
    known = {line.strip() for line in open(directory / REACTION_ID_FILE) if line.strip()}
    for _, pathway in rows:
        assert {token for token in pathway.split(";") if ":" in token and token in known}


def test_a_limit_keeps_the_block_file_aligned_with_the_matrix(
    catalog: Path, tmp_path: Path
) -> None:
    directory = tmp_path / "matrix"
    full = build(catalog, tmp_path / "full")["building_blocks"]
    meta = build(catalog, directory, limit=2)
    loaded = load_matrix_dir(directory)
    assert loaded.num_blocks == 2
    assert np.load(directory / MATRIX_FILE, mmap_mode="r").shape[0] == 2
    # The recorded counts describe the truncated file the rows align with, not the whole catalog.
    assert meta["building_blocks"]["written"] == 2
    assert meta["building_blocks"]["written_before_limit"] == full["written"]
    assert json.loads((directory / META_FILE).read_text())["building_blocks"]["written"] == 2


def test_a_limit_drops_common_reagents_before_catalog_rows(catalog: Path, tmp_path: Path) -> None:
    from_source = build(catalog, tmp_path / "plain")["building_blocks"]["written"]
    with_reagents = build(catalog, tmp_path / "reagents", common_reagents=True)["building_blocks"]
    assert with_reagents["reagents_added"] > 1

    limited = build(catalog, tmp_path / "limited", common_reagents=True, limit=from_source + 1)
    assert limited["building_blocks"]["written"] == from_source + 1
    assert limited["building_blocks"]["reagents_added"] == 1


def test_load_template_records_rejects_an_unknown_library() -> None:
    with pytest.raises(ValueError, match="unknown reaction library 'nope'"):
        load_template_records(["nope"])


def test_load_template_records_reports_a_missing_template_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("synclaw.tasks.matrix.bundled_reaction_template_dir", lambda: tmp_path)
    with pytest.raises(FileNotFoundError, match="no template file for library 'synformer'"):
        load_template_records(["synformer"])


@pytest.mark.parametrize("salt", ["[Na+].[Cl-]", "[Cl-].[Na+]", "CCO.O", "O.CC(=O)[O-].[K+]"])
def test_build_matrix_and_emolecules_keep_the_same_major_component(salt):
    from rdkit import Chem

    from synclaw.core.structures import major_component
    from synclaw.tasks.matrix import _canonicalize

    expected = Chem.MolToSmiles(major_component(Chem.MolFromSmiles(salt)))

    assert _canonicalize(([salt], "major", "drop")) == [(expected, "ok")]
