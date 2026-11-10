# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdChemReactions

from synclaw.tasks.targets import (
    BUILDING_BLOCK_FILE,
    MATRIX_FILE,
    OFFSET_FILE,
    REACTION_FILE,
    TEMPLATE_ID_FILE,
    GenerationOptions,
    MatrixDir,
    generate_targets,
    load_matrix_dir,
    replay_route,
)

# A fixture-sized stand-in for a real matrix directory: two templates an amine and an acid chloride can
# satisfy, and blocks chosen so every reactant position has at least one match.
TEMPLATES = [
    ("test:amide", "[N;!$(N=*);!$(N-C=O):1].[C:2](=[O:3])[Cl]>>[N:1][C:2]=[O:3]"),
    ("test:nitro_reduction", "O=[N+;H0;D3:1](-[O-])-[c:2]>>[NH2;D1;+0:1]-[c:2]"),
]
BLOCKS = [
    "CCN",
    "NCc1ccccc1",
    "CC(=O)Cl",
    "O=C(Cl)c1ccccc1",
    "O=[N+]([O-])c1ccccc1",
    "O=[N+]([O-])c1ccc(C)cc1",
]


def write_matrix_dir(directory: Path, blocks: list[str] = BLOCKS) -> Path:
    """Write a matrix directory by hand, which pins the on-disk contract `tasks.matrix` writes to."""
    directory.mkdir(parents=True, exist_ok=True)

    block_path = directory / BUILDING_BLOCK_FILE
    offsets, position = [], 0
    with open(block_path, "w", newline="\n") as handle:
        position += handle.write("SMILES\n")
        for smiles in blocks:
            offsets.append(position)
            position += handle.write(smiles + "\n")
    offsets.append(position)
    np.save(directory / OFFSET_FILE, np.array(offsets, dtype=np.int64))

    (directory / REACTION_FILE).write_text("".join(f"{smarts}\n" for _, smarts in TEMPLATES))
    (directory / TEMPLATE_ID_FILE).write_text("".join(f"{name}\n" for name, _ in TEMPLATES))

    matrix = np.lib.format.open_memmap(
        directory / MATRIX_FILE,
        mode="w+",
        dtype=np.uint8,
        shape=(len(blocks), len(TEMPLATES)),
        fortran_order=True,
    )
    for column, (_, smarts) in enumerate(TEMPLATES):
        reaction = AllChem.ReactionFromSmarts(smarts)
        rdChemReactions.ChemicalReaction.Initialize(reaction)
        for row, smiles in enumerate(blocks):
            mol = Chem.MolFromSmiles(smiles)
            flag = 0
            for position_index in range(reaction.GetNumReactantTemplates()):
                if mol.HasSubstructMatch(reaction.GetReactantTemplate(position_index)):
                    flag |= 1 << position_index
            matrix[row, column] = flag
    matrix.flush()
    del matrix
    return directory


@pytest.fixture
def matrix_dir(tmp_path: Path) -> Path:
    return write_matrix_dir(tmp_path / "matrix_dir")


def test_load_matrix_dir_reads_the_documented_layout(matrix_dir: Path) -> None:
    matrix_dir = load_matrix_dir(matrix_dir)
    assert isinstance(matrix_dir, MatrixDir)
    assert matrix_dir.num_blocks == len(BLOCKS)
    assert matrix_dir.num_templates == len(TEMPLATES)
    assert matrix_dir.template_ids == [name for name, _ in TEMPLATES]


def test_load_matrix_dir_names_missing_files(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match=MATRIX_FILE):
        load_matrix_dir(tmp_path / "empty")


def test_load_matrix_dir_rejects_a_matrix_that_does_not_match_its_templates(matrix_dir: Path) -> None:
    # A matrix directory is built by a separate tool, so a stale file must fail loudly rather than
    # silently index the wrong reaction.
    (matrix_dir / REACTION_FILE).write_text(TEMPLATES[0][1] + "\n")
    with pytest.raises(ValueError, match="matrix columns"):
        load_matrix_dir(matrix_dir)


def test_generated_targets_are_canonical_and_deduplicated(matrix_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "targets.tsv"
    summary = generate_targets(
        matrix_dir, output, GenerationOptions(count=25, workers=1, seed=0, max_steps=2)
    )
    rows = [line.rstrip("\n").split("\t") for line in output.read_text().splitlines()]
    assert rows and summary["targets"] == len(rows)
    products = [row[0] for row in rows]
    assert len(set(products)) == len(products)
    for product in products:
        mol = Chem.MolFromSmiles(product)
        assert mol is not None
        assert Chem.MolToSmiles(mol) == product


def test_routes_name_real_templates_and_are_replayable(matrix_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir,
        output,
        GenerationOptions(count=25, workers=1, seed=1, max_steps=2, with_intermediates=True),
    )
    known = {name for name, _ in TEMPLATES}
    smarts_by_id = dict(TEMPLATES)
    compiled: dict = {}
    checked = 0
    for line in output.read_text().splitlines():
        product, pathway = line.split("\t")
        tokens = pathway.split(";")
        assert known & set(tokens), "a route must name at least one template"
        # With intermediates the last token is the step product, which is the target itself.
        assert tokens[-1] == product
        # Naming a template is not evidence the step runs: re-execute each one through the engine
        # and require it to produce the recorded product.
        outcome = replay_route(pathway, product, smarts_by_id, compiled=compiled)
        assert outcome["ok"], (pathway, outcome["reasons"])
        assert outcome["steps"] == outcome["steps_reproduced"] > 0
        checked += outcome["steps"]
    assert checked > 0


def test_a_seed_reproduces_the_same_targets(matrix_dir: Path, tmp_path: Path) -> None:
    first, second = tmp_path / "a.tsv", tmp_path / "b.tsv"
    options = GenerationOptions(count=15, workers=1, seed=7, max_steps=2)
    generate_targets(matrix_dir, first, options)
    generate_targets(matrix_dir, second, options)
    assert first.read_text() == second.read_text()


def test_exclude_keeps_a_second_run_disjoint(matrix_dir: Path, tmp_path: Path) -> None:
    first = tmp_path / "first.tsv"
    generate_targets(matrix_dir, first, GenerationOptions(count=10, workers=1, seed=0, max_steps=2))
    second = tmp_path / "second.tsv"
    generate_targets(
        matrix_dir,
        second,
        GenerationOptions(count=10, workers=1, seed=3, max_steps=2, exclude=str(first)),
    )
    original = {line.split("\t")[0] for line in first.read_text().splitlines()}
    follow_up = {line.split("\t")[0] for line in second.read_text().splitlines()}
    assert not (original & follow_up)


def test_library_selection_restricts_the_templates_used(matrix_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir,
        output,
        GenerationOptions(count=10, workers=1, seed=0, max_steps=1, libraries=("test",)),
    )
    assert output.read_text()
    with pytest.raises(ValueError, match="no templates belong to libraries"):
        generate_targets(
            matrix_dir,
            tmp_path / "none.tsv",
            GenerationOptions(count=5, workers=1, libraries=("absent",)),
        )


def test_summary_records_template_usage(matrix_dir: Path, tmp_path: Path) -> None:
    output = tmp_path / "targets.tsv"
    summary = generate_targets(
        matrix_dir, output, GenerationOptions(count=20, workers=1, seed=0, max_steps=2)
    )
    stats = json.loads(Path(summary["stats_path"]).read_text())
    assert stats["template_usage"]
    assert set(stats["template_usage"]) <= {name for name, _ in TEMPLATES}
    assert summary["distinct_templates_used"] == len(stats["template_usage"])


def test_molecular_weight_filter_bounds_the_output(matrix_dir: Path, tmp_path: Path) -> None:
    from rdkit.Chem import Descriptors

    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir,
        output,
        GenerationOptions(count=10, workers=1, seed=0, max_steps=2, mw_max=150.0),
    )
    for line in output.read_text().splitlines():
        assert Descriptors.MolWt(Chem.MolFromSmiles(line.split("\t")[0])) <= 150.0


def test_invalid_step_bounds_are_rejected(matrix_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="min_steps"):
        generate_targets(
            matrix_dir, tmp_path / "x.tsv", GenerationOptions(count=5, min_steps=3, max_steps=1)
        )


def test_unscoreable_targets_are_rejected_by_default(tmp_path: Path) -> None:
    """A target make-taskset can never match must not reach the file; ferrocene is the real case."""
    from synclaw.core.chem import compare_molecular_identity
    from synclaw.tasks.targets import GenerationOptions as _Options

    # The dative-bond form, as some vendor catalogs write it. The ionic form "[cH-]1cccc1.[cH-]1cccc1.[Fe+2]"
    # converts fine, so the hazard is the bond type, not the metal.
    ferrocene = "CN(C)C[c]12->[Fe+2]3456789(<-[cH]%10[cH]->3[cH]->4[cH-]->5[cH]->6%10)<-[cH]([cH]->71)[cH]->8[cH-]->92"
    assert compare_molecular_identity(ferrocene, ferrocene)["valid"] is False
    assert _Options().require_scoreable is True


def test_every_generated_target_builds_a_task(matrix_dir: Path, tmp_path: Path) -> None:
    from synclaw.tasks.datasets import build_llm_task

    output = tmp_path / "targets.tsv"
    summary = generate_targets(matrix_dir, output, GenerationOptions(count=8, workers=1, seed=5))
    assert "unscoreable_filtered" in summary  # the filter must be reportable, not silent
    for line in open(output):
        smiles = line.rstrip("\n").split("\t")[0]
        build_llm_task(smiles, task_id="t-1", dataset="t", source_file="f")


def test_routes_never_fall_below_min_steps(matrix_dir: Path, tmp_path: Path) -> None:
    """A stalled extension must discard the draw, not emit a route shorter than asked for."""
    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir, output, GenerationOptions(count=25, min_steps=2, max_steps=3, workers=1, seed=9)
    )
    template_ids = {line.strip() for line in open(matrix_dir / TEMPLATE_ID_FILE) if line.strip()}
    for line in open(output):
        pathway = line.rstrip("\n").split("\t")[1]
        assert sum(1 for token in pathway.split(";") if token in template_ids) >= 2


def test_replay_rejects_a_tampered_route(matrix_dir: Path, tmp_path: Path) -> None:
    """Guards the test above: replay must fail when the recorded product is wrong."""
    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir,
        output,
        GenerationOptions(count=5, workers=1, seed=1, max_steps=1, with_intermediates=True),
    )
    smarts_by_id = dict(TEMPLATES)
    product, pathway = output.read_text().splitlines()[0].split("\t")
    assert replay_route(pathway, product, smarts_by_id)["ok"]

    tokens = pathway.split(";")
    tokens[-1] = "CCCCCCCCO"  # a product that step cannot make
    tampered = ";".join(tokens)
    outcome = replay_route(tampered, "CCCCCCCCO", smarts_by_id)
    assert not outcome["ok"]
    assert "not reproduced" in outcome["reasons"]

    # An unconsumed token sits below the final product, so checking only the top of the stack
    # would accept this route.
    orphan = replay_route("CCCCCCCCO;" + pathway, product, smarts_by_id)
    assert not orphan["ok"]
    assert any("entries on the stack" in reason for reason in orphan["reasons"])


def test_an_extension_keeps_a_partner_equal_to_the_running_product() -> None:
    """Dropping every match loses a partner that is the same molecule as the product."""
    from synclaw.tasks.targets import _new_reactant_tokens

    product = "CCO"
    # Three reactants, one partner identical to the running product: both partners must survive.
    assert _new_reactant_tokens([product, "A", product], product, ["A", product]) == ["A", product]
    assert _new_reactant_tokens([product, product], product, [product]) == [product]
    assert _new_reactant_tokens([product, "A", "B"], product, ["A", "B"]) == ["A", "B"]
    # If a permutation left no copy of the product, fall back to the partners as drawn.
    assert _new_reactant_tokens(["A", "B"], product, ["A", "B"]) == ["A", "B"]


def test_block_sampling_starts_from_the_block_it_drew(matrix_dir: Path) -> None:
    """`--sampling block` must use the drawn block, not just let it pick a template."""
    from synclaw.tasks.targets import RouteBuilder

    loaded = load_matrix_dir(matrix_dir)
    builder = RouteBuilder(
        loaded,
        GenerationOptions(sampling="block", max_steps=1, with_intermediates=True),
        seed=3,
    )
    chosen = 0  # "CCN", an amine the fixture's amide template accepts
    builder.pools.random_block = lambda: chosen  # type: ignore[method-assign]
    drawn = builder.pools.smiles(chosen)

    for _ in range(50):
        built = builder.build()
        if built is not None:
            _, tokens = built
            assert drawn in tokens, (drawn, tokens)
            break
    else:
        raise AssertionError("no route was built")
    builder.pools.close()


def test_template_usage_counts_only_the_routes_that_were_kept(matrix_dir: Path, tmp_path: Path) -> None:
    """Shard totals include routes the merge drops as duplicates; the report must not."""
    output = tmp_path / "targets.tsv"
    summary = generate_targets(
        matrix_dir, output, GenerationOptions(count=60, workers=4, seed=2, max_steps=2)
    )
    template_ids = {line.strip() for line in open(matrix_dir / TEMPLATE_ID_FILE) if line.strip()}
    actual: dict[str, int] = {}
    for line in open(output):
        pathway = line.rstrip("\n").split("\t")[1]
        for token in pathway.split(";"):
            if token in template_ids:
                actual[token] = actual.get(token, 0) + 1
    stats = json.loads(Path(summary["stats_path"]).read_text())
    assert stats["template_usage"] == actual
    assert summary["distinct_templates_used"] == len(actual)


def test_no_pathway_output_is_one_column_but_still_reports_usage(matrix_dir: Path, tmp_path: Path) -> None:
    """Shards keep the route so usage stays countable; the output file still drops it."""
    output = tmp_path / "targets.txt"
    summary = generate_targets(
        matrix_dir, output, GenerationOptions(count=20, workers=2, seed=4, with_pathway=False)
    )
    for line in open(output):
        assert "\t" not in line
    assert summary["distinct_templates_used"] > 0


def test_output_never_exceeds_the_requested_count(matrix_dir: Path, tmp_path: Path) -> None:
    """Each worker aims for a rounded-up share, so the merge must trim to the count."""
    output = tmp_path / "targets.tsv"
    summary = generate_targets(
        matrix_dir, output, GenerationOptions(count=3, workers=4, seed=0, max_steps=2)
    )
    assert 0 < summary["targets"] <= 3
    assert len(output.read_text().splitlines()) == summary["targets"]


def test_adjacent_seeds_give_workers_distinct_streams() -> None:
    """Seed 0's worker 1 must not replay seed 1's worker 0, or a top-up run regenerates old routes."""
    import random

    from synclaw.tasks.targets import worker_seed

    assert worker_seed(0, 1) != worker_seed(1, 0)
    draws = lambda seed: [random.Random(seed).random() for _ in range(3)]  # noqa: E731
    assert draws(worker_seed(0, 1)) != draws(worker_seed(1, 0))
    assert draws(worker_seed(7, 2)) == draws(worker_seed(7, 2))


def test_build_errors_are_counted_and_reported(
    matrix_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A systematic failure must explain an empty output, not end silently with zero targets."""
    from synclaw.tasks.targets import RouteBuilder

    def broken(self: RouteBuilder) -> None:
        raise RuntimeError("bad SMARTS")

    monkeypatch.setattr(RouteBuilder, "build", broken)
    summary = generate_targets(
        matrix_dir,
        tmp_path / "targets.tsv",
        GenerationOptions(count=2, workers=1, max_attempts_per_target=3),
    )
    assert summary["targets"] == 0
    assert summary["build_errors"] == {"RuntimeError": 6}
    assert summary["first_build_error"] == "RuntimeError: bad SMARTS"
    stats = json.loads(Path(summary["stats_path"]).read_text())
    assert stats["build_errors"] == {"RuntimeError": 6}


def test_extension_skips_templates_whose_other_positions_have_no_blocks(tmp_path: Path) -> None:
    """An aniline fits the amide template's amine position, but no acid chloride is in stock."""
    from synclaw.tasks.targets import RouteBuilder

    blocks = [smiles for smiles in BLOCKS if "Cl" not in smiles]
    loaded = load_matrix_dir(write_matrix_dir(tmp_path / "no_acid_chlorides", blocks))
    builder = RouteBuilder(loaded, GenerationOptions(), seed=0)
    aniline = Chem.MolFromSmiles("Nc1ccccc1")
    assert aniline.HasSubstructMatch(builder.reaction(0).GetReactantTemplate(0))
    assert builder._fitting_template(aniline) == (None, None)
    builder.pools.close()


def test_replay_names_default_output_as_lacking_products(matrix_dir: Path, tmp_path: Path) -> None:
    """Default output reads a new reactant as the previous product; say so instead of failing steps."""
    output = tmp_path / "targets.tsv"
    generate_targets(
        matrix_dir, output, GenerationOptions(count=5, workers=1, seed=1, min_steps=2, max_steps=2)
    )
    product, pathway = output.read_text().splitlines()[0].split("\t")
    outcome = replay_route(pathway, product, dict(TEMPLATES))
    assert not outcome["ok"]
    assert outcome["reasons"] == ["a reaction has no recorded product; regenerate with --with-intermediates"]
