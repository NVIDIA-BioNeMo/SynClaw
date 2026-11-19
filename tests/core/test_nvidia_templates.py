# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scope checks for the NVIDIA-authored templates in data/reaction_templates/nvidia/.

Each rule gives its expected product on a small example, and inputs outside its scope give
nothing: ring leaving atoms are not cut out of the product, and plain amides, esters and
basic amines do not react.
"""

import pytest
from rdkit import Chem

from synclaw.core.reactions import ingest_reaction_libraries
from synclaw.tools.synthesis import SynthesisTools


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    database = tmp_path_factory.mktemp("nvidia-templates") / "catalog.sqlite"
    ingest_reaction_libraries(database)
    return SynthesisTools(database)


def _canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# (template, reactants in template order, expected products; empty = must not react).
CASES = [
    ("nvidia:R000", ["CCN", "CC(=O)Cl"], {"CCNC(C)=O"}),
    ("nvidia:R000", ["CCN", "CC(=O)Nc1ccccc1"], set()),
    ("nvidia:R000", ["CCN", "O=C1CCCO1"], set()),
    ("nvidia:R000", ["CCN", "O=C1CCC(=O)O1"], set()),
    ("nvidia:R001", ["Nc1ccccc1", "BrCc1ccccc1"], {"c1ccc(CNc2ccccc2)cc1"}),
    ("nvidia:R001", ["Nc1ccccc1", "C1CCOC1"], set()),
    ("nvidia:R002", ["COC(=O)c1ccccc1"], {"OCc1ccccc1"}),
    ("nvidia:R002", ["O=C1CCCO1"], set()),
    ("nvidia:R003", ["C[C@@H](O)CC", "Oc1ccccc1"], {"CC[C@H](C)Oc1ccccc1"}),
    ("nvidia:R003", ["OCc1ccccc1", "O=C1NC(=O)c2ccccc12"], {"O=C1c2ccccc2C(=O)N1Cc1ccccc1"}),
    ("nvidia:R003", ["OCc1ccccc1", "C1COCCN1"], set()),
    ("nvidia:R003", ["CC(C)(C)O", "Oc1ccccc1"], set()),
    # Aromatic N-H and S-H pronucleophiles; only the nitrogen drawn with the H reacts.
    ("nvidia:R003", ["C[C@@H](O)CC", "c1ccc(-c2nn[nH]n2)cc1"], {"CC[C@H](C)n1nnc(-c2ccccc2)n1"}),
    ("nvidia:R003", ["OCc1ccccc1", "CC(=O)S"], {"CC(=O)SCc1ccccc1"}),
    ("nvidia:R004", ["Oc1ccccc1", "BrCc1ccccc1"], {"c1ccc(COc2ccccc2)cc1"}),
    ("nvidia:R004", ["CC(=O)O", "BrCc1ccccc1"], {"CC(=O)OCc1ccccc1"}),
    ("nvidia:R004", ["COc1ccccc1", "BrCc1ccccc1"], set()),
    # Unactivated aryl halides are accepted; the copper or palladium catalyst is not encoded.
    ("nvidia:R004", ["OC1CCCCC1", "Brc1ccccc1"], {"c1ccc(OC2CCCCC2)cc1"}),
    ("nvidia:R005", ["O=Cc1ccccc1", "CCOC(=O)CP(=O)(OCC)OCC"], {"CCOC(=O)C=Cc1ccccc1"}),
    ("nvidia:R005", ["CCOC(C)=O", "CCOC(=O)CP(=O)(OCC)OCC"], set()),
    ("nvidia:R006", ["CCCCBr", "[I-]"], {"CCCCI"}),
    ("nvidia:R007", ["CC(C)(C)OC(=O)N1CCCCC1"], {"C1CCNCC1"}),
    ("nvidia:R007", ["O=C1OCCN1c1ccccc1"], set()),
]


@pytest.mark.parametrize(("reaction_id", "reactants", "expected"), CASES)
def test_nvidia_rule_products(tools, reaction_id, reactants, expected):
    result = tools.simulate_reaction(reaction_id, reactants)
    assert set(result["generated_products"]) == {_canonical(s) for s in expected}
    assert bool(result["issues"]) == (not expected)


# Reverse simulation gives exact precursors: no radicals on the alcohol carbon or aniline nitrogen.
@pytest.mark.parametrize(
    ("reaction_id", "product", "precursors"),
    [
        ("nvidia:R003", "CC(=O)SCc1ccccc1", ("OCc1ccccc1", "CC(=O)S")),
        ("nvidia:R003", "O=C1c2ccccc2C(=O)N1Cc1ccccc1", ("OCc1ccccc1", "O=C1NC(=O)c2ccccc21")),
    ],
)
def test_reverse_precursors_are_exact(tools, reaction_id, product, precursors):
    reverse = tools.simulate_reverse_reaction(reaction_id, product, include_near=False)
    assert {tuple(c["reactants"]) for c in reverse["candidate_reactant_sets"]} == {
        tuple(_canonical(s) for s in precursors)
    }


def test_aniline_rule_recovers_a_primary_aniline_in_reverse(tools):
    reverse = tools.simulate_reverse_reaction(
        "nvidia:R001", "c1ccc(NCc2ccccc2)cc1", include_near=False
    )
    anilines = {c["reactants"][0] for c in reverse["candidate_reactant_sets"]}
    assert _canonical("Nc1ccccc1") in anilines


def test_mitsunobu_rule_inverts_the_alcohol_carbon_in_reverse(tools):
    # (S)-2-phenoxybutane comes from (R)-butan-2-ol under Mitsunobu inversion.
    reverse = tools.simulate_reverse_reaction(
        "nvidia:R003", "CC[C@H](C)Oc1ccccc1", include_near=False
    )
    assert {tuple(c["reactants"]) for c in reverse["candidate_reactant_sets"]} == {
        (_canonical("C[C@@H](O)CC"), _canonical("Oc1ccccc1"))
    }
