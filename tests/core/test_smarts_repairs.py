# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Checks for the SMARTS rules corrected from upstream, listed in data/reaction_templates/SMARTS_REPAIRS.md.

Each repaired rule gives its expected product on a small or recorded example, and the
input its original rule wrongly accepted gives nothing. Isotope labels check atom
correspondence where the unlabelled product is unchanged by the repair.
"""

import pytest
from rdkit import Chem
from rdkit.Chem import rdChemReactions

from synclaw.core.reactions import ingest_reaction_libraries
from synclaw.tools.synthesis import SynthesisTools

ARYL_BROMIDE = "CCCc1nc(CC)c(Br)c(=O)n1Cc1ccc(-c2ccccc2C#N)cc1"
ENAMINONE = "CN(C)C=CC(=O)c1nn(-c2cccc(C)c2)ccc1=O"
ESTER = "COC(=O)COc1ccc(C(CC)(CC)c2cc(C)cs2)cc1C"
PHOSPHONIUM = "C[P+](c1ccccc1)(c1ccccc1)c1ccccc1"


@pytest.fixture(scope="module")
def tools(tmp_path_factory):
    database = tmp_path_factory.mktemp("smarts-repairs") / "catalog.sqlite"
    ingest_reaction_libraries(database)
    return SynthesisTools(database)


def _canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


# (template, reactants in template order, expected products; empty = must not react).
# R173's nitromethane case is a preservation control for its rejection case below it.
CASES = [
    ("synformer:R031", ["CC(=O)[O-]", "CCBr"], {"CCOC(C)=O"}),
    ("synformer:R035", ["Nc1ccccc1C#N", "CN"], {"CNc1ncnc2ccccc12"}),
    ("synformer:R041", ["NC(N)=S", "CN"], {"CNC(=N)N"}),
    ("synformer:R051", ["CC(=O)NN", "CC(=O)[O-]"], {"Cc1nnc(C)o1"}),
    ("synformer:R051", ["CC(=O)NN", "CC(=O)OC"], set()),
    ("synformer:R051", ["CC(=O)NN", "O=C(O)O"], set()),
    ("synformer:R062", ["CC(=O)NN", "ON=C=S"], set()),
    ("synformer:R070", ["NC(=S)N(C)C", "CC(=O)NN"], {"Cc1nnc(N(C)C)[nH]1"}),
    ("synformer:R070", ["NC(=S)N(C)O", "CC(=O)NN"], set()),
    ("synformer:R077", ["CC(=O)NN", "OC(=N)OC"], set()),
    ("synformer:R090", ["CC(=O)CCC(C)=O", "C[NH3+]"], {"Cc1ccc(C)n1C"}),
    ("synformer:R105", ["CC(=O)c1ccccc1C(=O)O", "Cc1ccc(Br)cc1"], {"CC(=O)c1ccccc1-c1ccc(C)cc1"}),
    (
        "synformer:R105",
        ["O=C(O)c1ccccc1[N+](=O)[O-]", "Brc1ccccc1"],
        {"O=[N+]([O-])c1ccccc1-c1ccccc1"},
    ),
    ("ord:R199", ["CI", "CC1(C)COC(c2ccccc2)=N1"], {"Cc1ccccc1C1=NC(C)(C)CO1"}),
    ("ord:R199", ["CCCC[Li]", "c1ccccc1"], set()),
    ("ord:R202", ["C=O", "CCNCC"], {"CCN(C)CC"}),
    ("ord:R202", ["CCN(CC)CC", "CNC"], set()),
    ("ord:R173", ["CCOc1cccc(C=O)c1", "C[N+](=O)[O-]"], {"CCOc1cccc(CC[N+](=O)[O-])c1"}),
    ("ord:R173", ["O=Cc1ccccc1", "C[n+]1ccccc1"], set()),
    ("ord:R194", [PHOSPHONIUM, "O=Cc1ccc(Cl)nc1"], {"C=Cc1ccc(Cl)nc1"}),
    ("ord:R194", ["CC(C)(C)O", "O=Cc1ccc(Cl)nc1"], set()),
    ("ord:R200", ["CC[Mg]Br", ESTER, "CC[Mg]Br"], {"CCC(O)(CC)COc1ccc(C(CC)(CC)c2cc(C)cs2)cc1C"}),
    ("ord:R200", ["CC[Mg]Br", ESTER, "C1CCOC1"], set()),
]


@pytest.mark.parametrize(("reaction_id", "reactants", "expected"), CASES)
def test_repaired_rule_products(tools, reaction_id, reactants, expected):
    result = tools.simulate_reaction(reaction_id, reactants)
    assert set(result["generated_products"]) == {_canonical(s) for s in expected}


# Exact precursor SMILES exclude radicals and dummy atoms from reverse execution.
@pytest.mark.parametrize(
    ("reaction_id", "reactants", "product"),
    [
        ("synformer:R051", ["CC(=O)NN", "CC(=O)O"], "Cc1nnc(C)o1"),
        ("synformer:R062", ["CC(=O)NN", "CN=C=S"], "CNc1nnc(C)o1"),
        ("synformer:R070", ["NC(=S)N(C)C", "CC(=O)NN"], "Cc1nnc(N(C)C)[nH]1"),
        ("synformer:R077", ["CC(=O)NN", "CC(=N)OC"], "Cc1nnc(C)[nH]1"),
    ],
)
def test_hydrogen_queries_recover_precursors(tools, reaction_id, reactants, product):
    forward = tools.simulate_reaction(reaction_id, reactants)
    assert set(forward["generated_products"]) == {_canonical(product)}
    reverse = tools.simulate_reverse_reaction(reaction_id, product, include_near=False)
    assert {tuple(c["reactants"]) for c in reverse["candidate_reactant_sets"]} == {
        tuple(_canonical(smiles) for smiles in reactants)
    }


# The labelled donor atom must reach the product (same unlabelled product before and after).
LABELS = [
    ("paroutes:PRECF08F5BBF5F", ["[13CH2]=CCOc1ccccc1"], "C=C[13CH2]c1ccccc1O"),
    (
        "ord:R053",
        [ARYL_BROMIDE, "CCCC[Sn](CCCC)(CCCC)[13CH]=C"],
        "C=[13CH]c1c(CC)nc(CCC)n(Cc2ccc(-c3ccccc3C#N)cc2)c1=O",
    ),
    (
        "ord:R065",
        [ENAMINONE, "[15NH2]Nc1ccccc1"],
        "Cc1cccc(-n2ccc(=O)c(-c3cc[15n]n3-c3ccccc3)n2)c1",
    ),
    (
        "ord:R194",
        ["[13CH3][P+](c1ccccc1)(c1ccccc1)c1ccccc1", "O=Cc1ccc(Cl)nc1"],
        "[13CH2]=Cc1ccc(Cl)nc1",
    ),
    (
        "ord:R200",
        ["CC[Mg]Br", "COC(=[18O])COc1ccc(C(CC)(CC)c2cc(C)cs2)cc1C", "CC[Mg]Br"],
        "CCC([18OH])(CC)COc1ccc(C(CC)(CC)c2cc(C)cs2)cc1C",
    ),
]


@pytest.mark.parametrize(("reaction_id", "reactants", "expected"), LABELS)
def test_repaired_rule_atom_correspondence(tools, reaction_id, reactants, expected):
    reaction = rdChemReactions.ReactionFromSmarts(
        tools.get_reaction_template(reaction_id)["template"]["smarts"]
    )
    products = set()
    for output in reaction.RunReactants(tuple(Chem.MolFromSmiles(s) for s in reactants)):
        Chem.SanitizeMol(output[0])
        products.add(Chem.MolToSmiles(output[0]))
    assert products == {_canonical(expected)}


def test_triazole_rule_matches_its_product_in_reverse(tools):
    # R065 once read [nHo+0] (holmium), so the reversed rule matched nothing.
    reactants, products = tools.get_reaction_template("synformer:R065")["template"]["smarts"].split(
        ">>"
    )
    reverse = rdChemReactions.ReactionFromSmarts(f"{products}>>{reactants}")
    assert reverse.RunReactants((Chem.MolFromSmiles("Cc1nc(-c2cn(-c3ccccc3)nn2)c(C)s1"),))
