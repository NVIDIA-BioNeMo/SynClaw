# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from rdkit import Chem

from synclaw.core.chem import compare_molecular_identity, strip_isotopes


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        (
            "CCC(COC)n1nnc2c(Nc3ccc(O[11CH3])cc3C)nc(C)cc21",
            "CCC(COC)n1nnc2c(Nc3ccc(OC)cc3C)nc(C)cc21",
        ),
        (
            "CN(CCOCC[18F])C[C@H]1[C@@H](c2ccc(Cl)cc2)C[C@@H]2CC[C@H]1N2C",
            "CN(CCOCCF)C[C@H]1[C@@H](c2ccc(Cl)cc2)C[C@@H]2CC[C@H]1N2C",
        ),
        (
            "CN1CCN(C2=Nc3cc([125I])ccc3Nc3ccccc32)CC1",
            "CN1CCN(C2=Nc3cc(I)ccc3Nc3ccccc32)CC1",
        ),
    ],
    ids=["chembl-0124", "chembl-0318", "chembl-0319"],
)
def test_strip_isotopes_chembl_targets(smiles, expected):
    result = strip_isotopes(smiles)

    assert result == expected
    assert strip_isotopes(result) == result
    assert all(atom.GetIsotope() == 0 for atom in Chem.MolFromSmiles(result).GetAtoms())
    # Preprocessing is explicit; exact identity itself still distinguishes isotopes.
    identity = compare_molecular_identity(smiles, result)
    assert identity["valid"]
    assert not identity["identical"]


@pytest.mark.parametrize("smiles", ["OCC", "C(C)O", "Cl.C[C@@H](O)F", " CCO "])
def test_strip_isotopes_preserves_unlabelled_input_verbatim(smiles):
    assert strip_isotopes(smiles) == smiles


@pytest.mark.parametrize(
    ("smiles", "expected"),
    [
        ("[2H]O[3H]", "O"),
        ("[2H][C@](F)(Cl)[H]", "FCCl"),
        ("[2H][C@](F)(Cl)Br", "F[C@@H](Cl)Br"),
        ("C[C@H](F)[13CH3]", "CC(C)F"),
        ("[13CH3][C@H](O)F", "C[C@H](O)F"),
        ("[13CH3][C@@H](O)F", "C[C@@H](O)F"),
        ("[13CH3]/C=C/F", "C/C=C/F"),
        ("[13CH3]/C=C\\F", "C/C=C\\F"),
        ("[13CH3]/C(C)=C/F", "CC(C)=CF"),
        ("[2H]/C([H])=C/F", "C=CF"),
        ("[2H]/C(F)=C/Cl", "F/C=C\\Cl"),
        ("[2H]/C(F)=C\\Cl", "F/C=C/Cl"),
        ("[13CH3]C(=O)[O-].[23Na+]", "CC(=O)[O-].[Na+]"),
    ],
)
def test_strip_isotopes_preserves_nonisotopic_chemistry(smiles, expected):
    result = strip_isotopes(smiles)

    assert result == expected
    assert strip_isotopes(result) == result


@pytest.mark.parametrize("smiles", ["", " \t ", "not_smiles", "C1CC", "[13CH5]"])
def test_strip_isotopes_rejects_empty_or_invalid_smiles(smiles):
    with pytest.raises(ValueError, match="SMILES is empty or invalid"):
        strip_isotopes(smiles)
