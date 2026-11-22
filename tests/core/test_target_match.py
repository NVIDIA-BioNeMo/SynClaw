# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-match scoring credits open target stereo and unmerged tautomers, and nothing else."""

import pytest
from rdkit import Chem
from rdkit.Chem import rdinchi

from synclaw.core import chem


@pytest.mark.parametrize(
    ("target", "product"),
    [
        # Keto-enol and enaminone forms that Standard InChI keeps apart (ChEMBL-1k rows).
        (
            "CC1=NC2=C(C(=O)CCC2)C(c2cccnc2)/C1=C(\\O)OC(C)C",
            "CC1=C(C(=O)OC(C)C)C(c2cccnc2)C2=C(CCCC2=O)N1",
        ),
        (
            "COc1ccc(OC)c(SC2=C(O)CC(c3ccccc3)CC2=O)c1",
            "COc1ccc(OC)c(SC2C(=O)CC(c3ccccc3)CC2=O)c1",
        ),
        (
            "O=C1c2c(O)c(O)c(I)c(=O)n2CCN1Cc1ccc(F)cc1",
            "O=C1c2c(O)c(=O)c(I)c(O)n2CCN1Cc1ccc(F)cc1",
        ),
        (
            "O=C(O)/C(O)=C/C(=O)c1cn(Cc2ccc(F)cc2)c2ccccc2c1=O",
            "O=C(O)C(=O)CC(=O)c1cn(Cc2ccc(F)cc2)c2ccccc2c1=O",
        ),
        ("O=C1CCCCC1", "OC1=CCCCC1"),
    ],
)
def test_tautomers_standard_inchi_keeps_apart_match(target: str, product: str) -> None:
    assert Chem.MolToInchi(Chem.MolFromSmiles(target)) != Chem.MolToInchi(
        Chem.MolFromSmiles(product)
    )
    assert chem.matches_target(target, product) is True


@pytest.mark.parametrize(
    ("target", "product"),
    [
        ("FC(Cl)Br", "F[C@H](Cl)Br"),
        ("FC(Cl)Br", "FC(Cl)Br"),
        ("FC=CF", "F/C=C/F"),
        # One centre fixed, two open; the product fixes one of the open ones.
        ("C[C@H](O)C(F)CC(Cl)C", "C[C@H](O)[C@@H](F)CC(Cl)C"),
        # Open stereo in a ChEMBL-1k target: the anomeric carbon.
        (
            "Nc1nc(O)c2ncn(C3O[C@H](COP(=O)(O)C(F)(F)F)[C@@H](O)[C@H]3O)c2n1",
            "Nc1nc(O)c2ncn([C@@H]3O[C@H](COP(=O)(O)C(F)(F)F)[C@@H](O)[C@H]3O)c2n1",
        ),
    ],
)
def test_specific_isomer_of_open_target_stereo_matches(target: str, product: str) -> None:
    assert chem.matches_target(target, product) is True


@pytest.mark.parametrize(
    ("target", "product"),
    [
        ("F[C@H](Cl)Br", "FC(Cl)Br"),  # Leaves open what the target specifies.
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"),  # Wrong configuration.
        ("F/C=C/F", "F/C=C\\F"),
        ("C[C@H](O)C(F)CC(Cl)C", "C[C@@H](O)[C@@H](F)CC(Cl)C"),
        # Same tautomer, missing stereo that tautomer handling could otherwise erase.
        (
            "COc1ccc(C(=O)NCC(=O)N/N=C/C(C)=C/c2ccccc2)cc1OC",
            "COc1ccc(C(=O)NCC(=O)NN=C/C(C)=C/c2ccccc2)cc1OC",
        ),
        (
            "N#C[C@@H]1CCCN1C(=O)[C@@H]1C[C@@H](CC(=O)Nc2ccccc2)C(=O)N1",
            "N#C[C@@H]1CCCN1C(=O)[C@@H]1CC(CC(=O)Nc2ccccc2)C(=O)N1",
        ),
        (
            "COc1ccc(C(=O)/C=C2/C(=O)Nc3ccccc32)cc1OC",
            "COc1ccc(C(=O)C=C2C(=O)Nc3ccccc32)cc1OC",
        ),
        (
            "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=C(C(=O)[C@H](O)[C@@]21C)[C@@]1(C)CCC(=O)"
            "[C@@H](C)[C@@H]1CC3=O)[C@@H](C)C(=O)O",
            "C=C(CC[C@@H](C)[C@H]1CC[C@H]2C3=C(C(=O)C(O)[C@]12C)[C@@]1(C)CCC(=O)"
            "[C@@H](C)[C@@H]1CC3=O)[C@H](C)C(=O)O",
        ),
    ],
)
def test_missing_or_wrong_stereo_does_not_match(target: str, product: str) -> None:
    assert chem.matches_target(target, product) is False


@pytest.mark.parametrize(
    ("target", "product"),
    [
        ("CCCCCCC", "CCCCCCCC"),
        ("N1CCCCC1", "N1CCCCCC1"),
        ("Oc1ccccc1N", "Oc1ccc(N)cc1"),
        ("CC(=O)O", "CC(=O)[O-]"),
        ("[Na+].CC(=O)[O-]", "CC(=O)[O-]"),
        ("[13CH3]O", "CO"),
    ],
)
def test_different_molecules_do_not_match(target: str, product: str) -> None:
    assert chem.matches_target(target, product) is False


def test_perfect_fingerprint_similarity_does_not_establish_identity() -> None:
    assert chem.similarity("CCCCCCC", "CCCCCCCC")["similarity"] == 1.0
    assert chem.matches_target("CCCCCCC", "CCCCCCCC") is False


@pytest.mark.parametrize(
    ("target", "product"),
    [
        ("CCO", "OCC"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("[CH3:7][OH:2]", "OC"),
        ("[Na+].[Cl-]", "[Cl-].[Na+]"),
        ("O=c1cccc[nH]1", "Oc1ccccn1"),
        ("NCC(=O)O", "[NH3+]CC(=O)[O-]"),  # Standard InChI ignores charge placement.
    ],
)
def test_standard_inchi_identity_still_matches(target: str, product: str) -> None:
    assert chem.matches_target(target, product) is True


def test_open_stereo_beyond_the_cap_is_compared_strictly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(chem, "MAX_OPEN_STEREOISOMERS", 1)
    assert chem.matches_target("FC(Cl)Br", "F[C@H](Cl)Br") is False
    assert chem.matches_target("FC(Cl)Br", "FC(Cl)Br") is True


@pytest.mark.parametrize("invalid", [None, "", "not-smiles", "*"])
def test_invalid_inputs_cannot_be_compared(invalid: str | None) -> None:
    assert chem.matches_target(invalid, "CCO") is None
    assert chem.matches_target("CCO", invalid) is None
    assert chem.unscoreable_reason(invalid)


def test_failed_inchi_conversion_cannot_be_compared(inchi_unavailable: None) -> None:
    assert chem.matches_target("CCO", "CCO") is None
    assert chem.unscoreable_reason("CCO") == "InChI unavailable"


def test_inchi_error_status_cannot_be_compared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rdinchi, "MolToInchi", lambda mol: ("", 2, "conversion error", "", ""))
    assert chem.matches_target("CCO", "CCO") is None
    assert chem.unscoreable_reason("CCO") == "conversion error"


def test_lookup_prefixes_cover_tautomers_and_every_stereoisomer() -> None:
    def prefix(smiles: str) -> str:
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]

    prefixes = chem.target_match_inchi_key_prefixes("CC(F)C(Cl)CC")
    for stored in ("CC(F)C(Cl)CC", "C[C@H](F)C(Cl)CC", "C[C@H](F)[C@@H](Cl)CC"):
        assert prefix(stored) in prefixes
    assert prefix("OC1=CCCCC1") in chem.target_match_inchi_key_prefixes("O=C1CCCCC1")
    with pytest.raises(ValueError, match="wildcard"):
        chem.target_match_inchi_key_prefixes("*CCO")
