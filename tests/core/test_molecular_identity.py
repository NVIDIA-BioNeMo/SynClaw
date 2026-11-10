# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Molecular identity must distinguish analogues that fingerprints can conflate."""

import pytest
from rdkit.Chem import rdinchi

from synclaw.core import chem

MODES = ("standard_inchi", "canonical_isomeric", "canonical_nonisomeric")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CCO", "OCC"),
        ("c1ccccc1", "C1=CC=CC=C1"),
        ("[CH3:7][OH:2]", "OC"),
        ("[Na+].[Cl-]", "[Cl-].[Na+]"),
    ],
)
def test_representation_changes_preserve_identity(mode, left, right):
    result = chem.compare_molecular_identity(left, right, mode=mode)

    assert result["valid"] and result["identical"]
    assert result["mode"] == mode
    assert result["error"] is None
    assert result["left_identity"] == result["right_identity"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("CCCCCCC", "CCCCCCCC"),  # Identical binary Morgan fingerprints.
        ("N1CCCCC1", "N1CCCCCC1"),  # Different ring sizes.
        ("Oc1ccccc1N", "Oc1ccc(N)cc1"),  # Constitutional regioisomers.
        ("CC(=O)O", "CC(=O)[O-]"),  # Protonation is part of the identity.
        ("[Na+].CC(=O)[O-]", "CC(=O)[O-]"),  # Counterions are not stripped.
    ],
)
def test_different_molecules_are_not_identical(mode, left, right):
    result = chem.compare_molecular_identity(left, right, mode=mode)

    assert result["valid"] and not result["identical"]
    assert result["left_identity"] != result["right_identity"]


def test_perfect_fingerprint_similarity_does_not_establish_identity():
    left, right = "CCCCCCC", "CCCCCCCC"
    assert chem.similarity(left, right)["similarity"] == 1.0
    assert not chem.compare_molecular_identity(left, right)["identical"]


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("F[C@H](Cl)Br", "F[C@@H](Cl)Br"),
        ("F[C@H](Cl)Br", "FC(Cl)Br"),
        ("F/C=C/F", "F/C=C\\F"),
        ("[13CH3]O", "CO"),
    ],
)
def test_only_nonisomeric_mode_discards_stereochemistry_and_isotopes(left, right):
    for mode in MODES:
        result = chem.compare_molecular_identity(left, right, mode=mode)
        assert result["valid"]
        assert result["identical"] == (mode == "canonical_nonisomeric")


def test_standard_inchi_recognizes_supported_tautomers():
    left, right = "O=c1cccc[nH]1", "Oc1ccccn1"
    result = chem.compare_molecular_identity(left, right)

    assert result["mode"] == "standard_inchi"
    assert result["valid"] and result["identical"]
    assert result["left_identity"].startswith("InChI=1S/")
    for mode in ("canonical_isomeric", "canonical_nonisomeric"):
        assert not chem.compare_molecular_identity(left, right, mode=mode)["identical"]


def test_standard_inchi_ignores_charge_placement_at_equal_net_protonation():
    left, right = "NCC(=O)O", "[NH3+]CC(=O)[O-]"
    assert chem.compare_molecular_identity(left, right)["identical"]
    assert not chem.compare_molecular_identity(left, right, mode="canonical_isomeric")["identical"]


@pytest.mark.parametrize("invalid", [None, "", " ", "not-smiles", "*"])
def test_invalid_inputs_never_compare_equal(invalid):
    # Input validation runs before the mode is consulted, so one mode suffices.
    for left, right in ((invalid, invalid), ("CCO", invalid), (invalid, "CCO")):
        result = chem.compare_molecular_identity(left, right)
        assert not result["valid"] and not result["identical"]
        assert result["error"]


@pytest.mark.parametrize(
    "conversion", [("", 0, "", "", ""), ("InChI=1S/invalid", 2, "conversion error", "", "")]
)
def test_failed_inchi_conversion_does_not_fall_back_to_other_identity(monkeypatch, conversion):
    monkeypatch.setattr(rdinchi, "MolToInchi", lambda mol: conversion)
    result = chem.compare_molecular_identity("CCO", "CCO")

    assert not result["valid"] and not result["identical"]
    assert result["error"]


def test_inchi_unavailable_is_reported_as_invalid(monkeypatch):
    def unavailable(mol):
        raise RuntimeError("InChI support unavailable")

    monkeypatch.setattr(rdinchi, "MolToInchi", unavailable)
    result = chem.compare_molecular_identity("CCO", "CCO")

    assert not result["valid"] and not result["identical"]
    assert "InChI support unavailable" in result["error"]


def test_inchi_normalization_warnings_are_visible():
    result = chem.compare_molecular_identity("[NH4+]", "[NH4+]")

    assert result["valid"] and result["identical"]
    assert result["warnings"]


def test_unknown_mode_is_a_configuration_error():
    with pytest.raises(ValueError, match="Unknown molecular identity mode"):
        chem.compare_molecular_identity("CCO", "CCO", mode="fingerprint")
