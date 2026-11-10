# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest import mock

from rdkit import Chem

from synclaw.core import chem


class ChemTests(unittest.TestCase):
    def test_validate_smiles_returns_canonical_and_properties(self):
        result = chem.validate_smiles("CCO")

        self.assertTrue(result["valid"])
        self.assertTrue(result["canonical_smiles"])
        self.assertEqual(result["properties"]["heavy_atom_count"], 3)

    def test_validate_smiles_rejects_empty_input(self):
        result = chem.validate_smiles("")

        self.assertFalse(result["valid"])
        self.assertIn("empty", result["error"])

    def test_validate_smiles_rejects_plain_words(self):
        result = chem.validate_smiles("product")

        self.assertEqual(result["backend"], "rdkit")
        self.assertFalse(result["valid"])

    def test_unmapped_product_atoms_include_unbracketed_atoms(self):
        self.assertTrue(chem.smarts_products_have_unmapped_atoms("[#6:1]S[#6:2]"))
        self.assertTrue(chem.smarts_products_have_unmapped_atoms("[N:1][C](=[O])[N:2]"))
        self.assertFalse(chem.smarts_products_have_unmapped_atoms("[c:2]:[cH;D2;+0:1]:[c:3]"))
        self.assertIsNone(chem.smarts_products_have_unmapped_atoms("[C:1"))

    def test_similarity_exact_match_is_one(self):
        result = chem.similarity("CCO", "CCO")

        self.assertTrue(result["valid"])
        self.assertEqual(result["similarity"], 1.0)
        self.assertEqual(result["constitutional_similarity"], 1.0)
        self.assertEqual(result["stereo_aware_similarity"], 1.0)
        self.assertEqual(result["method"], "rdkit_morgan_tanimoto")

    def test_similarity_reports_constitutional_and_stereo_aware_scores(self):
        result = chem.similarity("F[C@H](Cl)Br", "F[C@@H](Cl)Br")

        self.assertTrue(result["valid"])
        self.assertEqual(result["constitutional_similarity"], 1.0)
        self.assertLess(result["stereo_aware_similarity"], 1.0)
        self.assertEqual(result["similarity"], result["stereo_aware_similarity"])

    def test_mces_returns_bond_tanimoto_and_parseable_fragment(self):
        result = chem.mces("CCO", "CCN")

        self.assertTrue(result["valid"])
        self.assertEqual(result["matched_bonds"], 1)
        self.assertEqual(result["union_bonds"], 3)
        self.assertAlmostEqual(result["similarity"], 1 / 3)
        self.assertIsNotNone(Chem.MolFromSmiles(result["mces_smiles"]))
        self.assertEqual(result["method"], chem.MCES_METHOD)

    def test_mces_returns_full_disconnected_match(self):
        result = chem.mces("CCOc1ccccc1", "CCNc1ccccc1")

        self.assertTrue(result["valid"])
        self.assertIn(".", result["mces_smiles"])
        self.assertIsNotNone(Chem.MolFromSmiles(result["mces_smiles"]))

    def test_mces_includes_partial_aromatic_ring_matches(self):
        result = chem.mces("c1ccccc1", "c1ccncc1")

        self.assertTrue(result["valid"])
        self.assertEqual(result["matched_bonds"], 4)
        self.assertEqual(result["union_bonds"], 8)
        self.assertEqual(result["similarity"], 0.5)
        self.assertIsNotNone(Chem.MolFromSmiles(result["mces_smiles"]))

    def test_mces_handles_more_than_rascal_default_bond_pair_limit(self):
        result = chem.mces("C" * 40, "C" * 41)

        self.assertTrue(result["valid"])
        self.assertEqual(result["matched_bonds"], 39)
        self.assertEqual(result["union_bonds"], 40)
        self.assertEqual(result["similarity"], 39 / 40)

    def test_mces_rejects_bond_pairs_above_safe_limit_without_fallback(self):
        with mock.patch.object(chem.rdRascalMCES, "FindMCES") as find_mces:
            result = chem.mces("C" * 101, "C" * 102)

        find_mces.assert_not_called()
        self.assertFalse(result["valid"])
        self.assertIsNone(result["matched_bonds"])
        self.assertIsNone(result["similarity"])
        self.assertIn("exceeding the safe limit", result["error"])

    def test_mces_graph_equivalent_inputs_bypass_safe_pair_cap(self):
        chem._cached_mces.cache_clear()
        self.addCleanup(chem._cached_mces.cache_clear)
        cases = (
            (f"F[C@H](Cl){'C' * 103}", f"F[C@@H](Cl){'C' * 103}"),
            (f"[CH3:1]{'C' * 101}[CH3:2]", f"[CH3:7]{'C' * 101}[CH3:8]"),
        )
        for left, right in cases:
            with self.subTest(left=left[:20], right=right[:20]):
                mol_left = Chem.MolFromSmiles(left)
                mol_right = Chem.MolFromSmiles(right)
                self.assertGreater(
                    chem._bond_match_pair_count(mol_left, mol_right),
                    chem.MCES_MAX_BOND_MATCH_PAIRS,
                )
                chem._cached_mces.cache_clear()
                with mock.patch.object(chem.rdRascalMCES, "FindMCES") as find_mces:
                    result = chem.mces(left, right)

                find_mces.assert_not_called()
                self.assertTrue(result["valid"])
                self.assertEqual(result["similarity"], 1.0)
                self.assertEqual(result["matched_bonds"], result["union_bonds"])

    def test_mces_cached_result_preserves_argument_order_for_bond_counts(self):
        forward = chem.mces("CCCO", "CC")
        reverse = chem.mces("CC", "CCCO")

        self.assertEqual((forward["left_bonds"], forward["right_bonds"]), (3, 1))
        self.assertEqual((reverse["left_bonds"], reverse["right_bonds"]), (1, 3))
        self.assertEqual(forward["similarity"], reverse["similarity"])

    def test_mces_kekulizes_partial_aromatic_endpoints_as_valid_smiles(self):
        result = chem.mces("CCc1ccccc1", "CCc1ncccc1")

        self.assertTrue(result["valid"])
        self.assertEqual(result["matched_bonds"], 6)
        self.assertEqual(result["similarity"], 0.6)
        self.assertIsNotNone(Chem.MolFromSmiles(result["mces_smiles"]))

    def test_mces_smiles_drops_input_stereochemistry(self):
        for left, right in (
            ("C[C@H](O)F", "C[C@@H](O)F"),
            ("F/C=C/F", "F/C=C\\F"),
        ):
            with self.subTest(left=left, right=right):
                result = chem.mces(left, right)

                self.assertTrue(result["valid"])
                self.assertEqual(result["similarity"], 1.0)
                self.assertNotIn("@", result["mces_smiles"])
                self.assertNotIn("/", result["mces_smiles"])
                self.assertNotIn("\\", result["mces_smiles"])

    def test_mces_smiles_drops_atom_maps(self):
        for left, right in (
            ("[CH3:1][OH:2]", "[CH3:1][OH:2]"),
            ("[CH3:1][OH:2]", "[CH3:7][OH:8]"),
        ):
            with self.subTest(left=left, right=right):
                result = chem.mces(left, right)
                fragment = Chem.MolFromSmiles(result["mces_smiles"])

                self.assertTrue(result["valid"])
                self.assertEqual(result["similarity"], 1.0)
                self.assertIsNotNone(fragment)
                self.assertTrue(all(atom.GetAtomMapNum() == 0 for atom in fragment.GetAtoms()))

    def test_mces_exact_edge_free_molecule_is_one(self):
        result = chem.mces("[Na+]", "[Na+]")

        self.assertTrue(result["valid"])
        self.assertEqual(result["similarity"], 1.0)
        self.assertEqual(result["matched_bonds"], 0)
        self.assertEqual(result["union_bonds"], 0)

    def test_mces_valid_pair_with_no_common_edge_is_zero(self):
        result = chem.mces("CC", "NN")

        self.assertTrue(result["valid"])
        self.assertEqual(result["similarity"], 0.0)
        self.assertIsNone(result["mces_smiles"])

    def test_mces_rejects_invalid_smiles(self):
        result = chem.mces("not-smiles", "CC")

        self.assertFalse(result["valid"])
        self.assertIsNone(result["similarity"])
        self.assertIn("valid", result["error"])

    def test_mces_rejects_timed_out_partial_result(self):
        class TimedOutResult:
            timedOut = True

        chem._cached_mces.cache_clear()
        self.addCleanup(chem._cached_mces.cache_clear)
        with mock.patch.object(chem.rdRascalMCES, "FindMCES", return_value=[TimedOutResult()]):
            result = chem.mces("CCC", "CCN")

        self.assertFalse(result["valid"])
        self.assertTrue(result["timed_out"])
        self.assertIsNone(result["similarity"])
        self.assertEqual(result["timeout_seconds"], 20)
        self.assertIn("20-second", result["error"])


if __name__ == "__main__":
    unittest.main()
