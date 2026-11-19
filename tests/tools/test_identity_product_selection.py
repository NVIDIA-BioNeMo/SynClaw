# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact products must survive candidate selection before the floor is checked."""

import pytest

from synclaw.core import chem
from synclaw.tools.synthesis import score_pathway_against_target


@pytest.mark.parametrize(
    ("target", "exact", "preferred"),
    [
        ("CCCCCCC", "CCCCCCC", "CCCCCCCC"),
        ("O=c1cccc[nH]1", "Oc1ccccn1", "O=c1ccccn1C"),
    ],
)
def test_exact_generated_product_outranks_preferred_analogue(target, exact, preferred):
    result = score_pathway_against_target(
        [preferred, exact],
        target_smiles=target,
        target_validation=chem.validate_smiles(target),
        preferred_smiles=preferred,
        accepted=True,
    )
    assert result["actual_final_smiles"] == exact
    assert result["identical_to_target"] is True
