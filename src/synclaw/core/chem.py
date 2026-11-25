# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chemistry utilities backed by RDKit.

RDKit is a required dependency for this package. We fail at import time when it
is unavailable because fallback SMILES validation can make invalid synthesis
plans look valid.
"""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Any

try:
    from rdkit import Chem, DataStructs, RDLogger, rdBase
    from rdkit.Chem import rdChemReactions, rdFingerprintGenerator, rdinchi, rdRascalMCES
    from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, StereoEnumerationOptions
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError as exc:  # pragma: no cover - exercised only in broken environments.
    raise RuntimeError(
        "RDKit is required for BioNeMo-SynClaw chemistry validation. "
        "Install project dependencies, for example: uv sync or python -m pip install -e ."
    ) from exc

RDLogger.DisableLog("rdApp.*")

MCES_TIMEOUT_SECONDS = 20
MCES_CACHE_SIZE = 2048
MCES_MAX_BOND_MATCH_PAIRS = 10_000
MORGAN_RADIUS = 2
MORGAN_N_BITS = 2048
_CONSTITUTIONAL_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_N_BITS,
)
_STEREO_AWARE_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=MORGAN_RADIUS,
    fpSize=MORGAN_N_BITS,
    includeChirality=True,
)
RDKIT_VERSION = str(rdBase.rdkitVersion)


def validate_smiles(smiles: str | None) -> dict[str, Any]:
    """Validate and canonicalize a SMILES string with RDKit."""

    raw = "" if smiles is None else str(smiles)
    text = raw.strip()
    if not text:
        return {
            "input": raw,
            "valid": False,
            "canonical_smiles": None,
            "error": "SMILES is empty",
            "properties": {},
        }

    mol = Chem.MolFromSmiles(text)
    if mol is None:
        return {
            "input": raw,
            "valid": False,
            "canonical_smiles": None,
            "error": "RDKit could not parse SMILES",
            "properties": {},
        }

    return validate_mol(mol, raw)


def validate_mol(mol: Any, raw: str = "") -> dict[str, Any]:
    """The ``validate_smiles`` result for a molecule that has already parsed."""
    return {
        "input": raw,
        "valid": True,
        "canonical_smiles": Chem.MolToSmiles(mol, canonical=True),
        "error": None,
        "properties": {
            "heavy_atom_count": int(mol.GetNumHeavyAtoms()),
            "inchi_key": _inchi_key(mol),
        },
    }


@lru_cache(maxsize=4096)
def smarts_products_have_unmapped_atoms(products: str) -> bool | None:
    """Whether any product-side SMARTS atom lacks an atom map; None if the SMARTS does not parse."""
    try:
        reaction = rdChemReactions.ReactionFromSmarts(f">>{products}")
    except ValueError:
        return None
    return any(
        atom.GetAtomMapNum() == 0
        for template in reaction.GetProducts()
        for atom in template.GetAtoms()
    )


def strip_isotopes(smiles: str) -> str:
    """Remove isotope labels from a valid SMILES for opt-in target preprocessing.

    Isotope-free inputs are returned unchanged. Modified inputs are serialized
    as canonical isomeric SMILES, preserving remaining stereochemistry, charges
    and disconnected components. Isotopic hydrogens become ordinary hydrogens;
    stereochemistry that depended only on isotope distinctions is removed.
    Empty or invalid SMILES raise ``ValueError``. This does not change the
    isotope-sensitive behavior of :func:`matches_target`.
    """

    mol = Chem.MolFromSmiles(smiles) if smiles.strip() else None
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES is empty or invalid")
    if not any(atom.GetIsotope() for atom in mol.GetAtoms()):
        return smiles
    # Zero the isotope labels only; skipping standardization keeps charges and
    # disconnected components as written.
    rdMolStandardize.IsotopeParentInPlace(mol, skipStandardize=True)
    # Isotopic hydrogens are now ordinary hydrogens, and a centre or double bond
    # whose substituents differed only by isotope is no longer stereogenic.
    hydrogen_removal = Chem.RemoveHsParameters()
    hydrogen_removal.removeDefiningBondStereo = True
    mol = Chem.RemoveHs(mol, hydrogen_removal)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _inchi_key(mol: Any) -> str | None:
    # RDKit reports a failed conversion as an empty key, not an exception.
    return Chem.MolToInchiKey(mol) or None


def _standard_inchi(mol: Any) -> str:
    """Standard InChI of a molecule.

    Raises:
        ValueError: InChI conversion failed or did not produce a Standard InChI.
    """

    identity, status, message, _, _ = rdinchi.MolToInchi(mol)
    if status not in (0, 1) or not identity.startswith("InChI=1S/"):
        raise ValueError(message or "Standard InChI conversion failed")
    return identity


# Exact-match scoring starts from Standard InChI identity and relaxes it for stereo the
# target leaves open and for tautomers Standard InChI does not merge; see matches_target.
MAX_OPEN_STEREOISOMERS = 256
_INCHI_STEREO_LAYERS = frozenset("btms")


def _stereo_keeping_tautomer_enumerator() -> Any:
    # By default RDKit drops stereo at atoms and bonds that tautomerism touches, which
    # would let a product missing the target's stereo match after canonicalization.
    params = rdMolStandardize.CleanupParameters()
    params.tautomerRemoveSp3Stereo = False
    params.tautomerRemoveBondStereo = False
    return rdMolStandardize.TautomerEnumerator(params)


_TAUTOMERS = _stereo_keeping_tautomer_enumerator()


def _identity_mol(smiles: str | None) -> Any:
    """Parse a SMILES for identity checks, without atom maps.

    Raises:
        ValueError: The SMILES is empty, invalid or contains a wildcard atom.
    """

    text = "" if smiles is None else str(smiles).strip()
    mol = Chem.MolFromSmiles(text) if text else None
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("SMILES is empty or invalid")
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        raise ValueError("SMILES contains a wildcard atom")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return mol


def unscoreable_reason(smiles: str | None) -> str | None:
    """Why :func:`matches_target` can never compare this molecule, or None if it can."""

    try:
        _standard_inchi(_identity_mol(smiles))
    except (ValueError, RuntimeError) as exc:
        return str(exc)
    return None


def _inchi_layers(inchi: str) -> list[str]:
    return inchi.split("/")[1:]


def _without_stereo_layers(inchi: str) -> tuple[str, ...]:
    return tuple(layer for layer in _inchi_layers(inchi) if layer[:1] not in _INCHI_STEREO_LAYERS)


def _open_stereoisomers(mol: Any) -> list[Any] | None:
    """Every assignment of the stereo ``mol`` leaves open; None above MAX_OPEN_STEREOISOMERS."""

    options = StereoEnumerationOptions(
        onlyUnassigned=True, unique=True, maxIsomers=MAX_OPEN_STEREOISOMERS + 1
    )
    isomers = list(EnumerateStereoisomers(mol, options=options))
    return isomers if len(isomers) <= MAX_OPEN_STEREOISOMERS else None


def _open_stereoisomer_inchis(mol: Any) -> frozenset[str] | None:
    isomers = _open_stereoisomers(mol)
    return None if isomers is None else frozenset(_standard_inchi(iso) for iso in isomers)


def _satisfies_stereo(target: Any, target_inchi: str, product: Any, product_inchi: str) -> bool:
    """Same constitution, and every stereoisomer the product denotes is one the target allows."""

    if target_inchi == product_inchi:
        return True
    if _without_stereo_layers(target_inchi) != _without_stereo_layers(product_inchi):
        return False
    allowed = _open_stereoisomer_inchis(target)
    if allowed is None:
        return False
    denoted = _open_stereoisomer_inchis(product)
    return denoted is not None and denoted <= allowed


def matches_target(target_smiles: str | None, product_smiles: str | None) -> bool | None:
    """Whether a product counts as an exact reconstruction of the target.

    Starts from Standard InChI identity: full Standard InChI strings, including their
    isotope, stereo and net-protonation layers and InChI's normalization of supported
    tautomers. Some charge placements, such as neutral glycine and its zwitterion,
    normalize to the same identity. Atom maps and atom or component order are ignored;
    salts are not stripped and nothing is neutralized. It relaxes this in two ways:

    - Stereo the target leaves open. The product matches when every stereoisomer it
      denotes is one the target allows: a specific isomer at a centre or double bond the
      target leaves open counts, while leaving open one the target specifies does not.
      A side with more than ``MAX_OPEN_STEREOISOMERS`` open isomers is compared strictly.
    - Tautomers Standard InChI does not merge, such as keto-enol forms. Both sides are
      put in RDKit's canonical tautomer, keeping their stereo, and compared again as
      above. This applies only when the Standard InChIs differ beyond their stereo
      layers, so a product that is the target's own tautomer with wrong or missing
      stereo cannot match by having tautomer handling erase that stereo.

    The relation is not symmetric, so the target goes first. The canonical tautomer
    depends on the RDKit version. Returns None when either side cannot be compared,
    including when InChI conversion fails.
    """

    try:
        target = _identity_mol(target_smiles)
        product = _identity_mol(product_smiles)
        target_inchi = _standard_inchi(target)
        product_inchi = _standard_inchi(product)
        if _satisfies_stereo(target, target_inchi, product, product_inchi):
            return True
        if _without_stereo_layers(target_inchi) == _without_stereo_layers(product_inchi):
            return False
        # Tautomers share a formula; skip canonicalization for every other analogue.
        if _inchi_layers(target_inchi)[0] != _inchi_layers(product_inchi)[0]:
            return False
        target = _TAUTOMERS.Canonicalize(target)
        product = _TAUTOMERS.Canonicalize(product)
        return _satisfies_stereo(target, _standard_inchi(target), product, _standard_inchi(product))
    except (ValueError, RuntimeError):
        return None


def target_match_inchi_key_prefixes(target_smiles: str) -> frozenset[str]:
    """First InChIKey blocks under which a molecule that :func:`matches_target` may be stored.

    The first 14 characters of a Standard InChIKey hash the skeleton and hydrogens but
    not stereo, so one prefix per tautomer of the target covers every stereoisomer of
    it, including ones that fix only part of the stereo. Candidates found by these
    prefixes must still be checked with ``matches_target``.

    Raises:
        ValueError: ``target_smiles`` is invalid or contains a wildcard atom.
    """

    target = _identity_mol(target_smiles)
    keys = (_inchi_key(tautomer) for tautomer in [target, *_TAUTOMERS.Enumerate(target)])
    return frozenset(key[:14] for key in keys if key)


def similarity(smiles_a: str, smiles_b: str) -> dict[str, Any]:
    """Compute calibrated Morgan/Tanimoto similarity between two SMILES strings.

    The active ``similarity`` is the stereo-aware score; the constitutional score
    is reported alongside it.
    """

    left = validate_smiles(smiles_a)
    right = validate_smiles(smiles_b)
    if not left["valid"] or not right["valid"]:
        return {
            "valid": False,
            "similarity": None,
            "constitutional_similarity": None,
            "left": left,
            "right": right,
            "error": "Both SMILES must be valid to compute similarity",
        }

    mol_a = Chem.MolFromSmiles(left["canonical_smiles"])
    mol_b = Chem.MolFromSmiles(right["canonical_smiles"])
    constitutional_score = _tanimoto(_CONSTITUTIONAL_MORGAN_GENERATOR, mol_a, mol_b)
    stereo_score = _tanimoto(_STEREO_AWARE_MORGAN_GENERATOR, mol_a, mol_b)
    return {
        "valid": True,
        "similarity": stereo_score,
        "constitutional_similarity": constitutional_score,
        "left": left,
        "right": right,
        "error": None,
    }


def constitutional_morgan_fingerprint(smiles: str) -> Any:
    """Return the constitutional Morgan fingerprint used by BB near search."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for fingerprinting: {smiles!r}")
    return constitutional_morgan_fingerprint_from_mol(mol)


def constitutional_morgan_fingerprint_from_mol(mol: Any) -> Any:
    """Return the constitutional Morgan fingerprint of an already-parsed molecule."""

    return _CONSTITUTIONAL_MORGAN_GENERATOR.GetFingerprint(mol)


def stereo_aware_morgan_fingerprint(smiles: str) -> Any:
    """Return the chirality-aware Morgan fingerprint behind the stereo-aware similarity."""

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES for fingerprinting: {smiles!r}")
    return _STEREO_AWARE_MORGAN_GENERATOR.GetFingerprint(mol)


def bulk_tanimoto(query_fingerprint: Any, candidate_fingerprints: list[Any]) -> list[float]:
    """Tanimoto similarity of one fingerprint against many of the same kind."""

    return [
        float(score)
        for score in DataStructs.BulkTanimotoSimilarity(
            query_fingerprint,
            candidate_fingerprints,
        )
    ]


def mces(smiles_a: str, smiles_b: str) -> dict[str, Any]:
    """Return the full RASCAL MCES and its bond-set Tanimoto overlap."""

    left = validate_smiles(smiles_a)
    right = validate_smiles(smiles_b)
    if not left["valid"] or not right["valid"]:
        return {
            "valid": False,
            "mces_smiles": None,
            "similarity": None,
            "matched_bonds": None,
            "left_bonds": None,
            "right_bonds": None,
            "union_bonds": None,
            "timed_out": False,
            "left": left,
            "right": right,
            "error": "Both SMILES must be valid to compute an MCES",
        }

    canonical_a = str(left["canonical_smiles"])
    canonical_b = str(right["canonical_smiles"])
    first, second = sorted((canonical_a, canonical_b))
    result = dict(_cached_mces(first, second))
    if canonical_a != first:
        result["left_bonds"], result["right_bonds"] = (
            result["right_bonds"],
            result["left_bonds"],
        )
    result.update({"left": left, "right": right})
    return result


@lru_cache(maxsize=MCES_CACHE_SIZE)
def _cached_mces(canonical_a: str, canonical_b: str) -> dict[str, Any]:
    mol_a = Chem.MolFromSmiles(canonical_a)
    mol_b = Chem.MolFromSmiles(canonical_b)
    left_bonds = int(mol_a.GetNumBonds())
    right_bonds = int(mol_b.GetNumBonds())

    graph_a = _mces_graph_smiles(mol_a)
    if canonical_a == canonical_b or graph_a == _mces_graph_smiles(mol_b):
        if Chem.MolFromSmiles(graph_a) is None:
            return _mces_result(
                mces_smiles=None,
                similarity=None,
                matched_bonds=left_bonds,
                left_bonds=left_bonds,
                right_bonds=right_bonds,
                union_bonds=left_bonds,
                error="Could not convert the identical-graph MCES to SMILES",
            )
        return _mces_result(
            mces_smiles=graph_a,
            similarity=1.0,
            matched_bonds=left_bonds,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=left_bonds,
        )

    bond_match_pair_count = _bond_match_pair_count(mol_a, mol_b)
    if bond_match_pair_count > MCES_MAX_BOND_MATCH_PAIRS:
        return _mces_result(
            mces_smiles=None,
            similarity=None,
            matched_bonds=None,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=None,
            error=(
                f"RDKit RASCAL MCES requires {bond_match_pair_count} compatible bond pairs, "
                f"exceeding the safe limit of {MCES_MAX_BOND_MATCH_PAIRS}"
            ),
        )

    options = rdRascalMCES.RascalOptions()
    options.similarityThreshold = 0.0
    options.completeAromaticRings = False
    options.maxBondMatchPairs = MCES_MAX_BOND_MATCH_PAIRS
    options.timeout = MCES_TIMEOUT_SECONDS
    options.allBestMCESs = False
    options.singleLargestFrag = False
    try:
        results = rdRascalMCES.FindMCES(mol_a, mol_b, options)
    except Exception as exc:
        return _mces_result(
            mces_smiles=None,
            similarity=None,
            matched_bonds=None,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=None,
            error=f"RDKit RASCAL MCES failed: {exc}",
        )

    if not results:
        fallback_bond = _single_common_bond(mol_a, mol_b)
        if fallback_bond is None:
            return _mces_result(
                mces_smiles=None,
                similarity=0.0,
                matched_bonds=0,
                left_bonds=left_bonds,
                right_bonds=right_bonds,
                union_bonds=left_bonds + right_bonds,
            )
        bond = mol_a.GetBondWithIdx(fallback_bond)
        atom_indices = sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        union_bonds = left_bonds + right_bonds - 1
        try:
            mces_smiles = _mces_fragment_smiles(
                mol_a,
                atom_indices=atom_indices,
                bond_indices=[fallback_bond],
            )
        except Exception as exc:
            return _mces_result(
                mces_smiles=None,
                similarity=None,
                matched_bonds=1,
                left_bonds=left_bonds,
                right_bonds=right_bonds,
                union_bonds=union_bonds,
                error=f"Could not convert the one-bond MCES to SMILES: {exc}",
            )
        return _mces_result(
            mces_smiles=mces_smiles,
            similarity=float(1 / union_bonds),
            matched_bonds=1,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=union_bonds,
        )

    result = results[0]
    if bool(result.timedOut):
        return _mces_result(
            mces_smiles=None,
            similarity=None,
            matched_bonds=None,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=None,
            timed_out=True,
            error=f"RDKit RASCAL MCES exceeded the {MCES_TIMEOUT_SECONDS}-second timeout",
        )

    bond_indices = sorted({int(pair[0]) for pair in result.bondMatches()})
    atom_indices = sorted({int(pair[0]) for pair in result.atomMatches()})
    matched_bonds = len(bond_indices)
    union_bonds = left_bonds + right_bonds - matched_bonds
    if not matched_bonds:
        return _mces_result(
            mces_smiles=None,
            similarity=0.0,
            matched_bonds=0,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=union_bonds,
        )
    try:
        mces_smiles = _mces_fragment_smiles(
            mol_a,
            atom_indices=atom_indices,
            bond_indices=bond_indices,
        )
    except Exception as exc:
        return _mces_result(
            mces_smiles=None,
            similarity=None,
            matched_bonds=matched_bonds,
            left_bonds=left_bonds,
            right_bonds=right_bonds,
            union_bonds=union_bonds,
            error=f"Could not convert the RASCAL MCES to SMILES: {exc}",
        )
    score = matched_bonds / union_bonds if union_bonds > 0 else 0.0
    return _mces_result(
        mces_smiles=mces_smiles,
        similarity=float(score),
        matched_bonds=matched_bonds,
        left_bonds=left_bonds,
        right_bonds=right_bonds,
        union_bonds=union_bonds,
    )


def _mces_fragment_smiles(
    mol: Any,
    *,
    atom_indices: list[int],
    bond_indices: list[int],
) -> str:
    # Kekulizing the source first avoids invalid non-ring aromatic atoms at fragment boundaries.
    source = _mces_output_mol(mol)
    Chem.Kekulize(source, clearAromaticFlags=True)
    smiles = Chem.MolFragmentToSmiles(
        source,
        atomsToUse=atom_indices,
        bondsToUse=bond_indices,
        canonical=True,
        isomericSmiles=False,
        kekuleSmiles=True,
    )
    if not smiles or Chem.MolFromSmiles(smiles) is None:
        raise ValueError("generated fragment is not a valid SMILES string")
    return smiles


def _mces_graph_smiles(mol: Any) -> str:
    return Chem.MolToSmiles(_mces_output_mol(mol), canonical=True, isomericSmiles=False)


def _mces_output_mol(mol: Any) -> Any:
    source = Chem.Mol(mol)
    Chem.RemoveStereochemistry(source)
    for atom in source.GetAtoms():
        atom.SetAtomMapNum(0)
    return source


def _mces_result(
    *,
    mces_smiles: str | None,
    similarity: float | None,
    matched_bonds: int | None,
    left_bonds: int,
    right_bonds: int,
    union_bonds: int | None,
    timed_out: bool = False,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "valid": error is None,
        "mces_smiles": mces_smiles,
        "similarity": similarity,
        "matched_bonds": matched_bonds,
        "left_bonds": left_bonds,
        "right_bonds": right_bonds,
        "union_bonds": union_bonds,
        "timed_out": timed_out,
        "error": error,
    }


def _single_common_bond(mol_a: Any, mol_b: Any) -> int | None:
    right_labels = {_bond_match_label(bond) for bond in mol_b.GetBonds()}
    for bond_a in mol_a.GetBonds():
        if _bond_match_label(bond_a) in right_labels:
            return int(bond_a.GetIdx())
    return None


def _bond_match_pair_count(mol_a: Any, mol_b: Any) -> int:
    left_labels = Counter(_bond_match_label(bond) for bond in mol_a.GetBonds())
    right_labels = Counter(_bond_match_label(bond) for bond in mol_b.GetBonds())
    return sum(count * right_labels[label] for label, count in left_labels.items())


def _bond_match_label(bond: Any) -> tuple[Any, int, int]:
    begin_atomic_number = int(bond.GetBeginAtom().GetAtomicNum())
    end_atomic_number = int(bond.GetEndAtom().GetAtomicNum())
    lower_atomic_number, upper_atomic_number = sorted((begin_atomic_number, end_atomic_number))
    return bond.GetBondType(), lower_atomic_number, upper_atomic_number


def _tanimoto(generator: Any, mol_a: Any, mol_b: Any) -> float:
    return float(
        DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(mol_a), generator.GetFingerprint(mol_b)
        )
    )
