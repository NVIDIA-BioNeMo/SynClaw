# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Forward-synthesis generation of synthesis targets from a reactant/reaction matrix.

`tasks/datasets.py` turns a list of target SMILES into LLM tasks, but the list itself has to
come from somewhere. Running an external generator over the same catalogs is expensive and
produces routes this package cannot necessarily reproduce. This module generates targets
directly: pick a reaction template, pick building blocks that fit its reactant positions, run
the reaction through `core.reactions`, and repeat on the product. Because every step is executed
by the same engine that backs `simulate_reaction`, each route replays exactly through the tools
an agent is given.

One product is carried forward per step. Keeping the full product set and crossing it with the
next one -- as a stack-based enumerator does -- is combinatorially unbounded on the broader ORD
templates: a 100k run was measured at 202 GB resident before it stalled. Templates are sampled
uniformly rather than in proportion to how many blocks match them, so a run covers the library
instead of the handful of promiscuous rules.

## The reactant-template matrix directory

Choosing blocks needs a question `search_building_blocks` cannot answer: *which* blocks fit
reactant position 2 of a given template. Exact and Tanimoto search cannot express it, so this
module reads a prebuilt matrix directory, written by `tasks.matrix` (`synclaw build-matrix`) from
this package's own templates plus a building-block catalog. Its layout is:

    matrix.npy              uint8 [blocks, templates], Fortran-ordered, entry [i, j] a bitmask
                            of which reactant positions of template j block i can fill
    building_blocks.txt     'SMILES' header, then one canonical isomeric SMILES per row
    building_blocks.off.npy int64 byte offsets of each SMILES line (header excluded)
    reactions.txt           one reaction SMARTS per line, aligned with the matrix columns
    reaction_ids.txt        the reaction id for each line, aligned with the matrix columns

Fortran order matters: a column is read whole for every template, and in C order that strides
across every page of a multi-gigabyte file. `load_matrix_dir` validates the shapes against each
other and reports what is inconsistent, since a matrix outlives the run that built it;
`scripts/verify_generated_routes.py` replays a generated file to catch the rest.

Blocks are memory-mapped and parsed on demand, and pools of qualifying blocks are materialized
only when small (`pool_cache_max`), with denser pools rejection-sampled instead, so worker
memory does not scale with catalog size.
"""

from __future__ import annotations

import json
import mmap
import multiprocessing as mp
import os
import random
import sys
from collections.abc import Container, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from synclaw.core.chem import compare_molecular_identity
from synclaw.core.reactions import _rdkit_reaction_modules, _run_reaction_with_permutations

MATRIX_FILE = "matrix.npy"
BUILDING_BLOCK_FILE = "building_blocks.txt"
OFFSET_FILE = "building_blocks.off.npy"
REACTION_FILE = "reactions.txt"
REACTION_ID_FILE = "reaction_ids.txt"
REQUIRED_FILES = (MATRIX_FILE, BUILDING_BLOCK_FILE, REACTION_FILE, REACTION_ID_FILE)

MAX_REACTANTS = 3
DEFAULT_POOL_CACHE_MAX = 200_000


@dataclass
class GenerationOptions:
    """Everything `generate_targets` needs that is not the matrix directory or the output path."""

    count: int = 100_000
    workers: int = 8
    seed: int = 0
    min_steps: int = 1
    max_steps: int = 5
    max_num_atoms: int = 80
    sampling: str = "template"
    libraries: tuple[str, ...] = ()
    with_pathway: bool = True
    with_intermediates: bool = False
    mw_min: float | None = None
    mw_max: float | None = None
    sa_min: float | None = None
    sa_max: float | None = None
    require_scoreable: bool = True
    exclude: str | None = None
    pool_cache_max: int = DEFAULT_POOL_CACHE_MAX
    max_attempts_per_target: int = 50


@dataclass
class MatrixDir:
    """A validated reactant-template matrix directory.

    `matrix` stays memory-mapped; nothing here is copied.
    """

    directory: Path
    matrix: Any
    offsets: Any
    reaction_smarts: list[str] = field(default_factory=list)
    reaction_ids: list[str] = field(default_factory=list)

    @property
    def num_blocks(self) -> int:
        return int(self.matrix.shape[0])

    @property
    def num_templates(self) -> int:
        return int(self.matrix.shape[1])


def load_matrix_dir(directory: str | Path) -> MatrixDir:
    """Load and validate a matrix directory, naming every inconsistency it finds."""
    directory = Path(directory)
    missing = [name for name in REQUIRED_FILES if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{directory} is not a reactant-template matrix directory; "
            f"missing {', '.join(missing)}"
        )

    matrix = np.load(directory / MATRIX_FILE, mmap_mode="r")
    if matrix.ndim != 2 or matrix.dtype != np.uint8:
        raise ValueError(f"{MATRIX_FILE} must be a 2-D uint8 array, got {matrix.ndim}-D {matrix.dtype}")

    with open(directory / REACTION_FILE) as handle:
        reaction_smarts = [line.strip() for line in handle if line.strip()]
    with open(directory / REACTION_ID_FILE) as handle:
        reaction_ids = [line.strip() for line in handle if line.strip()]

    offset_path = directory / OFFSET_FILE
    offsets = np.load(offset_path, mmap_mode="r") if offset_path.is_file() else line_offsets(
        directory / BUILDING_BLOCK_FILE
    )

    problems = []
    if matrix.shape[1] != len(reaction_smarts):
        problems.append(f"{matrix.shape[1]} matrix columns vs {len(reaction_smarts)} reaction SMARTS")
    if len(reaction_ids) != len(reaction_smarts):
        problems.append(f"{len(reaction_ids)} reaction ids vs {len(reaction_smarts)} reaction SMARTS")
    if matrix.shape[0] != len(offsets) - 1:
        problems.append(f"{matrix.shape[0]} matrix rows vs {len(offsets) - 1} building blocks")
    if problems:
        raise ValueError(f"inconsistent matrix directory at {directory}: " + "; ".join(problems))

    return MatrixDir(
        directory=directory,
        matrix=matrix,
        offsets=offsets,
        reaction_smarts=reaction_smarts,
        reaction_ids=reaction_ids,
    )


def read_smiles_lines(path: Path) -> list[str]:
    """Read a 'SMILES'-headed building-block file into a list of SMILES."""
    lines = [line.strip() for line in open(path)]
    if lines and lines[0].upper() == "SMILES":
        lines = lines[1:]
    return [line for line in lines if line]


def line_offsets(path: Path, skip_header: bool) -> np.ndarray:
    """Byte offsets of every data line in `path`, as int64 [num_lines + 1] start/end pairs.

    The trailing entry is the end of the last line, so line `i` is `buf[off[i]:off[i + 1]]`.
    `tasks.matrix` writes the offsets file with this and `load_matrix_dir` rebuilds a missing one
    with it, so the two cannot disagree about which row a block is.
    """
    with open(path, "rb") as handle:
        buf = handle.read()
    ends = np.flatnonzero(np.frombuffer(buf, dtype=np.uint8) == 0x0A)
    if len(ends) == 0:
        raise ValueError(f"{path} has no newline-terminated lines")
    if not buf.endswith(b"\n"):  # count a final unterminated line
        ends = np.append(ends, len(buf))
    starts = np.concatenate(([0], ends[:-1] + 1))
    offsets = np.append(starts, ends[-1] + 1).astype(np.int64)
    return offsets[1:] if skip_header else offsets


class BuildingBlockPools:
    """Which blocks fit a template's reactant position, and a uniform draw from them.

    The matrix is Fortran-ordered, so a column is a contiguous memory-mapped view that costs
    nothing to take. Small pools are materialized as int32 index arrays and cached; denser ones
    are rejection-sampled against the column, which bounds memory regardless of how promiscuous
    a template is.
    """

    def __init__(self, matrix_dir: MatrixDir, pool_cache_max: int, rng: random.Random) -> None:
        self._matrix_dir = matrix_dir
        self._cache_max = pool_cache_max
        self._rng = rng
        self._handle = open(matrix_dir.directory / BUILDING_BLOCK_FILE, "rb")
        self._mmap = mmap.mmap(self._handle.fileno(), 0, access=mmap.ACCESS_READ)
        self._pools: dict[tuple[int, int], Any] = {}
        self._counts: dict[tuple[int, int], int] = {}
        self._columns: dict[int, Any] = {}

    def close(self) -> None:
        self._mmap.close()
        self._handle.close()

    def smiles(self, index: int) -> str:
        offsets = self._matrix_dir.offsets
        return self._mmap[offsets[index] : offsets[index + 1]].decode("ascii").strip()

    def column(self, template: int) -> Any:
        if template not in self._columns:
            self._columns[template] = self._matrix_dir.matrix[:, template]
        return self._columns[template]

    def count(self, template: int, position: int) -> int:
        """How many blocks fit this reactant position, cached without materializing the pool."""
        key = (template, position)
        if key not in self._counts:
            column = np.asarray(self.column(template))
            self._counts[key] = int(np.count_nonzero(column & (1 << position)))
        return self._counts[key]

    def pool(self, template: int, position: int) -> Any:
        """A cached index array for a small pool, or None when it should be rejection-sampled."""
        key = (template, position)
        if key not in self._pools:
            hits = self.count(template, position)
            if hits == 0:
                self._pools[key] = np.empty(0, dtype=np.int32)
            elif hits <= self._cache_max:
                column = np.asarray(self.column(template))
                self._pools[key] = np.flatnonzero(column & (1 << position)).astype(np.int32)
            else:
                self._pools[key] = None
        return self._pools[key]

    def sample(self, template: int, position: int) -> int | None:
        """A uniformly random block index fitting this reactant position, or None if none fits."""
        pool = self.pool(template, position)
        if pool is not None:
            return int(pool[self._rng.randrange(len(pool))]) if len(pool) else None
        column = self.column(template)
        bit = 1 << position
        for _ in range(4096):
            index = self._rng.randrange(self._matrix_dir.num_blocks)
            if column[index] & bit:
                return index
        return None

    def random_block(self) -> int:
        return self._rng.randrange(self._matrix_dir.num_blocks)


class RouteBuilder:
    """Grows one forward route: choose a template, choose blocks, execute, repeat."""

    def __init__(self, matrix_dir: MatrixDir, options: GenerationOptions, seed: int | str) -> None:
        self.matrix_dir = matrix_dir
        self.options = options
        self.rng = random.Random(seed)
        self.pools = BuildingBlockPools(matrix_dir, options.pool_cache_max, self.rng)
        self.Chem, self.AllChem, self.rdChemReactions = _rdkit_reaction_modules()
        self._compiled: dict[int, Any] = {}
        self.usable = self._select_templates()

    def _select_templates(self) -> list[int]:
        indices = range(self.matrix_dir.num_templates)
        if self.options.libraries:
            wanted = set(self.options.libraries)
            indices = [
                i for i in indices if self.matrix_dir.reaction_ids[i].split(":", 1)[0] in wanted
            ]
            if not indices:
                raise ValueError(f"no templates belong to libraries {sorted(wanted)}")
        usable = [i for i in indices if self.reaction(i).GetNumReactantTemplates() <= MAX_REACTANTS]
        if not usable:
            raise ValueError("no usable templates: all have more reactants than the engine runs")
        return usable

    def reaction(self, template: int) -> Any:
        if template not in self._compiled:
            reaction = self.AllChem.ReactionFromSmarts(self.matrix_dir.reaction_smarts[template])
            self.rdChemReactions.ChemicalReaction.Initialize(reaction)
            self._compiled[template] = reaction
        return self._compiled[template]

    def _execute(self, template: int, reactants: list[str]) -> tuple[str | None, list[str] | None]:
        """Run one step; return the product and the reactants in the order that produced it."""
        mols = [self.Chem.MolFromSmiles(smiles) for smiles in reactants]
        if any(mol is None for mol in mols):
            return None, None
        generated, _, productive_orders, _, _ = _run_reaction_with_permutations(self.reaction(template), mols, self.Chem)
        if not generated:
            return None, None
        order = productive_orders[0] if productive_orders else list(range(1, len(mols) + 1))
        return self.rng.choice(generated), [reactants[index - 1] for index in order]

    def _fitting_template(self, product_mol: Any) -> tuple[int | None, int | None]:
        """A uniformly random template the product can run through, by the first hit in a shuffle.

        Walking a shuffled list and stopping at the first match draws uniformly among the
        fitting templates, so staying unbiased costs no full scan of the library. A template fits
        only when the product matches one position and every other position has a block to fill
        it; matching alone would let the step stall at partner sampling, which cuts routes short.
        """
        order = self.usable[:]
        self.rng.shuffle(order)
        for template in order:
            reaction = self.reaction(template)
            count = reaction.GetNumReactantTemplates()
            positions = list(range(count))
            self.rng.shuffle(positions)
            for position in positions:
                if not product_mol.HasSubstructMatch(reaction.GetReactantTemplate(position)):
                    continue
                if all(self.pools.count(template, other) for other in range(count) if other != position):
                    return template, position
        return None, None

    def _seed(self) -> tuple[int, int | None, int | None]:
        """The template to start from and, under `block` sampling, the block that chose it.

        The drawn block is carried back with the position it fits, so the first step actually
        starts from it. Returning only the template would let `build` redraw every reactant,
        which samples templates by how many blocks they match -- the opposite of what the
        option asks for.
        """
        if self.options.sampling == "block":
            index = self.pools.random_block()
            mol = self.Chem.MolFromSmiles(self.pools.smiles(index))
            if mol is not None:
                template, position = self._fitting_template(mol)
                if template is not None:
                    return template, position, index
        return self.rng.choice(self.usable), None, None

    def build(self) -> tuple[str, list[str]] | None:
        """One route as (product, bottom-up tokens), or None if the draw did not produce one."""
        steps = self.rng.randint(self.options.min_steps, self.options.max_steps)
        template, seed_position, seed_index = self._seed()
        reaction = self.reaction(template)

        indices = [
            seed_index if position == seed_position else self.pools.sample(template, position)
            for position in range(reaction.GetNumReactantTemplates())
        ]
        if any(index is None for index in indices):
            return None
        product, ordered = self._execute(template, [self.pools.smiles(i) for i in indices])
        if product is None:
            return None

        tokens = list(ordered) + [self.matrix_dir.reaction_ids[template]]
        if self.options.with_intermediates:
            tokens.append(product)

        completed = 1
        for _ in range(steps - 1):
            product_mol = self.Chem.MolFromSmiles(product)
            if product_mol is None or product_mol.GetNumAtoms() > self.options.max_num_atoms:
                break
            template, position = self._fitting_template(product_mol)
            if template is None:
                break
            reaction = self.reaction(template)
            partners = []
            for other in range(reaction.GetNumReactantTemplates()):
                if other == position:
                    continue
                index = self.pools.sample(template, other)
                if index is None:
                    partners = None
                    break
                partners.append(self.pools.smiles(index))
            if partners is None:
                break
            next_product, ordered = self._execute(template, [product] + partners)
            if next_product is None:
                break
            # The product is already on the stack; only its partners are new tokens.
            tokens.extend(_new_reactant_tokens(ordered, product, partners))
            tokens.append(self.matrix_dir.reaction_ids[template])
            product = next_product
            completed += 1
            if self.options.with_intermediates:
                tokens.append(product)

        # Every break above leaves a route shorter than the draw asked for: no template fits the
        # product, no block fills a partner position, the reaction yields nothing, or the atom cap
        # is hit. Returning it anyway would put routes below `min_steps` in the output, so the draw
        # is discarded and the caller tries again.
        if completed < self.options.min_steps:
            return None
        return product, tokens


def _new_reactant_tokens(ordered: Sequence[str], product: str, partners: Sequence[str]) -> list[str]:
    """The reactants of an extension step that are not already on the stack.

    The running product occupies one reactant position and is on the stack, so exactly one
    occurrence of it is dropped. Removing every equal string instead would lose a partner that
    happens to be the same molecule -- possible with three reactants, where the survivors alone
    cannot tell how many were consumed -- and replay would then pop the wrong reactants.
    """
    remaining = list(ordered)
    if product in remaining:
        remaining.remove(product)
        return remaining
    return list(partners)


NO_PRODUCTS_REASON = "a reaction has no recorded product; regenerate with --with-intermediates"


def route_records_products(pathway: str, reaction_ids: Container[str]) -> bool:
    """Whether a route was written with `--with-intermediates`, judged by its last token.

    With intermediates every reaction is followed by its product, so a route ends on a SMILES;
    without them it ends on the final reaction id.
    """
    return pathway.rsplit(";", 1)[-1] not in reaction_ids


def replay_route(
    pathway: str,
    product: str,
    smarts_by_id: Mapping[str, str],
    *,
    compiled: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Re-execute one route's steps and report what the reaction engine reproduces.

    `tasks.targets` owns the token grammar -- reactant SMILES and reaction ids interleaved
    bottom-up, each reaction followed by the product recorded for that step -- so the interpreter
    for it lives here too, and both the verifier script and the tests read the same one. Steps run
    through `_run_reaction_with_permutations`, the function behind `simulate_reaction`; a
    step passes when the recorded product is among the ones the engine generates.

    Only `--with-intermediates` output carries those products. Default output ends on a template
    id instead, and is rejected up front: read under this grammar, its next reactant would pass
    for the previous step's product and every later step would fail for that reason alone.

    `smarts_by_id` maps a reaction id to its SMARTS. Pass `compiled` to reuse initialized
    reactions across routes. Returns `ok`, `steps`, `steps_reproduced`, the reasons a route fell
    short, and for each unreproduced step the template, its reactants and the expected product,
    so a caller can diagnose further without re-walking the route.
    """
    Chem, AllChem, rdChemReactions = _rdkit_reaction_modules()
    compiled = {} if compiled is None else compiled

    def reaction_for(reaction_id: str) -> Any:
        if reaction_id not in compiled:
            reaction = AllChem.ReactionFromSmarts(smarts_by_id[reaction_id])
            rdChemReactions.ChemicalReaction.Initialize(reaction)
            compiled[reaction_id] = reaction
        return compiled[reaction_id]

    tokens = pathway.split(";")
    if not route_records_products(pathway, smarts_by_id):
        return {"ok": False, "steps": 0, "steps_reproduced": 0, "reasons": [NO_PRODUCTS_REASON], "failures": []}

    steps = steps_ok = 0
    reasons: list[str] = []
    failures: list[dict[str, Any]] = []
    stack: list[Any] = []
    ok = True
    for token in tokens:
        if token in smarts_by_id:
            count = reaction_for(token).GetNumReactantTemplates()
            if len(stack) < count:
                reasons.append("stack underflow")
                ok = False
                break
            reactants = stack[-count:]
            del stack[-count:]
            stack.append(("pending", token, reactants))
        elif stack and isinstance(stack[-1], tuple):
            _, reaction_id, reactants = stack[-1]
            steps += 1
            mols = [Chem.MolFromSmiles(smiles) for smiles in reactants]
            if any(mol is None for mol in mols):
                reasons.append("unparseable reactant")
                ok = False
                break
            generated, *_ = _run_reaction_with_permutations(reaction_for(reaction_id), mols, Chem)
            if token in generated:
                steps_ok += 1
            else:
                reasons.append("not reproduced")
                failures.append({"reaction_id": reaction_id, "reactants": list(reactants), "expected": token})
                ok = False
            stack[-1] = token
        else:
            stack.append(token)
    # A well-formed route reduces to exactly one entry. Checking only the top would accept a
    # token that was never consumed, since the orphan sits below the final product.
    if ok:
        if any(isinstance(entry, tuple) for entry in stack):
            reasons.append(NO_PRODUCTS_REASON)
            ok = False
        elif len(stack) != 1:
            reasons.append(f"route left {len(stack)} entries on the stack, expected 1")
            ok = False
        elif stack[0] != product:
            reasons.append("final product does not match the target column")
            ok = False
    return {
        "ok": ok,
        "steps": steps,
        "steps_reproduced": steps_ok,
        "reasons": reasons,
        "failures": failures,
    }


def _load_sascorer() -> Any:
    """RDKit's own Contrib SA_Score, which ships with the wheel but is not importable by name."""
    import importlib.util

    from rdkit.Chem import RDConfig

    path = Path(RDConfig.RDContribDir) / "SA_Score" / "sascorer.py"
    spec = importlib.util.spec_from_file_location("sascorer", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"RDKit Contrib SA_Score not found at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _progress(message: str) -> None:
    """Progress goes to stderr; stdout stays a single JSON document for callers to parse."""
    print(message, file=sys.stderr, flush=True)


def worker_seed(seed: int, worker_id: int) -> str:
    """One worker's RNG seed. Adding the id instead would give seed 0's worker 1 the same stream
    as seed 1's worker 0, so a top-up run with the next seed would mostly regenerate old routes.
    """
    return f"{seed}:{worker_id}"


def _worker(task: tuple[int, str, GenerationOptions, int, str]) -> dict[str, Any]:
    worker_id, directory, options, quota, out_path = task
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")

    matrix_dir = load_matrix_dir(directory)
    builder = RouteBuilder(matrix_dir, options, seed=worker_seed(options.seed, worker_id))
    Chem = builder.Chem
    sascorer = _load_sascorer() if (options.sa_min is not None or options.sa_max is not None) else None

    mw_low = options.mw_min if options.mw_min is not None else float("-inf")
    mw_high = options.mw_max if options.mw_max is not None else float("inf")
    sa_low = options.sa_min if options.sa_min is not None else float("-inf")
    sa_high = options.sa_max if options.sa_max is not None else float("inf")

    excluded: set[str] = set()
    if options.exclude:
        with open(options.exclude) as handle:
            excluded = {line.split("\t", 1)[0].strip() for line in handle if line.strip()}

    seen: set[str] = set()
    attempts = mw_filtered = sa_filtered = unscoreable_filtered = 0
    build_errors: dict[str, int] = {}
    first_build_error: str | None = None
    limit = options.max_attempts_per_target * quota

    with open(out_path, "w") as handle:
        while len(seen) < quota and attempts < limit:
            attempts += 1
            try:
                built = builder.build()
            except Exception as exc:  # noqa: BLE001 - one bad draw must not end the shard
                # Counted, so that a systematic failure -- a bad SMARTS, a truncated matrix --
                # shows up in the summary instead of as an unexplained empty output.
                name = type(exc).__name__
                build_errors[name] = build_errors.get(name, 0) + 1
                if first_build_error is None:
                    first_build_error = f"{name}: {exc}"
                continue
            if built is None:
                continue
            product, tokens = built
            if not product or product in seen or product in excluded:
                continue
            mol = Chem.MolFromSmiles(product)
            if mol is None:
                continue
            if options.mw_min is not None or options.mw_max is not None:
                if not mw_low <= Descriptors.MolWt(mol) <= mw_high:
                    mw_filtered += 1
                    continue
            if sascorer is not None and not sa_low <= sascorer.calculateScore(mol) <= sa_high:
                sa_filtered += 1
                continue
            # A target whose standard InChI cannot be computed can never be matched exactly, so
            # `tasks.datasets` refuses it. Dative-bonded organometallics such as ferrocenes are
            # the usual source. Rejecting it here keeps a target file and a task set the same
            # length instead of letting the mismatch surface downstream.
            if options.require_scoreable and not compare_molecular_identity(product, product)["valid"]:
                unscoreable_filtered += 1
                continue
            seen.add(product)
            # Shards always carry the route, even under `--no-pathway`: the merge drops routes
            # that duplicate another shard's, and it can only count template usage over the ones
            # it keeps if every shard line still says which templates it used.
            handle.write(product + "\t" + ";".join(tokens) + "\n")

    builder.pools.close()
    return {
        "shard": out_path,
        "written": len(seen),
        "attempts": attempts,
        "mw_filtered": mw_filtered,
        "sa_filtered": sa_filtered,
        "unscoreable_filtered": unscoreable_filtered,
        "build_errors": build_errors,
        "first_build_error": first_build_error,
        "exhausted": len(seen) < quota,
    }


def generate_targets(
    matrix_dir: str | Path,
    output: str | Path,
    options: GenerationOptions,
) -> dict[str, Any]:
    """Generate targets into `output` (TSV) and return a summary with a usage histogram.

    Each worker writes a shard; shards are merged with a global de-duplication on the product,
    so the final count can fall slightly short of `options.count`. It never exceeds it: workers
    each aim for a rounded-up share, and the merge stops once the count is reached. Pass `exclude` and run again
    to top up a set without overlap.
    """
    matrix_dir = load_matrix_dir(matrix_dir)
    if options.min_steps < 1 or options.max_steps < options.min_steps:
        raise ValueError("min_steps must be >= 1 and max_steps >= min_steps")
    if options.count < 1:
        raise ValueError("count must be positive")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    workers = max(1, options.workers)
    quota = -(-options.count // workers)
    tasks = [
        (index, str(matrix_dir.directory), options, quota, f"{output}.shard{index}")
        for index in range(workers)
    ]

    _progress(
        f"generating ~{options.count} targets with {workers} workers "
        f"(steps {options.min_steps}-{options.max_steps}, sampling {options.sampling})"
    )
    context = mp.get_context("spawn")
    if workers == 1:
        results = [_worker(tasks[0])]
    else:
        with context.Pool(processes=workers) as pool:
            results = pool.map(_worker, tasks)

    seen: set[str] = set()
    written = attempts = mw_filtered = sa_filtered = unscoreable_filtered = 0
    exhausted = 0
    build_errors: dict[str, int] = {}
    first_build_error: str | None = None
    template_usage: dict[str, int] = {}
    reaction_id_set = set(matrix_dir.reaction_ids)
    with open(output, "w") as out:
        for result in results:
            attempts += result["attempts"]
            mw_filtered += result["mw_filtered"]
            sa_filtered += result["sa_filtered"]
            unscoreable_filtered += result["unscoreable_filtered"]
            exhausted += int(result["exhausted"])
            for name, number in result["build_errors"].items():
                build_errors[name] = build_errors.get(name, 0) + number
            first_build_error = first_build_error or result["first_build_error"]
            with open(result["shard"]) as shard:
                for line in shard:
                    if written >= options.count:
                        break
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    product, _, pathway = line.partition("\t")
                    if product in seen:
                        continue
                    seen.add(product)
                    # Counted here, not in the worker: a shard's own totals include routes this
                    # merge is about to drop as duplicates of another shard's.
                    for token in pathway.split(";"):
                        if token in reaction_id_set:
                            template_usage[token] = template_usage.get(token, 0) + 1
                    out.write((line if options.with_pathway else product) + "\n")
                    written += 1
            os.remove(result["shard"])

    stats_path = Path(f"{output}.stats.json")
    summary = {
        "output": str(output),
        "targets": written,
        "requested": options.count,
        "attempts": attempts,
        "yield": round(written / attempts, 4) if attempts else None,
        "distinct_templates_used": len(template_usage),
        "templates_available": len(matrix_dir.reaction_ids),
        "mw_filtered": mw_filtered,
        "sa_filtered": sa_filtered,
        "unscoreable_filtered": unscoreable_filtered,
        "shards_exhausted": exhausted,
        "build_errors": build_errors,
        "first_build_error": first_build_error,
        "stats_path": str(stats_path),
    }
    stats_path.write_text(json.dumps({**summary, "template_usage": template_usage}, indent=2) + "\n")
    if build_errors:
        _progress(f"{sum(build_errors.values())} draws raised {build_errors}; first: {first_build_error}")
    _progress(f"wrote {written} targets to {output}")
    return summary
