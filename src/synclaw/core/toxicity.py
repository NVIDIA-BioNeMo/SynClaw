# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Optional similarity screen against ReaSyn's public watchlist or a user-supplied file.

The screen flags whole molecules whose Tanimoto similarity to a watchlist entry
exceeds the configured threshold. It uses SynClaw's 2,048-bit constitutional Morgan
fingerprints. ReaSyn uses 4,096 bits. Coverage is limited to listed compounds
and their close structural analogues.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path
from typing import Any, Final

from rdkit import Chem, DataStructs

from synclaw.core import chem

MODE_ON: Final = "on"
MODE_OFF: Final = "off"
MODES: Final = (MODE_ON, MODE_OFF)
DEFAULT_THRESHOLD: Final = 0.8
DEFAULT_WATCHLIST: Final = Path("var/safety/mols_to_filter.txt")
ENV_MODE: Final = "SYNCLAW_TOXICITY_SCREEN"
ENV_WATCHLIST: Final = "SYNCLAW_TOXICITY_WATCHLIST"
ENV_THRESHOLD: Final = "SYNCLAW_TOXICITY_THRESHOLD"
BLOCK_KEY: Final = "toxicity_screen"


class WatchlistUnavailable(RuntimeError):
    """The enabled screen could not read any valid molecules from its watchlist."""


class ToxicityConfigError(ValueError, RuntimeError):
    """A screen setting is invalid."""


@dataclass(frozen=True)
class ToxicityConfig:
    mode: str = MODE_OFF
    watchlist: Path = DEFAULT_WATCHLIST
    threshold: float = DEFAULT_THRESHOLD

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ToxicityConfigError(
                f"toxicity screen mode must be one of {MODES}, got {self.mode!r}"
            )
        if not 0.0 <= self.threshold <= 1.0:
            raise ToxicityConfigError("toxicity screen threshold must be between 0 and 1")

    @classmethod
    def resolve(
        cls, mode: str | None = None, watchlist: str | Path | None = None
    ) -> ToxicityConfig:
        """Build the config from explicit values, then the environment, then defaults."""

        raw_threshold = os.environ.get(ENV_THRESHOLD) or DEFAULT_THRESHOLD
        try:
            threshold = float(raw_threshold)
        except ValueError:
            raise ToxicityConfigError(
                f"{ENV_THRESHOLD} must be a number, got {raw_threshold!r}"
            ) from None
        return cls(
            mode=(mode or os.environ.get(ENV_MODE) or MODE_OFF).strip().lower(),
            watchlist=Path(watchlist or os.environ.get(ENV_WATCHLIST) or DEFAULT_WATCHLIST),
            threshold=threshold,
        )


class ToxicityScreen:
    """Compare SMILES against the watchlist loaded at construction."""

    def __init__(self, config: ToxicityConfig | None = None) -> None:
        self.config = config or ToxicityConfig.resolve()
        fingerprints = _load_fingerprints(self.config.watchlist) if self.enabled else ()
        self._score = lru_cache(maxsize=4096)(partial(_best_match, fingerprints))

    @property
    def enabled(self) -> bool:
        return self.config.mode == MODE_ON

    def check(self, molecules: Iterable[tuple[str, str | None]]) -> dict[str, Any] | None:
        """Report screened molecules and matches; return ``None`` if disabled or empty.

        Like ReaSyn, compare each complete SMILES without separately screening
        disconnected fragments. Invalid or empty SMILES are left to the tool's
        own validation.
        """

        if not self.enabled:
            return None
        screened: list[str] = []
        refused: list[dict[str, Any]] = []
        for role, smiles in molecules:
            smiles = smiles.strip() if isinstance(smiles, str) else None
            if not smiles:
                continue
            match = self._score(smiles)
            if match is None:
                continue
            similarity, index = match
            screened.append(smiles)
            if similarity > self.config.threshold:
                refused.append(
                    {
                        "role": role,
                        "smiles": smiles,
                        "similarity": round(similarity, 3),
                        "watchlist_index": index,
                    }
                )
        if not screened:
            return None
        return {
            "mode": self.config.mode,
            "threshold": self.config.threshold,
            "screened": screened,
            "refused": refused,
        }

    def check_or_refuse(
        self, tool: str, molecules: Iterable[tuple[str, str | None]]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Screen ``molecules``; return ``(block, refusal)`` where ``refusal`` is the tool result if flagged."""

        block = self.check(molecules)
        return block, refused_result(tool, block) if block and block["refused"] else None

    def screen_target(self, smiles: str) -> dict[str, Any] | None:
        """Return a target-screening trace event when screening is enabled."""

        block = self.check([("target", smiles)])
        return target_screen_event(block, smiles) if block else None


def refused_result(tool: str, block: dict[str, Any]) -> dict[str, Any]:
    """Build the tool result returned when the screen flags a molecule."""

    message = refusal_message(block)
    return {
        "valid": False,
        "refused": True,
        "tool": tool,
        "issues": [message],
        "error_message": message,
        BLOCK_KEY: block,
    }


def add_toxicity_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--toxicity-screen",
        choices=MODES,
        default=None,
        help=(
            f"Watchlist similarity screen (default: ${ENV_MODE} or off). "
            "Enable explicitly with 'on'; the mode is recorded in run summaries."
        ),
    )
    parser.add_argument(
        "--toxicity-watchlist",
        default=None,
        help=f"Watchlist SMILES file (default: ${ENV_WATCHLIST} or {DEFAULT_WATCHLIST})",
    )


def config_from_args(args: argparse.Namespace) -> ToxicityConfig:
    return ToxicityConfig.resolve(
        getattr(args, "toxicity_screen", None), getattr(args, "toxicity_watchlist", None)
    )


def target_screen_event(block: dict[str, Any], target_smiles: str) -> dict[str, Any]:
    """Record the target check, including a reason when it prevents planning."""

    return {
        "event_type": "target_refused" if block["refused"] else "target_screened",
        "target_smiles": target_smiles,
        "payload": {
            BLOCK_KEY: block,
            **({"error_message": refusal_message(block)} if block["refused"] else {}),
        },
    }


def refusal_message(block: dict[str, Any]) -> str:
    items = "; ".join(f"{item['role']} {item['smiles']}" for item in block["refused"])
    return (
        f"Refused by the dangerous-molecule screen ({items}; Tanimoto similarity to the watchlist exceeds "
        f"{block['threshold']}). This system does not plan routes to or through such molecules."
    )


def toxicity_summary_fields(events: Iterable[dict[str, Any]], mode: str | None) -> dict[str, Any]:
    """Summarise screening from trace events for run summaries and result CSVs.

    ``mode`` is the configured mode; the counts come from the screen blocks that
    tool results and target-screening events carry.
    """

    screened: set[str] = set()
    refused: dict[str, None] = {}
    for event in events:
        payload = event.get("payload") or {}
        if event.get("event_type") == "tool_result":
            payload = payload.get("result") or {}
        block = payload.get(BLOCK_KEY) if isinstance(payload, dict) else None
        if not isinstance(block, dict):
            continue
        mode = mode or block.get("mode")
        screened.update(block.get("screened") or [])
        refused.update(dict.fromkeys(item["smiles"] for item in block.get("refused") or []))
    return {
        "toxicity_screen": mode or "unrecorded",
        "toxicity_screened_count": len(screened),
        "toxicity_refused_count": len(refused),
        "toxicity_refused_smiles": ";".join(refused),
    }


def _fingerprint(smiles: str) -> DataStructs.ExplicitBitVect | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return chem.constitutional_morgan_fingerprint_from_mol(mol)


def _load_fingerprints(path: Path) -> tuple[DataStructs.ExplicitBitVect, ...]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise WatchlistUnavailable(
            f"Cannot read toxicity watchlist {path}: {exc}. "
            "Use --toxicity-watchlist PATH or fetch the default with `synclaw download-public-data`."
        ) from exc
    candidates = (_fingerprint(line) for line in map(str.strip, lines) if line)
    fingerprints = tuple(fp for fp in candidates if fp is not None)
    if not fingerprints:
        raise WatchlistUnavailable(f"Toxicity watchlist {path} holds no valid SMILES")
    return fingerprints


def _best_match(
    fingerprints: tuple[DataStructs.ExplicitBitVect, ...], smiles: str
) -> tuple[float, int] | None:
    fingerprint = _fingerprint(smiles)
    if fingerprint is None:
        return None
    scores = DataStructs.BulkTanimotoSimilarity(fingerprint, fingerprints)
    best = max(scores)
    return best, scores.index(best)
