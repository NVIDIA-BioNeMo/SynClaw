# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider endpoints and credential lookup, loaded from data rather than code.

`providers.json` ships beside this module as package data. Each preset names an
API mode and may give its endpoints, its Responses state mode and the environment
variables that may hold its key; an endpoint left out is empty, the state mode
defaults to ``previous_id``, and no key variables means no key is needed. The
command-line flags (`--api-mode`, `--base-url`, `--responses-url`, `--messages-url`,
`--api-key-env`, `--auth-header`) override any of them, so the `custom` preset
reaches any other hosted endpoint without editing the table.

Read through `importlib.resources` rather than a `__file__` walk, so the table is
found the same way in a source checkout and an installed wheel.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

CONFIG_RESOURCE = "providers.json"

# Each API mode and the preset field holding its endpoint.
ENDPOINT_FIELD_BY_API_MODE = {
    "chat": "base_url",
    "responses": "responses_url",
    "messages": "messages_url",
}
ENDPOINT_FIELDS = tuple(ENDPOINT_FIELD_BY_API_MODE.values())
RESPONSES_STATE_MODES = ("previous_id", "full")
AUTH_HEADERS = ("bearer", "x-api-key")


@dataclass(frozen=True)
class ProviderPreset:
    """Endpoint and credential defaults for one provider."""

    api_mode: str
    base_url: str = ""
    responses_url: str = ""
    messages_url: str = ""
    responses_state_mode: str = "previous_id"
    api_key_envs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderConfig:
    """The provider table: every preset and the one used when no ``--provider`` is given."""

    default_provider: str
    providers: dict[str, ProviderPreset]

    def preset(self, provider: str) -> ProviderPreset:
        try:
            return self.providers[provider]
        except KeyError:
            raise RuntimeError(f"Unknown provider: {provider}") from None


@lru_cache(maxsize=4)
def load_provider_config(path: str | Path | None = None) -> ProviderConfig:
    """Load the provider table, from `path` when given or from package data."""

    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        raw = json.loads(
            resources.files("synclaw").joinpath(CONFIG_RESOURCE).read_text(encoding="utf-8")
        )
    providers = {name: _preset(name, entry) for name, entry in raw["providers"].items()}
    if raw["default_provider"] not in providers:
        raise RuntimeError(
            f"default_provider {raw['default_provider']!r} is not in the provider table"
        )
    return ProviderConfig(
        default_provider=raw["default_provider"],
        providers=providers,
    )


def _preset(name: str, entry: dict[str, Any]) -> ProviderPreset:
    api_mode = entry.get("api_mode")
    if api_mode not in ENDPOINT_FIELD_BY_API_MODE:
        raise RuntimeError(
            f"Provider {name!r}: api_mode must be one of {tuple(ENDPOINT_FIELD_BY_API_MODE)}, "
            f"not {api_mode!r}"
        )
    state_mode = entry.get("responses_state_mode", "previous_id")
    if state_mode not in RESPONSES_STATE_MODES:
        raise RuntimeError(
            f"Provider {name!r}: responses_state_mode must be one of {RESPONSES_STATE_MODES}, "
            f"not {state_mode!r}"
        )
    return ProviderPreset(
        api_mode=api_mode,
        responses_state_mode=state_mode,
        api_key_envs=tuple(entry.get("api_key_envs") or ()),
        **{field: entry.get(field, "") for field in ENDPOINT_FIELDS},
    )


def provider_names() -> tuple[str, ...]:
    return tuple(load_provider_config().providers)


def default_provider() -> str:
    return load_provider_config().default_provider


def resolve_api_key(
    provider: str, api_key_env: str = "", *, use_preset_keys: bool = True
) -> str | None:
    """Return the key from `--api-key-env` when given, else from the provider's own variables.

    Pass ``use_preset_keys=False`` when the request goes to an endpoint other than
    the preset's, so the preset's key is never sent to a host it was not issued for.
    """

    if api_key_env:
        return os.environ.get(api_key_env) or None
    if not use_preset_keys:
        return None
    names = load_provider_config().preset(provider).api_key_envs
    return next((value for name in names if (value := os.environ.get(name))), None)


def api_key_required(provider: str, api_key_env: str = "", *, use_preset_keys: bool = True) -> bool:
    """Whether a run must have a key: the preset names key variables, or the user named one.

    A preset without key variables (``custom``) reaches unauthenticated endpoints,
    such as a local server, when no ``--api-key-env`` is given.
    """

    if api_key_env:
        return True
    return use_preset_keys and bool(load_provider_config().preset(provider).api_key_envs)


def missing_api_key_message(provider: str, api_key_env: str = "") -> str:
    """The error for a run that needs a key and has none, naming where it was looked for."""

    if api_key_env:
        return f"API key variable {api_key_env} is not set"
    names = load_provider_config().preset(provider).api_key_envs
    return f"API key is required for provider {provider!r}; set {' or '.join(names)}, or pass --api-key-env"


def default_auth_header(api_mode: str) -> str:
    """The Messages API takes ``x-api-key``; Chat Completions and Responses take a bearer token."""

    return "x-api-key" if api_mode == "messages" else "bearer"


def auth_headers(api_key: str | None, auth_header: str) -> dict[str, str]:
    """Return the request header carrying `api_key`, or none when there is no key."""

    if not api_key:
        return {}
    if auth_header == "x-api-key":
        return {"x-api-key": api_key}
    if auth_header == "bearer":
        return {"Authorization": f"Bearer {api_key}"}
    raise ValueError(f"Unknown auth header {auth_header!r}; expected one of {AUTH_HEADERS}")
