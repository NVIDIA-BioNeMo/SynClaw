# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider endpoints and credential lookup, loaded from data rather than code.

`providers.json` ships beside this module as package data. Each preset names an
API mode, its endpoints and the environment variables that may hold its key; the
command-line flags (`--api-mode`, `--base-url`, `--responses-url`,
`--messages-url`, `--api-key-env`, `--auth-header`) override any of them, so the `custom` preset
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

CONFIG_RESOURCE = "providers.json"

ENDPOINT_FIELDS = ("base_url", "responses_url", "messages_url")
AUTH_HEADERS = ("bearer", "x-api-key")


@dataclass(frozen=True)
class ProviderPreset:
    """Endpoint and credential defaults for one provider."""

    api_mode: str
    base_url: str
    responses_url: str
    messages_url: str
    responses_state_mode: str
    api_key_envs: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProviderConfig:
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
    providers = {
        name: ProviderPreset(
            api_mode=entry["api_mode"],
            base_url=entry["base_url"],
            responses_url=entry["responses_url"],
            messages_url=entry["messages_url"],
            responses_state_mode=entry["responses_state_mode"],
            api_key_envs=tuple(entry.get("api_key_envs") or ()),
        )
        for name, entry in raw["providers"].items()
    }
    if raw["default_provider"] not in providers:
        raise RuntimeError(
            f"default_provider {raw['default_provider']!r} is not in the provider table"
        )
    return ProviderConfig(
        default_provider=raw["default_provider"],
        providers=providers,
    )


def provider_names() -> tuple[str, ...]:
    return tuple(load_provider_config().providers)


def default_provider() -> str:
    return load_provider_config().default_provider


def first_env(names: tuple[str, ...] | list[str]) -> str | None:
    """Return the value of the first set, non-empty environment variable."""

    for name in names:
        if name and (value := os.environ.get(name)):
            return value
    return None


def resolve_api_key(provider: str, api_key_env: str = "", *, use_preset_keys: bool = True) -> str | None:
    """Return the key from `--api-key-env` when given, else from the provider's own variables.

    Pass ``use_preset_keys=False`` when the request goes to an endpoint other than
    the preset's, so the preset's key is never sent to a host it was not issued for.
    """

    if api_key_env:
        return os.environ.get(api_key_env) or None
    if not use_preset_keys:
        return None
    return first_env(load_provider_config().preset(provider).api_key_envs)


def api_key_required(provider: str, api_key_env: str = "", *, use_preset_keys: bool = True) -> bool:
    """Whether a run must have a key: the preset names key variables, or the user named one.

    A preset without key variables (``custom``) reaches unauthenticated endpoints,
    such as a local server, when no ``--api-key-env`` is given.
    """

    if api_key_env:
        return True
    return use_preset_keys and bool(load_provider_config().preset(provider).api_key_envs)


def missing_api_key_message(provider: str, api_key_env: str = "") -> str:
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
