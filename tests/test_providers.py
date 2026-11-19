# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The provider table is data, so a deployment can swap it without code changes."""

from __future__ import annotations

import argparse
import json

import pytest

from synclaw import providers

REPLACEMENT_TABLE = {
    "default_provider": "example",
    "providers": {
        "example": {
            "api_mode": "responses",
            "base_url": "https://example.test/v1",
            "responses_url": "https://example.test/v1/responses",
            "messages_url": "",
            "responses_state_mode": "previous_id",
            "api_key_envs": ["EXAMPLE_API_KEY"],
        }
    },
}

KEY_ENVS = (
    "NVIDIA_API_KEY",
    "NVBUILD_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture
def replacement_config(tmp_path):
    path = tmp_path / "providers.json"
    path.write_text(json.dumps(REPLACEMENT_TABLE), encoding="utf-8")
    providers.load_provider_config.cache_clear()
    yield path
    providers.load_provider_config.cache_clear()


@pytest.fixture
def no_keys(monkeypatch):
    for name in KEY_ENVS:
        monkeypatch.delenv(name, raising=False)


def test_shipped_table_is_loadable_and_self_consistent() -> None:
    config = providers.load_provider_config()
    assert config.default_provider in config.providers
    for name, preset in config.providers.items():
        assert preset.api_mode in {"chat", "responses", "messages"}, name
        assert preset.responses_state_mode in {"previous_id", "full"}, name
        for field in providers.ENDPOINT_FIELDS:
            value = str(getattr(preset, field))
            # Empty means the user supplies it with the matching flag.
            assert value == "" or value.startswith("https://"), (name, field)


def test_shipped_presets() -> None:
    config = providers.load_provider_config()
    assert config.default_provider == "nvbuild"
    expected = {
        "nvbuild": ("chat", "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
        "openai": ("responses", "https://api.openai.com/v1/responses", "OPENAI_API_KEY"),
        "anthropic": ("messages", "https://api.anthropic.com/v1/messages", "ANTHROPIC_API_KEY"),
        "openrouter": ("chat", "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    }
    field_for_mode = {"chat": "base_url", "responses": "responses_url", "messages": "messages_url"}
    for name, (api_mode, endpoint, key_env) in expected.items():
        preset = config.preset(name)
        assert preset.api_mode == api_mode, name
        assert getattr(preset, field_for_mode[api_mode]) == endpoint, name
        assert preset.api_key_envs[0] == key_env, name

    custom = config.preset("custom")
    assert custom.api_mode == "chat"
    assert custom.api_key_envs == ()
    assert all(getattr(custom, field) == "" for field in providers.ENDPOINT_FIELDS)


def test_a_replacement_table_changes_providers_without_code_changes(replacement_config) -> None:
    config = providers.load_provider_config(replacement_config)
    assert set(config.providers) == {"example"}
    assert config.default_provider == "example"


def test_unknown_provider_names_itself_in_the_error() -> None:
    with pytest.raises(RuntimeError, match="nope"):
        providers.load_provider_config().preset("nope")


def test_each_provider_reads_only_its_own_variables(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia")

    assert providers.resolve_api_key("nvbuild") == "nvidia"
    # No cross-provider fallback: a missing key is "no credential", never somebody else's.
    for provider in ("openai", "anthropic", "openrouter", "custom"):
        assert providers.resolve_api_key(provider) is None, provider

    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter")
    monkeypatch.setenv("OPENAI_API_KEY", "openai")
    assert providers.resolve_api_key("anthropic") == "anthropic"
    assert providers.resolve_api_key("openrouter") == "openrouter"
    assert providers.resolve_api_key("openai") == "openai"


def test_nvbuild_accepts_the_alternate_variable(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("NVBUILD_API_KEY", "nvb")
    assert providers.resolve_api_key("nvbuild") == "nvb"


def test_an_explicit_env_name_wins_over_the_table(monkeypatch, no_keys) -> None:
    monkeypatch.setenv("MY_KEY", "explicit")
    monkeypatch.setenv("OPENAI_API_KEY", "table")
    assert providers.resolve_api_key("openai", "MY_KEY") == "explicit"
    assert providers.resolve_api_key("custom", "MY_KEY") == "explicit"
    # A named variable that is unset does not fall back to the table.
    assert providers.resolve_api_key("openai", "UNSET_KEY_FOR_TEST") is None


def test_missing_key_message_names_the_provider_variables() -> None:
    assert "OPENROUTER_API_KEY" in providers.missing_api_key_message("openrouter")
    assert "--api-key-env" in providers.missing_api_key_message("openrouter")
    assert (
        providers.missing_api_key_message("custom", "MY_KEY")
        == "API key variable MY_KEY is not set"
    )


def test_only_a_named_key_variable_makes_the_key_required() -> None:
    assert providers.api_key_required("openai")
    assert providers.api_key_required("custom", "MY_KEY")
    # custom names no variables, so it can reach an unauthenticated local server.
    assert not providers.api_key_required("custom")


def test_auth_headers_follow_the_scheme_and_omit_a_missing_key() -> None:
    assert providers.auth_headers("k", "bearer") == {"Authorization": "Bearer k"}
    assert providers.auth_headers("k", "x-api-key") == {"x-api-key": "k"}
    assert providers.auth_headers(None, "bearer") == {}
    assert providers.default_auth_header("messages") == "x-api-key"
    assert providers.default_auth_header("chat") == "bearer"
    with pytest.raises(ValueError, match="Unknown auth header"):
        providers.auth_headers("k", "basic")


@pytest.mark.parametrize(
    ("provider", "api_mode", "flag"),
    [
        ("custom", None, "--base-url"),
        ("custom", "responses", "--responses-url"),
        ("nvbuild", "messages", "--messages-url"),
        ("anthropic", "chat", "--base-url"),
    ],
)
def test_missing_endpoint_fails_before_any_request(provider, api_mode, flag) -> None:
    from synclaw.runners.model_runner import apply_provider_defaults, require_endpoint

    args = argparse.Namespace(
        provider=provider,
        model="example-model",
        api_mode=api_mode,
        base_url=None,
        responses_url=None,
        messages_url=None,
        responses_state_mode=None,
        reasoning_effort=None,
    )
    apply_provider_defaults(args)
    with pytest.raises(RuntimeError) as excinfo:
        require_endpoint(args)
    message = str(excinfo.value)
    assert message.endswith(f"pass {flag}")
    assert "set " not in message


@pytest.mark.parametrize(
    ("argv_overrides", "expected"),
    [
        ({}, True),
        ({"base_url": "https://integrate.api.nvidia.com/v1"}, True),
        ({"base_url": "https://other.example.test/v1"}, False),
        # An override of an endpoint the API mode does not use changes nothing.
        ({"messages_url": "https://other.example.test/v1/messages"}, True),
    ],
)
def test_preset_keys_follow_only_the_preset_endpoint(
    monkeypatch, no_keys, argv_overrides, expected
) -> None:
    from synclaw.runners.model_runner import apply_provider_defaults, uses_preset_endpoint

    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia")
    args = argparse.Namespace(
        provider="nvbuild",
        model="example-model",
        api_mode=None,
        base_url=None,
        responses_url=None,
        messages_url=None,
        responses_state_mode=None,
        reasoning_effort=None,
    )
    for name, value in argv_overrides.items():
        setattr(args, name, value)
    apply_provider_defaults(args)

    use_preset_keys = uses_preset_endpoint(args)
    assert use_preset_keys is expected
    key = providers.resolve_api_key("nvbuild", use_preset_keys=use_preset_keys)
    assert key == ("nvidia" if expected else None)
    # Elsewhere the key is optional, and --api-key-env still supplies one.
    assert providers.api_key_required("nvbuild", use_preset_keys=use_preset_keys) is expected
    monkeypatch.setenv("OTHER_KEY", "other")
    assert (
        providers.resolve_api_key("nvbuild", "OTHER_KEY", use_preset_keys=use_preset_keys)
        == "other"
    )


class _FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return None

    def read(self):
        return b'{"id":"msg_1","content":[],"usage":{"input_tokens":1,"output_tokens":1}}'


def _sent_headers(api_key, auth_header):
    from unittest.mock import patch

    from synclaw.runners.model_runner import anthropic_completion

    captured = {}

    def fake_urlopen(request, timeout):
        captured.update({key.lower(): value for key, value in request.header_items()})
        return _FakeResponse()

    with patch("urllib.request.urlopen", side_effect=fake_urlopen):
        anthropic_completion(
            messages_url="https://gateway.example.test/v1/messages",
            api_key=api_key,
            model="m",
            system="s",
            messages=[{"role": "user", "content": "q"}],
            max_tokens=16,
            temperature=None,
            top_p=None,
            reasoning_effort=None,
            retries=0,
            request_timeout=5.0,
            tool_choice=None,
            tools=[],
            auth_header=auth_header,
        )
    return captured


def test_messages_gateway_can_take_a_bearer_token() -> None:
    headers = _sent_headers("k", "bearer")
    assert headers["authorization"] == "Bearer k"
    assert "x-api-key" not in headers


def test_a_keyless_request_sends_no_credential_header() -> None:
    headers = _sent_headers(None, None)
    assert "authorization" not in headers
    assert "x-api-key" not in headers
