"""The profile's two behaviours, without Hermes: reasoning values per model and the model list."""

from __future__ import annotations

import pytest
from conftest import CATALOG, CATALOG_IDS, StubProfile, plugin, registered_profiles

GPT_OSS = "openai/gpt-oss-120b"
QWEN = "qwen/qwen3.8-27b"


@pytest.mark.parametrize("config, expected", [
    (None, None),                                   # unset: Groq's default
    ({"enabled": False}, "low"),                    # gpt-oss cannot turn reasoning off; its floor
    ({"enabled": True, "effort": "none"}, "low"),
    ({"enabled": True, "effort": "minimal"}, "low"),
    ({"enabled": True, "effort": "low"}, "low"),
    ({"enabled": True, "effort": "medium"}, "medium"),
    ({"enabled": True, "effort": "high"}, "high"),
    ({"enabled": True, "effort": "xhigh"}, "high"),
    ({"enabled": True, "effort": "max"}, "high"),
    ({"enabled": True, "effort": "ultra"}, "high"),
    ({"enabled": True, "effort": "HIGH"}, "high"),
    ({"enabled": True}, None),                      # no level: Groq's default
    ({"enabled": True, "effort": "turbo"}, None),   # not a Hermes level: never guessed onto the wire
])
def test_gpt_oss_gets_only_low_medium_or_high(config, expected):
    assert plugin.reasoning_effort_for(GPT_OSS, config) == expected
    assert plugin.reasoning_effort_for("openai/gpt-oss-20b", config) == expected


@pytest.mark.parametrize("config, expected", [
    (None, None),
    ({"enabled": False}, "none"),
    ({"enabled": True, "effort": "none"}, "none"),
    ({"enabled": True, "effort": "minimal"}, "low"),  # never clamped down to "off"
    ({"enabled": True, "effort": "medium"}, "medium"),
    ({"enabled": True, "effort": "max"}, "high"),
])
def test_qwen_3_8_can_turn_reasoning_off(config, expected):
    assert plugin.reasoning_effort_for(QWEN, config) == expected


@pytest.mark.parametrize("model", [
    "llama-3.3-70b-versatile", "moonshotai/kimi-k2-instruct", "", None,
    "qwen/qwen3-32b",  # an older Qwen 3 took only none/default: unchecked models get no field
])
def test_other_models_get_no_reasoning_field(model):
    for config in ({"enabled": False}, {"enabled": True, "effort": "high"}):
        assert plugin.reasoning_effort_for(model, config) is None


def test_profile_puts_reasoning_effort_top_level_only():
    profile = registered_profiles[0]
    high = {"enabled": True, "effort": "high"}
    assert profile.build_api_kwargs_extras(reasoning_config=high, model=GPT_OSS) == ({}, {"reasoning_effort": "high"})
    assert profile.build_api_kwargs_extras(reasoning_config=high, model="llama-3.3-70b-versatile") == ({}, {})
    assert profile.build_api_kwargs_extras(reasoning_config=None, model=GPT_OSS) == ({}, {})
    assert profile.build_api_kwargs_extras(reasoning_config=high) == ({}, {})  # no model in context


def test_model_list_keeps_what_hermes_can_run():
    kept = [model_id for model_id in CATALOG_IDS if plugin.runs_in_hermes(model_id)]
    assert kept == ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "openai/gpt-oss-safeguard-20b", "qwen/qwen3.8-27b"]


def test_excluded_models_are_the_ones_groq_marks_as_unable_to_run_hermes():
    """Every excluded row is audio in/out or has too little context for Hermes' 64K minimum."""
    for item in CATALOG["data"]:
        if plugin.runs_in_hermes(item["id"]):
            assert item["context_window"] >= 64_000 and "tools" in (item.get("supported_features") or [])
        else:
            no_text_out = item.get("output_modalities") != ["text"]
            assert no_text_out or item["context_window"] < 64_000, item["id"]


def test_fetch_models_filters_the_live_list_and_passes_a_failure_through(monkeypatch):
    profile = registered_profiles[0]
    monkeypatch.setattr(StubProfile, "catalog", CATALOG_IDS)
    assert profile.fetch_models(api_key="k") == [m for m in CATALOG_IDS if plugin.runs_in_hermes(m)]
    monkeypatch.setattr(StubProfile, "catalog", None)
    assert profile.fetch_models(api_key="k") is None
    monkeypatch.setattr(StubProfile, "catalog", [])
    assert profile.fetch_models(api_key="k") == []


def test_fallback_models_are_all_listed_and_runnable():
    profile = registered_profiles[0]
    assert profile.fallback_models
    for model_id in profile.fallback_models:
        assert model_id in CATALOG_IDS and plugin.runs_in_hermes(model_id)
    assert profile.default_aux_model in CATALOG_IDS and plugin.runs_in_hermes(profile.default_aux_model)


def test_registers_one_groq_profile():
    assert len(registered_profiles) == 1
    profile = registered_profiles[0]
    assert profile.name == "groq"
    assert profile.base_url == "https://api.groq.com/openai/v1"
    assert profile.env_vars == ("GROQ_API_KEY",)
    assert profile.auth_type == "api_key"
