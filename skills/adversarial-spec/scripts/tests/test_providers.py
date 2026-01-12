"""Tests for providers module."""

import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from providers import (
    MODEL_COSTS,
    DEFAULT_COST,
    BEDROCK_MODEL_MAP,
    load_global_config,
    save_global_config,
    is_bedrock_enabled,
    resolve_bedrock_model,
    validate_bedrock_models,
    load_profile,
    save_profile,
)


class TestModelCosts:
    def test_model_costs_has_expected_models(self):
        expected = [
            "gpt-4o",
            "gemini/gemini-2.0-flash",
            "xai/grok-3",
            "mistral/mistral-large",
            "deepseek/deepseek-chat",
            "zhipu/glm-4",
        ]
        for model in expected:
            assert model in MODEL_COSTS

    def test_costs_have_input_and_output(self):
        for model, costs in MODEL_COSTS.items():
            assert "input" in costs
            assert "output" in costs
            assert isinstance(costs["input"], (int, float))
            assert isinstance(costs["output"], (int, float))

    def test_default_cost_exists(self):
        assert "input" in DEFAULT_COST
        assert "output" in DEFAULT_COST


class TestBedrockModelMap:
    def test_has_claude_models(self):
        assert "claude-3-sonnet" in BEDROCK_MODEL_MAP
        assert "claude-3-haiku" in BEDROCK_MODEL_MAP
        assert "claude-3-opus" in BEDROCK_MODEL_MAP

    def test_has_llama_models(self):
        assert "llama-3-8b" in BEDROCK_MODEL_MAP
        assert "llama-3-70b" in BEDROCK_MODEL_MAP

    def test_maps_to_full_bedrock_ids(self):
        for name, bedrock_id in BEDROCK_MODEL_MAP.items():
            assert "." in bedrock_id or ":" in bedrock_id


class TestGlobalConfig:
    def test_load_nonexistent_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                config = load_global_config()
                assert config == {}

    def test_save_and_load_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "subdir" / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                save_global_config(
                    {"bedrock": {"enabled": True, "region": "us-east-1"}}
                )

                assert config_path.exists()

                loaded = load_global_config()
                assert loaded["bedrock"]["enabled"] is True
                assert loaded["bedrock"]["region"] == "us-east-1"


class TestBedrockEnabled:
    def test_returns_false_when_not_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                assert is_bedrock_enabled() is False

    def test_returns_true_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.json"
            config_path.write_text(json.dumps({"bedrock": {"enabled": True}}))

            with patch("providers.GLOBAL_CONFIG_PATH", config_path):
                assert is_bedrock_enabled() is True


class TestResolveBrockModel:
    def test_resolves_friendly_name(self):
        result = resolve_bedrock_model("claude-3-sonnet")
        assert result == "anthropic.claude-3-sonnet-20240229-v1:0"

    def test_returns_full_id_as_is(self):
        full_id = "anthropic.claude-3-sonnet-20240229-v1:0"
        result = resolve_bedrock_model(full_id)
        assert result == full_id

    def test_returns_none_for_unknown(self):
        result = resolve_bedrock_model("unknown-model")
        assert result is None

    def test_uses_custom_aliases(self):
        config = {"custom_aliases": {"my-model": "custom.model-id"}}
        result = resolve_bedrock_model("my-model", config)
        assert result == "custom.model-id"


class TestValidateBedrockModels:
    def test_validates_available_models(self):
        config = {
            "available_models": ["claude-3-sonnet", "claude-3-haiku"],
        }
        valid, invalid = validate_bedrock_models(["claude-3-sonnet"], config)
        assert len(valid) == 1
        assert len(invalid) == 0

    def test_rejects_unavailable_models(self):
        config = {
            "available_models": ["claude-3-sonnet"],
        }
        valid, invalid = validate_bedrock_models(["claude-3-opus"], config)
        assert len(valid) == 0
        assert len(invalid) == 1
        assert "claude-3-opus" in invalid


class TestProfiles:
    def test_save_and_load_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            profiles_dir = Path(tmpdir) / "profiles"

            with patch("providers.PROFILES_DIR", profiles_dir):
                config = {
                    "models": "gpt-4o,gemini/gemini-2.0-flash",
                    "focus": "security",
                    "persona": "security-engineer",
                }
                save_profile("test-profile", config)

                assert (profiles_dir / "test-profile.json").exists()

                loaded = load_profile("test-profile")
                assert loaded["models"] == "gpt-4o,gemini/gemini-2.0-flash"
                assert loaded["focus"] == "security"
