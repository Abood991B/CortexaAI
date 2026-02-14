"""Tests for the Multi-LLM Provider System."""

import pytest
from unittest.mock import patch, MagicMock

from config.llm_providers import (
    LLMProvider,
    PROVIDER_CONFIGS,
    FALLBACK_ORDER,
    get_llm,
    llm_provider,
)


class TestProviderConfigs:
    """Tests for provider configuration registry."""

    def test_all_providers_registered(self):
        """Verify all 6 providers are registered."""
        expected = {"google", "openai", "anthropic", "groq", "deepseek", "openrouter"}
        assert set(PROVIDER_CONFIGS.keys()) == expected

    def test_each_provider_has_required_fields(self):
        """Each provider config must have init_fn, default_model, api_key_env."""
        for name, config in PROVIDER_CONFIGS.items():
            assert callable(config["init_fn"]), f"{name} missing init_fn"
            assert isinstance(config["default_model"], str), f"{name} missing default_model"
            assert isinstance(config["api_key_env"], str), f"{name} missing api_key_env"
            assert isinstance(config["free_tier"], bool), f"{name} missing free_tier"
            assert isinstance(config["models"], dict), f"{name} missing models"

    def test_free_tier_providers(self):
        """Google, Groq, and OpenRouter should be marked free."""
        free_providers = [n for n, c in PROVIDER_CONFIGS.items() if c["free_tier"]]
        assert "google" in free_providers
        assert "groq" in free_providers
        assert "openrouter" in free_providers

    def test_fallback_order_prioritizes_free(self):
        """Free tier providers should come first in fallback order."""
        # First 3 in fallback should all be free
        for provider in FALLBACK_ORDER[:3]:
            assert PROVIDER_CONFIGS[provider]["free_tier"], (
                f"{provider} is in first 3 fallback positions but not free tier"
            )


class TestLLMProvider:
    """Tests for the LLMProvider class."""

    def test_initialization(self):
        """Provider initializes with empty state."""
        provider = LLMProvider()
        assert provider._instances == {}
        assert provider._health == {}
        assert provider._call_counts == {}
        assert provider._error_counts == {}

    def test_get_available_providers_with_no_keys(self):
        """With no API keys, no providers are available."""
        provider = LLMProvider()
        with patch.object(type(provider), "__init__", lambda self: None):
            provider = LLMProvider.__new__(LLMProvider)
            provider._instances = {}
            provider._health = {}
            provider._call_counts = {}
            provider._error_counts = {}

        # Mock settings to have no API keys
        with patch("config.llm_providers.settings") as mock_settings:
            for config in PROVIDER_CONFIGS.values():
                setattr(mock_settings, config["api_key_env"], None)
            available = provider.get_available_providers()
            assert available == []

    def test_get_available_providers_with_google_key(self):
        """With only Google key set, only google is available."""
        provider = LLMProvider()
        with patch("config.llm_providers.settings") as mock_settings:
            mock_settings.google_api_key = "test-key"
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.deepseek_api_key = None
            mock_settings.openrouter_api_key = None
            available = provider.get_available_providers()
            assert available == ["google"]

    def test_get_provider_status_structure(self):
        """Provider status should have correct structure for each provider."""
        provider = LLMProvider()
        with patch("config.llm_providers.settings") as mock_settings:
            mock_settings.google_api_key = "test"
            mock_settings.openai_api_key = None
            mock_settings.anthropic_api_key = None
            mock_settings.groq_api_key = None
            mock_settings.deepseek_api_key = None
            mock_settings.openrouter_api_key = None

            status = provider.get_provider_status()

            assert "google" in status
            google_status = status["google"]
            assert google_status["configured"] is True
            assert "healthy" in google_status
            assert "call_count" in google_status
            assert "free_tier" in google_status
            assert "available_models" in google_status

            assert status["openai"]["configured"] is False

    def test_reset_health_single(self):
        """Reset health for a single provider."""
        provider = LLMProvider()
        provider._health["google"] = False
        provider._health["openai"] = False
        provider.reset_health("google")
        assert provider._health["google"] is True
        assert provider._health["openai"] is False

    def test_reset_health_all(self):
        """Reset health for all providers."""
        provider = LLMProvider()
        provider._health["google"] = False
        provider._health["openai"] = False
        provider.reset_health()
        assert all(v is True for v in provider._health.values())

    def test_get_model_raises_on_no_providers(self):
        """Should raise RuntimeError when no providers are available."""
        provider = LLMProvider()
        with patch("config.llm_providers.settings") as mock_settings:
            for config in PROVIDER_CONFIGS.values():
                setattr(mock_settings, config["api_key_env"], None)
            mock_settings.default_model_provider = "google"
            mock_settings.default_model_name = "gemini-2.0-flash"

            with pytest.raises(RuntimeError, match="No LLM provider available"):
                provider.get_model()


class TestGetLLMConvenience:
    """Tests for the get_llm() convenience function."""

    def test_get_llm_delegates_to_provider(self):
        """get_llm() should call llm_provider.get_model()."""
        with patch.object(llm_provider, "get_model") as mock_get:
            mock_get.return_value = MagicMock()
            result = get_llm("google", "gemini-2.0-flash", 0.5)
            mock_get.assert_called_once_with("google", "gemini-2.0-flash", 0.5)
