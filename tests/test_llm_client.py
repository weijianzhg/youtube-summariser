"""Tests for LLM client API key validation."""

import pytest

from youtube_summariser.llm_client import LLMClient


class TestLLMClientAPIKeyValidation:
    """Test that LLMClient properly validates API keys."""

    def test_no_api_keys_raises_unified_error(self, monkeypatch):
        """When no API keys are set, should raise ValueError with unified message."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "anthropic", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config)

        error_msg = str(exc_info.value)
        # Should show unified message mentioning all keys
        assert "No API keys found" in error_msg
        assert "youtube-summariser init" in error_msg
        assert "OPENAI_API_KEY" in error_msg
        assert "ANTHROPIC_API_KEY" in error_msg
        assert "OPENROUTER_API_KEY" in error_msg

    def test_no_api_keys_with_openai_provider_raises_unified_error(self, monkeypatch):
        """When no API keys are set and openai provider requested, should raise unified error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "openai", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="openai")

        error_msg = str(exc_info.value)
        assert "No API keys found" in error_msg
        assert "youtube-summariser init" in error_msg

    def test_no_api_keys_with_anthropic_provider_raises_unified_error(self, monkeypatch):
        """When no API keys are set and anthropic provider requested, raise unified error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "anthropic", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="anthropic")

        error_msg = str(exc_info.value)
        assert "No API keys found" in error_msg
        assert "youtube-summariser init" in error_msg

    def test_no_api_keys_with_openrouter_provider_raises_unified_error(self, monkeypatch):
        """When no API keys are set and openrouter provider requested, raise unified error."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "openrouter", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="openrouter")

        error_msg = str(exc_info.value)
        assert "No API keys found" in error_msg
        assert "youtube-summariser init" in error_msg

    def test_openai_provider_with_only_anthropic_key_suggests_alternative(self, monkeypatch):
        """When only ANTHROPIC_API_KEY is set but openai provider requested, suggest alternative."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "openai", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="openai")

        error_msg = str(exc_info.value)
        assert "OpenAI API key not configured" in error_msg
        # Should suggest using alternative providers
        assert "--provider anthropic" in error_msg or "--provider openrouter" in error_msg

    def test_anthropic_provider_with_only_openai_key_suggests_alternative(self, monkeypatch):
        """When only OPENAI_API_KEY is set but anthropic provider requested, suggest alternative."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "anthropic", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="anthropic")

        error_msg = str(exc_info.value)
        assert "Anthropic API key not configured" in error_msg
        # Should suggest using alternative providers
        assert "--provider openai" in error_msg or "--provider openrouter" in error_msg

    def test_openrouter_provider_with_only_openai_key_suggests_alternative(self, monkeypatch):
        """When only OPENAI_API_KEY is set but openrouter requested, suggest alternative."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {"provider": "openrouter", "openai": {}, "anthropic": {}, "openrouter": {}}
            LLMClient(config=config, provider="openrouter")

        error_msg = str(exc_info.value)
        assert "OpenRouter API key not configured" in error_msg
        # Should suggest using alternative providers
        assert "--provider openai" in error_msg or "--provider anthropic" in error_msg

    def test_unsupported_provider_raises_error(self, monkeypatch):
        """When an unsupported provider is specified, should raise ValueError."""
        # Need at least one key to pass the first check
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        with pytest.raises(ValueError) as exc_info:
            config = {
                "provider": "invalid_provider",
                "openai": {},
                "anthropic": {},
                "openrouter": {},
            }
            LLMClient(config=config, provider="invalid_provider")

        assert "Unsupported provider" in str(exc_info.value)

    def test_openai_with_valid_key_initializes(self, monkeypatch):
        """When OPENAI_API_KEY is set, OpenAI client should initialize."""
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-12345")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Should not raise - just tests initialization
        config = {"provider": "openai", "openai": {}, "anthropic": {}, "openrouter": {}}
        client = LLMClient(config=config, provider="openai")
        assert client.provider == "openai"
        assert client._client is not None

    def test_anthropic_with_valid_key_initializes(self, monkeypatch):
        """When ANTHROPIC_API_KEY is set, Anthropic client should initialize."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-12345")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

        # Should not raise - just tests initialization
        config = {"provider": "anthropic", "openai": {}, "anthropic": {}, "openrouter": {}}
        client = LLMClient(config=config, provider="anthropic")
        assert client.provider == "anthropic"
        assert client._client is not None

    def test_openrouter_with_valid_key_initializes(self, monkeypatch):
        """When OPENROUTER_API_KEY is set, OpenRouter client should initialize."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key-12345")

        # Should not raise - just tests initialization
        config = {"provider": "openrouter", "openai": {}, "anthropic": {}, "openrouter": {}}
        client = LLMClient(config=config, provider="openrouter")
        assert client.provider == "openrouter"
        assert client._client is not None
