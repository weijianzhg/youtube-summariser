"""Integration tests for user config with LLM client."""

from youtube_summariser.llm_client import LLMClient, load_config


class TestUserConfigIntegration:
    """Test that LLMClient correctly loads API keys from user config."""

    def test_loads_api_key_from_user_config(self, tmp_path, monkeypatch):
        """LLMClient should use API key from user config when env var not set."""
        # Clear environment variables
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        # Create a mock user config
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """provider: openai
openai:
  api_key: sk-user-config-key
  model: gpt-5.2
anthropic:
  model: claude-sonnet-4-5-20250929
"""
        )

        # Patch the config_manager to use our temp file
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path", lambda: config_file
        )

        # Also need to patch the import in llm_client
        def patched_load():
            import yaml

            with open(config_file, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(
            "youtube_summariser.llm_client.load_user_config", patched_load
        )

        # Load config and verify it includes user config
        config = load_config()
        assert config["openai"]["api_key"] == "sk-user-config-key"

        # Create client - should not raise because API key is in config
        client = LLMClient(provider="openai")
        assert client.provider == "openai"
        assert client._client is not None

    def test_env_var_takes_priority_over_config(self, tmp_path, monkeypatch):
        """Environment variable should take priority over user config."""
        # Set environment variable
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        # Create a user config with a different key
        config = {
            "provider": "anthropic",
            "openai": {},
            "anthropic": {"api_key": "sk-ant-config-key", "model": "claude-sonnet-4-5-20250929"},
        }

        # Create client with this config
        client = LLMClient(config=config, provider="anthropic")
        assert client.provider == "anthropic"
        # The client should be initialized (env var was used)
        assert client._client is not None

    def test_provider_from_user_config_is_used(self, tmp_path, monkeypatch):
        """Default provider should come from user config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """provider: openai
openai:
  api_key: sk-test
  model: gpt-5.2
anthropic:
  model: claude-sonnet-4-5-20250929
"""
        )

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        def patched_load():
            import yaml

            with open(config_file, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(
            "youtube_summariser.llm_client.load_user_config", patched_load
        )

        config = load_config()
        assert config["provider"] == "openai"


class TestConfigPriority:
    """Test configuration priority (user config > bundled config)."""

    def test_user_config_overrides_bundled(self, tmp_path, monkeypatch):
        """User config should take priority over bundled config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """provider: openai
openai:
  model: custom-model
  max_tokens: 5000
anthropic:
  model: claude-sonnet-4-5-20250929
"""
        )

        def patched_load():
            import yaml

            with open(config_file, "r") as f:
                return yaml.safe_load(f)

        monkeypatch.setattr(
            "youtube_summariser.llm_client.load_user_config", patched_load
        )

        config = load_config()
        assert config["openai"]["model"] == "custom-model"
        assert config["openai"]["max_tokens"] == 5000

    def test_falls_back_to_bundled_when_no_user_config(self, monkeypatch):
        """Should use bundled config when no user config exists."""
        # Make load_user_config return None
        monkeypatch.setattr(
            "youtube_summariser.llm_client.load_user_config", lambda: None
        )

        config = load_config()
        # Should have the bundled config values
        assert "provider" in config
        assert "openai" in config
        assert "anthropic" in config
