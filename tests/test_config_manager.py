"""Tests for config_manager module."""

import sys
from pathlib import Path

from youtube_summariser.config_manager import (
    get_config_dir,
    get_config_path,
    load_user_config,
    save_user_config,
)


class TestConfigPaths:
    """Test configuration path functions."""

    def test_get_config_dir_returns_path(self):
        """get_config_dir should return a Path object."""
        config_dir = get_config_dir()
        assert isinstance(config_dir, Path)
        # Windows uses 'youtube-summariser', Unix uses '.youtube-summariser'
        if sys.platform == "win32":
            assert config_dir.name == "youtube-summariser"
        else:
            assert config_dir.name == ".youtube-summariser"

    def test_get_config_path_returns_yaml_path(self):
        """get_config_path should return path to config.yaml."""
        config_path = get_config_path()
        assert isinstance(config_path, Path)
        assert config_path.name == "config.yaml"
        # Windows uses 'youtube-summariser', Unix uses '.youtube-summariser'
        if sys.platform == "win32":
            assert config_path.parent.name == "youtube-summariser"
        else:
            assert config_path.parent.name == ".youtube-summariser"

    def test_windows_uses_appdata(self, monkeypatch):
        """On Windows, should use APPDATA directory."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setenv("APPDATA", "C:\\Users\\Test\\AppData\\Roaming")

        # Re-import to pick up the monkeypatched values
        from youtube_summariser import config_manager

        config_dir = config_manager.get_config_dir()
        assert "AppData" in str(config_dir) or "youtube-summariser" in str(config_dir)

    def test_unix_uses_home_dotfile(self, monkeypatch):
        """On Unix, should use dotfile in home directory."""
        monkeypatch.setattr(sys, "platform", "darwin")

        from youtube_summariser import config_manager

        config_dir = config_manager.get_config_dir()
        assert config_dir.name == ".youtube-summariser"


class TestLoadUserConfig:
    """Test load_user_config function."""

    def test_returns_none_when_no_config(self, tmp_path, monkeypatch):
        """When config file doesn't exist, should return None."""
        # Point to a temp directory that doesn't have config
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path",
            lambda: tmp_path / "nonexistent" / "config.yaml",
        )
        result = load_user_config()
        assert result is None

    def test_loads_valid_yaml(self, tmp_path, monkeypatch):
        """Should load and return valid YAML config."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text(
            """provider: anthropic
openai:
  api_key: sk-test-openai
  model: gpt-5.2
anthropic:
  api_key: sk-ant-test
  model: claude-sonnet-4-5-20250929
"""
        )
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path", lambda: config_file
        )

        result = load_user_config()
        assert result is not None
        assert result["provider"] == "anthropic"
        assert result["openai"]["api_key"] == "sk-test-openai"
        assert result["anthropic"]["api_key"] == "sk-ant-test"

    def test_returns_none_on_invalid_yaml(self, tmp_path, monkeypatch):
        """Should return None if YAML is invalid."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("invalid: yaml: content: [")
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path", lambda: config_file
        )

        result = load_user_config()
        assert result is None


class TestSaveUserConfig:
    """Test save_user_config function."""

    def test_creates_directory_and_file(self, tmp_path, monkeypatch):
        """Should create config directory and file."""
        config_dir = tmp_path / ".youtube-summariser"
        config_file = config_dir / "config.yaml"

        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_dir", lambda: config_dir
        )
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path", lambda: config_file
        )

        test_config = {
            "provider": "openai",
            "openai": {"api_key": "sk-test", "model": "gpt-5.2"},
            "anthropic": {"model": "claude-sonnet-4-5-20250929"},
        }

        save_user_config(test_config)

        assert config_dir.exists()
        assert config_file.exists()

        # Verify content
        loaded = load_user_config()
        assert loaded["provider"] == "openai"
        assert loaded["openai"]["api_key"] == "sk-test"

    def test_overwrites_existing_config(self, tmp_path, monkeypatch):
        """Should overwrite existing config file."""
        config_dir = tmp_path / ".youtube-summariser"
        config_dir.mkdir()
        config_file = config_dir / "config.yaml"
        config_file.write_text("provider: old_value\n")

        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_dir", lambda: config_dir
        )
        monkeypatch.setattr(
            "youtube_summariser.config_manager.get_config_path", lambda: config_file
        )

        save_user_config({"provider": "new_value"})

        loaded = load_user_config()
        assert loaded["provider"] == "new_value"
