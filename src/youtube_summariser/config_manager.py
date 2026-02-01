"""Configuration manager for user settings."""

import getpass
import os
import sys
from pathlib import Path
from typing import Any, Optional

import yaml


def get_config_dir() -> Path:
    """
    Return the user config directory.

    Platform-specific locations:
    - Windows: %APPDATA%\\youtube-summariser\\
    - macOS/Linux: ~/.youtube-summariser/
    """
    if sys.platform == "win32":
        # Windows: use APPDATA
        appdata = os.environ.get("APPDATA")
        if appdata:
            return Path(appdata) / "youtube-summariser"
        # Fallback to home directory if APPDATA not set
        return Path.home() / "youtube-summariser"
    else:
        # macOS and Linux: use dotfile in home directory
        return Path.home() / ".youtube-summariser"


def get_config_path() -> Path:
    """Return the full path to user config.yaml."""
    return get_config_dir() / "config.yaml"


def load_user_config() -> Optional[dict[str, Any]]:
    """
    Load existing user configuration if it exists.

    Returns:
        Config dict if file exists and is valid, None otherwise.
    """
    config_path = get_config_path()
    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return None


def save_user_config(config: dict[str, Any]) -> None:
    """
    Save configuration to user directory.

    Args:
        config: Configuration dictionary to save.
    """
    config_dir = get_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = get_config_path()
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def prompt_with_default(prompt: str, default: str = "", password: bool = False) -> str:
    """
    Prompt user for input with an optional default value.

    Args:
        prompt: The prompt to display.
        default: Default value if user presses enter.
        password: If True, mask input using getpass.

    Returns:
        User input or default value.
    """
    if default and not password:
        full_prompt = f"{prompt} [{default}]: "
    elif default and password:
        full_prompt = f"{prompt} [****]: "
    else:
        full_prompt = f"{prompt}: "

    if password:
        value = getpass.getpass(full_prompt)
    else:
        value = input(full_prompt)

    return value.strip() if value.strip() else default


def run_init() -> None:
    """Run the interactive configuration setup."""
    print("\nYouTube Summariser Configuration")
    print("=" * 34)
    print()

    # Load existing config for defaults
    existing_config = load_user_config() or {}

    # Provider selection
    existing_provider = existing_config.get("provider", "anthropic")
    provider_default = "1" if existing_provider == "anthropic" else "2"

    print("Which LLM provider would you like to use by default?")
    print("  1. anthropic (Recommended)")
    print("  2. openai")
    selection = prompt_with_default("Select", provider_default)

    if selection == "2":
        provider = "openai"
    else:
        provider = "anthropic"

    # Initialize config structure
    config: dict[str, Any] = {
        "provider": provider,
        "openai": existing_config.get("openai", {}).copy(),
        "anthropic": existing_config.get("anthropic", {}).copy(),
    }

    # Ensure max_tokens defaults exist
    if "max_tokens" not in config["openai"]:
        config["openai"]["max_tokens"] = 3000
    if "max_tokens" not in config["anthropic"]:
        config["anthropic"]["max_tokens"] = 3000

    print()

    # Configure primary provider first
    if provider == "anthropic":
        _configure_anthropic(config, existing_config)
        print()
        configure_other = prompt_with_default("Do you also want to configure OpenAI? (y/N)", "n")
        if configure_other.lower() == "y":
            print()
            _configure_openai(config, existing_config)
    else:
        _configure_openai(config, existing_config)
        print()
        configure_other = prompt_with_default("Do you also want to configure Anthropic? (y/N)", "n")
        if configure_other.lower() == "y":
            print()
            _configure_anthropic(config, existing_config)

    # Save configuration
    save_user_config(config)

    config_path = get_config_path()
    print()
    print(f"Configuration saved to {config_path}")


def _configure_anthropic(config: dict[str, Any], existing_config: dict[str, Any]) -> None:
    """Configure Anthropic settings."""
    existing_anthropic = existing_config.get("anthropic", {})
    existing_key = existing_anthropic.get("api_key", "")
    existing_model = existing_anthropic.get("model", "claude-sonnet-4-5-20250929")

    api_key = prompt_with_default("Enter your Anthropic API key", existing_key, password=True)
    model = prompt_with_default("Model", existing_model)

    if api_key:
        config["anthropic"]["api_key"] = api_key
    config["anthropic"]["model"] = model


def _configure_openai(config: dict[str, Any], existing_config: dict[str, Any]) -> None:
    """Configure OpenAI settings."""
    existing_openai = existing_config.get("openai", {})
    existing_key = existing_openai.get("api_key", "")
    existing_model = existing_openai.get("model", "gpt-5.2")

    api_key = prompt_with_default("Enter your OpenAI API key", existing_key, password=True)
    model = prompt_with_default("Model", existing_model)

    if api_key:
        config["openai"]["api_key"] = api_key
    config["openai"]["model"] = model
