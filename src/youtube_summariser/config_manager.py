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
    provider_defaults = {"anthropic": "1", "openai": "2", "openrouter": "3", "local": "4"}
    provider_default = provider_defaults.get(existing_provider, "1")

    print("Which LLM provider would you like to use by default?")
    print("  1. anthropic (Recommended)")
    print("  2. openai")
    print("  3. openrouter (Access 300+ models)")
    print("  4. local (Run a downloaded Transformers model)")
    selection = prompt_with_default("Select", provider_default)

    if selection == "2":
        provider = "openai"
    elif selection == "3":
        provider = "openrouter"
    elif selection == "4":
        provider = "local"
    else:
        provider = "anthropic"

    # Initialize config structure
    config: dict[str, Any] = {
        "provider": provider,
        "openai": existing_config.get("openai", {}).copy(),
        "anthropic": existing_config.get("anthropic", {}).copy(),
        "openrouter": existing_config.get("openrouter", {}).copy(),
        "local": existing_config.get("local", {}).copy(),
    }

    # Ensure max_tokens defaults exist
    if "max_tokens" not in config["openai"]:
        config["openai"]["max_tokens"] = 3000
    if "max_tokens" not in config["anthropic"]:
        config["anthropic"]["max_tokens"] = 3000
    if "max_tokens" not in config["openrouter"]:
        config["openrouter"]["max_tokens"] = 3000
    if "max_tokens" not in config["local"]:
        config["local"]["max_tokens"] = 512
    if "max_input_tokens" not in config["local"]:
        config["local"]["max_input_tokens"] = 1536
    if "cache_dir" not in config["local"]:
        config["local"]["cache_dir"] = "~/.cache/youtube-summariser/models"
    if "device" not in config["local"]:
        config["local"]["device"] = "auto"
    if "torch_dtype" not in config["local"]:
        config["local"]["torch_dtype"] = "auto"

    print()

    # Configure primary provider first
    if provider == "anthropic":
        _configure_anthropic(config, existing_config)
        _ask_configure_others(config, existing_config, exclude=["anthropic"])
    elif provider == "openai":
        _configure_openai(config, existing_config)
        _ask_configure_others(config, existing_config, exclude=["openai"])
    elif provider == "openrouter":
        _configure_openrouter(config, existing_config)
        _ask_configure_others(config, existing_config, exclude=["openrouter"])
    else:  # local
        _configure_local(config, existing_config)
        _ask_configure_others(config, existing_config, exclude=["local"])

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


def _configure_openrouter(config: dict[str, Any], existing_config: dict[str, Any]) -> None:
    """Configure OpenRouter settings."""
    existing_openrouter = existing_config.get("openrouter", {})
    existing_key = existing_openrouter.get("api_key", "")
    existing_model = existing_openrouter.get("model", "anthropic/claude-sonnet-4.5")

    print("OpenRouter provides access to 300+ models from various providers.")
    print("Get your API key at: https://openrouter.ai/settings/keys")
    print()
    api_key = prompt_with_default("Enter your OpenRouter API key", existing_key, password=True)
    print()
    print("Model format: provider/model-name (e.g., anthropic/claude-sonnet-4.5)")
    model = prompt_with_default("Model", existing_model)

    if api_key:
        config["openrouter"]["api_key"] = api_key
    config["openrouter"]["model"] = model


def _configure_local(config: dict[str, Any], existing_config: dict[str, Any]) -> None:
    """Configure local model settings."""
    existing_local = existing_config.get("local", {})
    existing_path = existing_local.get("model_path", "")
    existing_max_tokens = str(existing_local.get("max_tokens", 512))
    existing_max_input_tokens = str(existing_local.get("max_input_tokens", 1536))

    print("Local provider runs an extracted Hugging Face model directory or .tar.gz archive.")
    model_path = prompt_with_default("Model path", existing_path)
    max_tokens = prompt_with_default("Max output tokens", existing_max_tokens)
    max_input_tokens = prompt_with_default("Max input tokens", existing_max_input_tokens)

    config["local"]["model_path"] = model_path
    config["local"]["max_tokens"] = int(max_tokens)
    config["local"]["max_input_tokens"] = int(max_input_tokens)


def _ask_configure_others(
    config: dict[str, Any], existing_config: dict[str, Any], exclude: list[str]
) -> None:
    """Ask user if they want to configure other providers."""
    providers = {
        "openai": ("OpenAI", _configure_openai),
        "anthropic": ("Anthropic", _configure_anthropic),
        "openrouter": ("OpenRouter", _configure_openrouter),
        "local": ("Local model", _configure_local),
    }

    for provider_key, (provider_name, configure_func) in providers.items():
        if provider_key not in exclude:
            print()
            configure = prompt_with_default(
                f"Do you also want to configure {provider_name}? (y/N)", "n"
            )
            if configure.lower() == "y":
                print()
                configure_func(config, existing_config)
