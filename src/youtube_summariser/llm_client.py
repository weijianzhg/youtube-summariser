"""LLM Client abstraction for OpenAI, Anthropic, and OpenRouter."""

import logging
import os
from importlib import resources
from typing import Iterator, Optional

import yaml

from .config_manager import load_user_config

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """
    Load configuration with priority: user config > bundled config.

    User config is loaded from ~/.youtube-summariser/config.yaml.
    Falls back to bundled config.yaml if user config doesn't exist.
    """
    # Try user config first
    user_config = load_user_config()
    if user_config is not None:
        return user_config

    # Fall back to bundled config
    try:
        config_file = resources.files(__package__) / "config.yaml"
        with config_file.open("r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        # Return default config if file not found
        return {
            "provider": "openai",
            "openai": {"model": "gpt-5.2", "max_tokens": 3000},
            "anthropic": {"model": "claude-sonnet-4-5-20250929", "max_tokens": 3000},
            "openrouter": {"model": "anthropic/claude-sonnet-4.5", "max_tokens": 3000},
        }
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


class LLMClient:
    """Unified LLM client supporting OpenAI, Anthropic, and OpenRouter."""

    def __init__(self, config: Optional[dict] = None, provider: Optional[str] = None):
        """
        Initialize the LLM client.

        Args:
            config: Optional configuration dict. If not provided, loads from config.yaml.
            provider: Optional provider override ('openai', 'anthropic', or 'openrouter').
        """
        self.config = config or load_config()
        self.provider = provider or self.config.get("provider", "openai")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        # API key priority: environment variable > user config
        openai_key = os.environ.get("OPENAI_API_KEY") or self.config.get("openai", {}).get(
            "api_key"
        )
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or self.config.get("anthropic", {}).get(
            "api_key"
        )
        openrouter_key = os.environ.get("OPENROUTER_API_KEY") or self.config.get(
            "openrouter", {}
        ).get("api_key")

        # Check if no keys are available
        if not openai_key and not anthropic_key and not openrouter_key:
            raise ValueError(
                "No API keys found. Run 'youtube-summariser init' to configure, "
                "or set OPENAI_API_KEY, ANTHROPIC_API_KEY, or OPENROUTER_API_KEY "
                "environment variable."
            )

        if self.provider == "openai":
            from openai import OpenAI

            if not openai_key:
                raise ValueError(
                    "OpenAI API key not configured. "
                    "Run 'youtube-summariser init' or set OPENAI_API_KEY environment variable. "
                    "Or use --provider anthropic or --provider openrouter instead."
                )
            self._client = OpenAI(api_key=openai_key)
        elif self.provider == "anthropic":
            import anthropic

            if not anthropic_key:
                raise ValueError(
                    "Anthropic API key not configured. "
                    "Run 'youtube-summariser init' or set ANTHROPIC_API_KEY environment variable. "
                    "Or use --provider openai or --provider openrouter instead."
                )
            self._client = anthropic.Anthropic(api_key=anthropic_key)
        elif self.provider == "openrouter":
            from openrouter import OpenRouter

            if not openrouter_key:
                raise ValueError(
                    "OpenRouter API key not configured. "
                    "Run 'youtube-summariser init' or set OPENROUTER_API_KEY environment variable. "
                    "Or use --provider openai or --provider anthropic instead."
                )
            self._client = OpenRouter(api_key=openrouter_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_model(self) -> str:
        """Get the model name for the current provider."""
        provider_config = self.config.get(self.provider, {})
        defaults = {
            "openai": "gpt-5.2",
            "anthropic": "claude-sonnet-4-5-20250929",
            "openrouter": "anthropic/claude-sonnet-4.5",
        }
        default = defaults.get(self.provider, "gpt-5.2")
        return provider_config.get("model", default)

    def get_max_tokens(self) -> int:
        """Get max tokens for the current provider."""
        provider_config = self.config.get(self.provider, {})
        return provider_config.get("max_tokens", 3000)

    def chat(self, system_prompt: str, user_message: str) -> str:
        """
        Send a chat request to the LLM.

        Args:
            system_prompt: The system prompt
            user_message: The user message

        Returns:
            The assistant's response text
        """
        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_message)
        elif self.provider == "anthropic":
            return self._chat_anthropic(system_prompt, user_message)
        elif self.provider == "openrouter":
            raise NotImplementedError(
                "Non-streaming mode is not supported for OpenRouter. "
                "Please use streaming mode (remove --no-stream flag)."
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def stream_chat(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """
        Send a streaming chat request to the LLM.

        Args:
            system_prompt: The system prompt
            user_message: The user message

        Yields:
            Text chunks as they are generated by the LLM
        """
        if self.provider == "openai":
            yield from self._stream_chat_openai(system_prompt, user_message)
        elif self.provider == "anthropic":
            yield from self._stream_chat_anthropic(system_prompt, user_message)
        elif self.provider == "openrouter":
            yield from self._stream_chat_openrouter(system_prompt, user_message)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(self, system_prompt: str, user_message: str) -> str:
        """Send chat request to OpenAI."""
        response = self._client.chat.completions.create(
            model=self.get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=self.get_max_tokens(),
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Send chat request to Anthropic."""
        response = self._client.messages.create(
            model=self.get_model(),
            max_tokens=self.get_max_tokens(),
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    def _stream_chat_openai(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Send streaming chat request to OpenAI."""
        stream = self._client.chat.completions.create(
            model=self.get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_completion_tokens=self.get_max_tokens(),
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and len(chunk.choices) > 0:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content

    def _stream_chat_anthropic(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Send streaming chat request to Anthropic."""
        with self._client.messages.stream(
            model=self.get_model(),
            max_tokens=self.get_max_tokens(),
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    def _stream_chat_openrouter(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Send streaming chat request to OpenRouter."""
        stream = self._client.chat.send(
            model=self.get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=self.get_max_tokens(),
            stream=True,
        )
        for event in stream:
            if event.choices and len(event.choices) > 0:
                delta = event.choices[0].delta
                if delta and delta.content:
                    yield delta.content
