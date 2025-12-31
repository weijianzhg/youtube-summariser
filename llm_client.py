"""
LLM Client abstraction for OpenAI and Anthropic.
"""
import os
import yaml
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Load configuration
def load_config() -> dict:
    """Load configuration from config.yaml."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in configuration file: {e}")


class LLMClient:
    """Unified LLM client supporting OpenAI and Anthropic."""

    def __init__(self, config: Optional[dict] = None):
        """Initialize the LLM client based on configuration."""
        self.config = config or load_config()
        self.provider = self.config.get("provider", "openai")
        self._client = None
        self._init_client()

    def _init_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            self._client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            self._client = anthropic.Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def get_model(self) -> str:
        """Get the model name for the current provider."""
        provider_config = self.config.get(self.provider, {})
        return provider_config.get("model", "gpt-4o" if self.provider == "openai" else "claude-sonnet-4-20250514")

    def get_max_tokens(self) -> int:
        """Get max tokens for the current provider."""
        provider_config = self.config.get(self.provider, {})
        return provider_config.get("max_tokens", 1500)

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
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _chat_openai(self, system_prompt: str, user_message: str) -> str:
        """Send chat request to OpenAI."""
        response = self._client.chat.completions.create(
            model=self.get_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            max_completion_tokens=self.get_max_tokens()
        )
        return response.choices[0].message.content

    def _chat_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Send chat request to Anthropic."""
        response = self._client.messages.create(
            model=self.get_model(),
            max_tokens=self.get_max_tokens(),
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        return response.content[0].text

