## OpenRouter Support - Access 300+ AI Models

This release adds OpenRouter as a third LLM provider, giving you access to 300+ AI models through a single unified API.

### What's New

#### OpenRouter Provider
Use `--provider openrouter` to access models from OpenAI, Anthropic, Meta, Google, Mistral, and many more:

```bash
# Set your API key
export OPENROUTER_API_KEY=sk-or-v1-xxxx

# Summarize with OpenRouter
youtube-summariser "https://youtube.com/watch?v=VIDEO_ID" --provider openrouter
```

Or configure it as your default provider:

```
$ youtube-summariser init

YouTube Summariser Configuration
==================================

Which LLM provider would you like to use by default?
  1. anthropic (Recommended)
  2. openai
  3. openrouter (Access 300+ models)
Select [1]: 3

OpenRouter provides access to 300+ models from various providers.
Get your API key at: https://openrouter.ai/settings/keys

Enter your OpenRouter API key: ********
Model format: provider/model-name (e.g., anthropic/claude-sonnet-4.5)
Model [anthropic/claude-sonnet-4.5]:

Configuration saved to ~/.youtube-summariser/config.yaml
```

#### Why OpenRouter?
- **300+ models**: Access models from multiple providers with one API key
- **Pay-per-use**: No subscriptions required
- **Model flexibility**: Easily switch between models like `anthropic/claude-sonnet-4.5`, `openai/gpt-4o`, `google/gemini-pro`, etc.
- **Full streaming support**: Real-time output just like native providers

### Configuration Priority

1. Environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`)
2. User config file (`~/.youtube-summariser/config.yaml`)
3. Bundled defaults

### Requirements

- Python 3.10+
- OpenAI SDK >= 1.60.0
- Anthropic SDK >= 0.40.0
- OpenRouter SDK >= 0.1.0

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
