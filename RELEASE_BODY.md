## Interactive Configuration Setup

This release introduces the `init` command for easy, guided configuration of your API keys and preferences.

### What's New

#### Interactive `init` Command
Run `youtube-summariser init` to configure the tool interactively:

```
$ youtube-summariser init

YouTube Summariser Configuration
==================================

Which LLM provider would you like to use by default?
  1. anthropic (Recommended)
  2. openai
Select [1]: 1

Enter your Anthropic API key: ********
Model [claude-sonnet-4-5-20250929]:

Do you also want to configure OpenAI? (y/N): n

Configuration saved to ~/.youtube-summariser/config.yaml
```

#### User Configuration File
- Settings stored in platform-appropriate location:
  - **Windows**: `%APPDATA%\youtube-summariser\config.yaml`
  - **macOS/Linux**: `~/.youtube-summariser/config.yaml`
- API keys can be stored in the config file (no more juggling environment variables!)
- Re-run `init` anytime to update your settings - existing values load as defaults

#### Subcommand Structure
The CLI now uses subcommands for better organization:
```bash
youtube-summariser init                    # Interactive setup
youtube-summariser <url>                   # Summarize (backward compatible)
youtube-summariser summarise <url>         # Explicit subcommand
youtube-summariser <url> --provider openai # Override provider
```

### Configuration Priority

1. Environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`)
2. User config file (`~/.youtube-summariser/config.yaml`)
3. Bundled defaults

### Migration

**No breaking changes!** All existing commands work exactly as before. The `init` command is optional - you can continue using environment variables if you prefer.

### Requirements

- Python 3.10+
- OpenAI SDK >= 1.60.0
- Anthropic SDK >= 0.40.0

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
