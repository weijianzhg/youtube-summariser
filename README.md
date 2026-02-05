# YouTube Summariser

A command-line tool that summarizes YouTube videos using AI. It extracts transcripts from YouTube videos and generates structured summaries using OpenAI, Anthropic, or OpenRouter (300+ models).

## Installation

```bash
pip install youtube-summariser
```

> **Note:** Both `youtube-summariser` (British) and `youtube-summarizer` (American) commands are available - use whichever you prefer!

Or install from source:

```bash
git clone https://github.com/weijianzhg/youtube-summariser
cd youtube-summariser
pip install -e .
```

## Quick Start

Run the interactive setup to configure your API keys:

```bash
youtube-summariser init
```

This guides you through:
- Selecting your default provider (Anthropic, OpenAI, or OpenRouter)
- Entering your API key (securely masked)
- Optionally configuring additional providers

## Configuration

### Option 1: Interactive Setup (Recommended)

```bash
youtube-summariser init
```

Settings are saved to a platform-appropriate location:
- **macOS/Linux**: `~/.youtube-summariser/config.yaml`
- **Windows**: `%APPDATA%\youtube-summariser\config.yaml`

Re-run `init` anytime to update your settings.

### Option 2: Environment Variables

```bash
# For Anthropic (default provider)
export ANTHROPIC_API_KEY=your_anthropic_api_key

# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For OpenRouter (access 300+ models)
export OPENROUTER_API_KEY=your_openrouter_api_key
```

Or create a `.env` file in your working directory.

### Configuration Priority

1. Environment variables (highest priority)
2. User config file (`~/.youtube-summariser/config.yaml`)
3. Bundled defaults

### Default Provider

The default provider is **Anthropic**. You can change this via `init` or override per command using `--provider`.

## Usage

```bash
# Interactive configuration
youtube-summariser init

# Summarize a video (saves to auto-generated filename)
youtube-summariser "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output filename
youtube-summariser "https://youtu.be/VIDEO_ID" -o my_summary.md

# Print to terminal only (no file saved)
youtube-summariser "https://youtube.com/watch?v=VIDEO_ID" --no-save

# Use a specific provider
youtube-summariser "https://youtu.be/VIDEO_ID" --provider openai

# Use OpenRouter with access to 300+ models
youtube-summariser "https://youtu.be/VIDEO_ID" --provider openrouter
```

### Search by Title

Don't have a URL? Search for videos by title:

```bash
# Interactive selection (shows top 5 results)
youtube-summariser search "How to make mass"

# Auto-select first result
youtube-summariser search "Python tutorial" --first

# Show more results
youtube-summariser search "cooking recipes" --max-results 10
```

### Commands

| Command | Description |
|---------|-------------|
| `init` | Interactive setup for API keys and preferences |
| `summarise` | Summarize a YouTube video (also aliased as `summarize`) |
| `search` | Search YouTube by title and summarize |

You can also pass a URL directly without the `summarise` subcommand for convenience.

### Options

| Flag | Description |
|------|-------------|
| `-o, --output` | Specify output filename (default: `summary_<video_id>_<timestamp>.md`) |
| `--no-save` | Print summary to terminal without saving to file |
| `--provider` | LLM provider to use: `openai`, `anthropic`, or `openrouter` |
| `--no-stream` | Disable streaming output |
| `--first, -1` | Auto-select first search result (search command only) |
| `--max-results` | Number of search results to display (default: 5) |
| `-v, --version` | Show version number |
| `-h, --help` | Show help message |

### Output Format

Summary files are saved as markdown (`.md`) with the following structure:

```markdown
# YouTube Video Summary

| | |
|---|---|
| **Video URL** | <https://www.youtube.com/watch?v=VIDEO_ID> |
| **Video ID** | `VIDEO_ID` |
| **Generated** | 2025-01-01 14:30:00 |
| **Model** | anthropic / claude-sonnet-4-5-20250929 |

---

## Main Topics
...

## Key Points
...

## Detailed Summary
...

## Notable Quotes
...

## Timestamps for Important Moments
...
```

## Requirements

- Python 3.10+
- An API key for OpenAI, Anthropic, or OpenRouter

## License

MIT License - see [LICENSE](LICENSE) for details.
