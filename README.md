# YouTube Summariser

A command-line tool that summarizes YouTube videos using AI. It extracts transcripts from YouTube videos and generates structured summaries using OpenAI, Anthropic, or OpenRouter (300+ models).

## Installation

```bash
pip install youtube-summariser
```

> **Note:** `youtube-summariser` (British), `youtube-summarizer` (American), and `ys` (short alias) commands are all available.

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

# Same command with the short alias
ys init

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

### Summarize a Channel

Summarize every video published on a channel:

```bash
youtube-summariser channel "https://www.youtube.com/@CHANNEL"
```

For large channels, limit the run to the most recent videos:

```bash
youtube-summariser channel "https://www.youtube.com/@CHANNEL" --max-videos 10
```

Each video goes through the same transcript and LLM summarization pipeline as the
single-video command. Summaries are saved individually under `channel-summaries/` by
default. Use `--output-dir` to choose another directory.

> **Note:** Without `--max-videos`, the command processes every video on the channel.
> This can take a long time and make many LLM API calls on a large channel.

### Commands


| Command     | Description                                             |
| ----------- | ------------------------------------------------------- |
| `init`      | Interactive setup for API keys and preferences          |
| `summarise` | Summarize a YouTube video (also aliased as `summarize`) |
| `search`    | Search YouTube by title and summarize                   |
| `channel`   | Summarize all or the most recent videos from a channel  |


You can also pass a URL directly without the `summarise` subcommand for convenience.

### Options


| Flag            | Description                                                            |
| --------------- | ---------------------------------------------------------------------- |
| `-o, --output`  | Specify output filename (default: `YYYY-MM-DD__video-title-slug__video-id.md`) |
| `--no-save`     | Print summary to terminal without saving to file                       |
| `--provider`    | LLM provider to use: `openai`, `anthropic`, or `openrouter`            |
| `--no-stream`   | Disable streaming output                                               |
| `--first, -1`   | Auto-select first search result (search command only)                  |
| `--max-results` | Number of search results to display (default: 5)                       |
| `--max-videos`  | Maximum recent channel videos to summarize (default: all)              |
| `--output-dir`  | Directory for channel summary files (default: `channel-summaries`)     |
| `-v, --version` | Show version number                                                    |
| `-h, --help`    | Show help message                                                      |


### zsh Tip (URLs Without Quotes)

In `zsh`, unquoted YouTube URLs containing `?` are treated as glob patterns before the CLI runs.

Use one of these approaches:

```bash
# Escape ? in the URL
ys https://www.youtube.com/watch\?v=VIDEO_ID

# Or disable globbing for this command
noglob ys https://www.youtube.com/watch?v=VIDEO_ID
```

To make this convenient permanently, add to `~/.zshrc`:

```bash
ys() { noglob command ys "$@"; }
```

### Output Format

Summary files are saved as Obsidian-friendly markdown (`.md`) with YAML frontmatter:

```markdown
---
title: "Video Title"
url: "https://www.youtube.com/watch?v=VIDEO_ID"
video_id: VIDEO_ID
channel: "Channel Name"
published_at: 2026-07-20
created: 2026-07-28
content_type: tutorial
series: "Example Course"
series_index: 2
concepts:
  - "gradient descent"
  - "backpropagation"
prerequisites:
  - "basic calculus"
tags:
  - youtube
  - summary
  - "machine-learning"
  - "neural-networks"
model: "anthropic/claude-sonnet-4-5-20250929"
---

# Video Title

## TL;DR
...

## Key Takeaways
- Insight with a linked timestamp [12:34](https://www.youtube.com/watch?v=VIDEO_ID&t=754)

## Detailed Summary
...

## Notable Quotes
...
```

Topics are normalized into reusable Obsidian tags. Concepts, prerequisites, series
position, content type, and the original publication date are emitted as structured
properties for filtering and relationship indexing.

These properties make corpus-level linking deterministic, but Obsidian's native Graph
View only draws edges for actual `[[wikilinks]]`. A vault-wide indexing step is still
needed to create shared concept notes and links between videos.

Notes are ready to drop into an Obsidian vault. Point `--output-dir` at a vault folder
for channel runs.

## Requirements

- Python 3.10+
- An API key for OpenAI, Anthropic, or OpenRouter

## License

MIT License - see [LICENSE](LICENSE) for details.
