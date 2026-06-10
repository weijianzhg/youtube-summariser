# YouTube Summariser

A command-line tool that summarizes YouTube videos using AI. It extracts transcripts from YouTube videos and generates structured summaries using OpenAI, Anthropic, OpenRouter (300+ models), or a local Transformers model.

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

- Selecting your default provider (Anthropic, OpenAI, OpenRouter, or local)
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

# For a local Transformers model directory or .tar.gz archive
export YOUTUBE_SUMMARISER_LOCAL_MODEL=/path/to/local-model.tar.gz
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

# Download and use the default local Qwen GGUF Q4_K_M fine-tune
youtube-summariser "https://youtu.be/VIDEO_ID" --local

# Use the higher-quality Q5_K_M GGUF variant
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model weijianzhg/youtube-summariser-qwen3.5-4b-GGUF:Q5_K_M

# Use a specific local model repo/archive without editing config
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model owner/model-name

# Use a local model archive from disk
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model /Users/eve/tuned-tensor-repos/youtube-summariser/downloads/youtube-summariser-qwen3.5-4b-run-728676d4.tar.gz

# Force map-reduce mode for a model with a small context window
youtube-summariser "https://youtu.be/VIDEO_ID" --summary-strategy map-reduce
```

### Local Models

Install the optional local runtime first:

```bash
pip install -e ".[local]"
```

Then either pass a model reference per command with `--local-model`, set `YOUTUBE_SUMMARISER_LOCAL_MODEL`, or save it in `~/.youtube-summariser/config.yaml`:

```bash
# Downloads the Q4_K_M GGUF file on first use, then reuses the cache
youtube-summariser "https://youtu.be/VIDEO_ID" --local

# Equivalent explicit GGUF repo form
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model weijianzhg/youtube-summariser-qwen3.5-4b-GGUF:Q4_K_M

# Use the larger Q5_K_M GGUF file
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model weijianzhg/youtube-summariser-qwen3.5-4b-GGUF:Q5_K_M

# Or use another compatible GGUF or Transformers Hugging Face model repo
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model owner/model-name
```

```yaml
provider: local
local:
  model_path: weijianzhg/youtube-summariser-qwen3.5-4b-GGUF
  model_file: youtube-summariser-qwen3.5-4b.Q4_K_M.gguf
  max_tokens: 512
  map_max_tokens: 180
  intermediate_max_tokens: 180
  final_max_tokens: 1200
  max_input_tokens: 8192
  n_ctx: 9600
```

The local provider accepts a GGUF Hub repo, a local `.gguf` file, a Hugging Face Transformers repo ID, an extracted Transformers model directory, or a `.tar`/`.tar.gz` archive. Hub models and archives are cached under `~/.cache/youtube-summariser/models`; the default Q4_K_M GGUF file is about 2.5 GiB to download.

Long transcripts are handled according to the selected model's token budget. In the default
`--summary-strategy auto` mode, the CLI uses a single model call when the transcript fits the
selected model context and switches to map-reduce only when it does not. High-context hosted
models such as Claude can stay on the single-shot path for videos that need chunking with the
local Qwen fine-tune. The default local prompt budget is intentionally higher than the original
short-context setting so local map-reduce uses fewer chunks on long videos. Pass
`--summary-strategy single --allow-truncate` only when you
intentionally want a quick partial smoke-test summary.

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


| Command     | Description                                             |
| ----------- | ------------------------------------------------------- |
| `init`      | Interactive setup for API keys and preferences          |
| `summarise` | Summarize a YouTube video (also aliased as `summarize`) |
| `search`    | Search YouTube by title and summarize                   |


You can also pass a URL directly without the `summarise` subcommand for convenience.

### Options


| Flag                 | Description                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `-o, --output`       | Specify output filename (default: `YYYY-MM-DD__video-title-slug__video-id.md`) |
| `--no-save`          | Print summary to terminal without saving to file                       |
| `--provider`         | LLM provider to use: `openai`, `anthropic`, `openrouter`, or `local`   |
| `--local`            | Download and use the default local Qwen GGUF Q4_K_M fine-tune from Hugging Face |
| `--local-model`      | Hugging Face repo ID, local `.gguf` file, Transformers directory, or archive |
| `--summary-strategy` | Summarization strategy: `auto`, `single`, or `map-reduce`              |
| `--allow-truncate`   | Allow an explicit partial smoke-test summary when input exceeds context |
| `--no-stream`        | Disable streaming output                                               |
| `--first, -1`        | Auto-select first search result (search command only)                  |
| `--max-results`      | Number of search results to display (default: 5)                       |
| `-v, --version`      | Show version number                                                    |
| `-h, --help`         | Show help message                                                      |


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
- An API key for OpenAI, Anthropic, or OpenRouter, or local model dependencies installed with `pip install -e ".[local]"`

## License

MIT License - see [LICENSE](LICENSE) for details.
