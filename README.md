# YouTube Summariser

A command-line tool that summarizes YouTube videos using AI. It extracts transcripts from YouTube videos and generates structured summaries using OpenAI or Anthropic models.

## Installation

```bash
pip install youtube-summariser
```

Or install from source:

```bash
git clone https://github.com/weijianzhg/youtube-summariser
cd youtube-summariser
pip install -e .
```

## Configuration

### API Keys

Set your API key for your preferred provider:

```bash
# For OpenAI
export OPENAI_API_KEY=your_openai_api_key

# For Anthropic
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

Or create a `.env` file in your working directory:

```
OPENAI_API_KEY=your_openai_api_key
# or
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Default Provider

The default provider is **Anthropic**. You can override this per command using `--provider`.

## Usage

```bash
# Basic usage - saves summary to auto-generated filename
youtube-summariser "https://www.youtube.com/watch?v=VIDEO_ID"

# Specify output filename
youtube-summariser "https://youtu.be/VIDEO_ID" -o my_summary.txt

# Print to terminal only (no file saved)
youtube-summariser "https://youtube.com/watch?v=VIDEO_ID" --no-save

# Use a specific provider
youtube-summariser "https://youtu.be/VIDEO_ID" --provider openai
```

### Options

| Flag | Description |
|------|-------------|
| `-o, --output` | Specify output filename (default: `summary_<video_id>_<timestamp>.txt`) |
| `--no-save` | Print summary to terminal without saving to file |
| `--provider` | LLM provider to use: `openai` or `anthropic` |
| `-v, --version` | Show version number |
| `-h, --help` | Show help message |

### Output Format

```
YouTube Video Summary
=====================
Video URL: https://www.youtube.com/watch?v=VIDEO_ID
Video ID: VIDEO_ID
Generated: 2025-01-01 14:30:00
Model: anthropic / claude-sonnet-4-5-20250929

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
- An API key for OpenAI or Anthropic

## License

MIT License - see [LICENSE](LICENSE) for details.
