# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2026-07-28

### Added
- Relationship-ready Obsidian properties for content type, concepts, prerequisites,
  and explicit series position
- Original YouTube publication date in summary frontmatter when metadata is available

### Changed
- Topical tags are normalized and rendered into frontmatter instead of appended as inline hashtags
- Model-provided knowledge graph metadata is parsed from validated JSON and omitted from the note body
- Search loads publication details only for the selected result instead of every candidate
- Default maximum output tokens increased from 3,000 to 6,000 for all providers

### Fixed
- Metadata is extracted even when a model places the hidden block before the summary
- Duplicate model-generated H1 headings are removed from saved notes
- Responses with missing or unfinished knowledge graph metadata are retried once and
  rejected instead of being saved as incomplete notes
- Untrusted video metadata and transcripts remain in the user-role prompt payload

## [0.7.1] - 2026-07-28

### Fixed
- Channel video discovery failing with `a bytes-like object is required, not 'str'` on older `pytubefix` builds
- `extract_video_id` now rejects non-string inputs instead of logging a cryptic parse error

### Changed
- Raised minimum `pytubefix` requirement to `>=10.10.0` for working channel URL extraction
- Channel listing now accepts both URL strings and YouTube objects from `video_urls`, and skips junk entries
- Summary notes are now Obsidian-friendly: YAML frontmatter, video title as H1, clickable timestamp deep links, topical tags, and no invented wikilinks
- Prompt asks the model for Obsidian callouts and `#youtube` tags when useful

### Added
- `format_summary_document` helper for consistent vault-ready markdown output
- Channel name included in frontmatter when available

## [0.7.0] - 2026-07-28

### Added
- `channel` command for summarizing every video from a YouTube channel
- `--max-videos` option for limiting channel runs to the most recent videos
- `--output-dir` option for organizing per-video channel summaries

### Changed
- Batch channel runs continue after individual transcript or summary failures and report them at the end

## [0.6.3] - 2026-04-13

### Added
- Human-searchable default filenames: `YYYY-MM-DD__video-title-slug__video-id.md`
- Video title metadata lookup for URL mode filename generation
- Test coverage for filename formatting and title propagation in both `search` and direct URL flows

### Changed
- Updated CLI help and README to document the new default filename format

### Removed
- Removed committed summary output artifacts from version control

## [0.6.2] - 2026-04-13

### Added
- **Short CLI alias**: Added `ys` as a compact command alias for `youtube-summariser`
  - Example: `ys init` and `ys "https://youtu.be/VIDEO_ID"`
- **Shell usage note**: Added README guidance for running unquoted YouTube URLs in zsh
  - Includes `watch\\?v=` escaping and `noglob ys ...` options

### Changed
- CLI help now uses the invoked command name dynamically (for example, `ys --help` shows `ys` examples)

## [0.6.1] - 2026-02-05

### Added
- **American English CLI alias**: Both `youtube-summarizer` and `youtube-summariser` commands now work
  - Use whichever spelling you prefer after installation
  - The `summarize` / `summarise` subcommand aliases were already supported

## [0.6.0] - 2026-02-03

### Added
- **Search by title**: New `search` command to find and summarize YouTube videos without needing the URL
  - `youtube-summariser search "video title"` - Interactive selection from search results
  - `--first` / `-1` flag to auto-select the first result
  - `--max-results` option to control number of results displayed (default: 5)
  - Shows video title, channel, and duration for each result
- New dependency: `pytubefix>=8.0.0` for YouTube search (no API key required)

### Changed
- Refactored CLI to share video processing logic between `summarise` and `search` commands

## [0.5.0] - 2026-02-02

### Added
- **OpenRouter provider support**: Access 300+ AI models through a unified API
  - Use `--provider openrouter` or set as default in config
  - Models specified in `provider/model-name` format (e.g., `anthropic/claude-sonnet-4.5`)
  - Full streaming support for real-time output
  - Configure via `youtube-summariser init` or `OPENROUTER_API_KEY` environment variable
- OpenRouter added as third option in interactive `init` command
- New dependency: `openrouter>=0.1.0`

### Changed
- Updated error messages to suggest all three providers as alternatives
- `init` command now offers to configure any combination of providers

## [0.4.0] - 2026-02-01

### Added
- **Interactive `init` command**: New `youtube-summariser init` subcommand for easy configuration
  - Guided setup for API keys with secure masked input
  - Select default provider (Anthropic or OpenAI)
  - Configure model settings for each provider
  - Option to set up both providers in one session
- **User configuration file**: Settings stored in platform-appropriate location
  - Windows: `%APPDATA%\youtube-summariser\config.yaml`
  - macOS/Linux: `~/.youtube-summariser/config.yaml`
  - API keys can now be stored in config instead of environment variables
  - Config file takes priority over bundled defaults
  - Environment variables still take priority over config file
- **Subcommand CLI structure**: Commands now organized as subcommands
  - `youtube-summariser init` - Interactive setup
  - `youtube-summariser summarise <url>` - Summarize a video (with `summarize` alias)
  - Backward compatible: `youtube-summariser <url>` still works

### Changed
- **Markdown output format**: Summary files now saved as `.md` with proper markdown formatting
  - Metadata displayed in a clean table format
  - Default filename changed from `.txt` to `.md`
- Improved error messages to suggest running `init` when API keys are missing
- CLI now uses argparse subparsers for better organization

## [0.3.2] - 2026-01-07

### Changed
- Revamped summarization prompt for better output quality:
  - Added TL;DR section for quick scanning
  - Consolidated redundant timestamp sections into Key Takeaways
  - Summary length now scales with video complexity (~50 words per 5 minutes)
  - Explicit markdown formatting for consistent output
  - More concise output—omits filler and tangents

## [0.3.1] - 2026-01-06

### Changed
- Improved response styling for a more professional appearance

## [0.3.0] - 2026-01-02

### Changed
- **Updated to latest AI models**:
  - OpenAI: Now uses GPT-5.2 (upgraded from GPT-4o)
  - Anthropic: Now uses Claude Sonnet 4.5 `claude-sonnet-4-5-20250929` (upgraded from Claude Sonnet 4)
- **Updated SDK dependencies**:
  - `openai>=1.60.0` (upgraded from 1.0.0) for GPT-5.2 support
  - `anthropic>=0.40.0` (upgraded from 0.18.0) for Claude Sonnet 4.5 support

### Technical Details
- Updated default model configurations in `config.yaml`
- Updated fallback defaults in `llm_client.py`
- All existing API calls remain compatible - no breaking changes

## [0.2.0] - 2026-01-02

### Added
- **Streaming support**: Real-time streaming output for both OpenAI and Anthropic models
  - Summary text now appears incrementally as it's generated, providing immediate feedback
  - Streaming is enabled by default for better user experience
  - Added `--no-stream` flag to disable streaming if needed

### Changed
- Default behavior now uses streaming for all LLM requests
- Improved user experience with real-time output display

### Technical Details
- Implemented `stream_chat()` method in `LLMClient` class
- Added streaming support for OpenAI using `chat.completions.create(stream=True)`
- Added streaming support for Anthropic using `messages.stream()` context manager
- Updated CLI to handle streaming output with proper error handling and keyboard interrupt support

## [0.1.0] - 2026-01-01

### Added
- Initial release
- Support for summarizing YouTube videos using OpenAI or Anthropic models
- Command-line interface with multiple output options
- Configuration via YAML file
- Support for both OpenAI and Anthropic API providers
