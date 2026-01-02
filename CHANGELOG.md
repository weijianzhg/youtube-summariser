# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

