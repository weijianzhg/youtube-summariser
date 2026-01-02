# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

