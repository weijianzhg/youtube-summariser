## 🎉 Streaming Support Release

This release introduces **real-time streaming support** for both OpenAI and Anthropic models, providing instant feedback as summaries are generated.

### ✨ What's New

#### Streaming Output (Default)
- **Real-time feedback**: Summary text appears incrementally as it's generated
- **Better UX**: See progress immediately instead of waiting
- **Universal support**: Works with both OpenAI and Anthropic models

#### New Command-Line Option
- `--no-stream`: Disable streaming if you prefer to wait for complete response

### 📝 Usage

```bash
# Streaming enabled by default
youtube-summariser "https://www.youtube.com/watch?v=VIDEO_ID"

# Disable streaming (previous behavior)
youtube-summariser "https://www.youtube.com/watch?v=VIDEO_ID" --no-stream
```

### 🔄 Migration

No breaking changes! Existing commands work as before, streaming is just enabled by default.

### 📦 Requirements

- OpenAI SDK >= 1.0.0
- Anthropic SDK >= 0.18.0

See [CHANGELOG.md](CHANGELOG.md) for detailed technical changes.

