## Local Model And Long-Video Support

This release adds first-class local summarization for Hugging Face/Transformers models and makes long-video summarization work without silent truncation.

### What's New

- `--local` downloads and uses the public Qwen fine-tune: `weijianzhg/youtube-summariser-qwen3.5-4b`
- `--local-model` accepts any compatible Hugging Face model repo ID, an extracted model directory, or a `.tar`/`.tar.gz` archive
- `--summary-strategy auto` uses one model call when the transcript fits, and switches to map-reduce only when the selected model context requires it
- `--summary-strategy single` now fails clearly on over-budget inputs unless `--allow-truncate` is explicitly passed
- Local model downloads and archive extraction are cached under `~/.cache/youtube-summariser/models`

### Example

```bash
# Use the default public local fine-tune
youtube-summariser "https://youtu.be/VIDEO_ID" --local

# Use another compatible Hugging Face model
youtube-summariser "https://youtu.be/VIDEO_ID" --local-model owner/model-name
```

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
