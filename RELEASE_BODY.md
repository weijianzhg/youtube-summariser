## Local Model And Long-Video Support

This release adds first-class local summarization for GGUF/llama.cpp and Hugging Face/Transformers models, and makes long-video summarization work without silent truncation.

### What's New

- `--local` downloads and uses the public Qwen Q4_K_M GGUF fine-tune: `weijianzhg/youtube-summariser-qwen3.5-4b-GGUF`
- `--local-model` accepts compatible GGUF repos, local `.gguf` files, Hugging Face Transformers repo IDs, extracted model directories, or `.tar`/`.tar.gz` archives
- `--summary-strategy auto` uses one model call when the transcript fits, and switches to map-reduce only when the selected model context requires it
- `--summary-strategy single` now fails clearly on over-budget inputs unless `--allow-truncate` is explicitly passed
- Local map-reduce uses compact map/intermediate summaries with a larger final synthesis budget, which is better suited to hour-long videos on the Q4 GGUF model
- Local model downloads and archive extraction are cached under `~/.cache/youtube-summariser/models`

### Example

```bash
# Use the default public local fine-tune
youtube-summariser "https://youtu.be/VIDEO_ID" --local

# Use the Q5_K_M GGUF variant
youtube-summariser "https://youtu.be/VIDEO_ID" \
  --local-model weijianzhg/youtube-summariser-qwen3.5-4b-GGUF:Q5_K_M
```

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
