## Channel-wide YouTube summarization

Version 0.7.0 adds a channel workflow that runs the existing transcript and AI
summarization pipeline across multiple videos.

### What's new

- Summarize every video published on a channel:

  ```bash
  youtube-summariser channel "https://www.youtube.com/@CHANNEL"
  ```

- Limit large or expensive runs to the most recent videos:

  ```bash
  youtube-summariser channel "https://www.youtube.com/@CHANNEL" --max-videos 10
  ```

- Store each video's Markdown summary under `channel-summaries/`, or select another
  directory with `--output-dir`.
- Continue past videos with unavailable transcripts or failed summaries, then report
  incomplete video IDs at the end.

The channel command supports the same provider and streaming options as single-video
summarization. Omitting `--max-videos` processes the full channel and may result in a
long-running job with many LLM API calls.

See [CHANGELOG.md](CHANGELOG.md) for the complete change list.
