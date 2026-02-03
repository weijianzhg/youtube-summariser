## Search by Title - No URL Required

This release adds the ability to search for YouTube videos by title, so you no longer need to copy/paste URLs.

### What's New

#### Search Command
Use `youtube-summariser search` to find and summarize videos by title:

```bash
# Interactive selection (shows top 5 results)
youtube-summariser search "How to make pasta"

# Output:
# Found 5 video(s):
#
#   1. How to Make Fresh Pasta | Basics with Babish
#      Channel: Babish Culinary Universe | Duration: 12:34
#
#   2. Gordon Ramsay's Ultimate Guide to Pasta
#      Channel: Gordon Ramsay | Duration: 18:45
#   ...
#
# Select video (1-5): 1
```

#### Auto-Select First Result
Skip the selection prompt with `--first`:

```bash
youtube-summariser search "Python tutorial for beginners" --first
```

#### Control Results Count
Show more or fewer results with `--max-results`:

```bash
youtube-summariser search "cooking recipes" --max-results 10
```

### Why Search?
- **No URL copying**: Just type what you're looking for
- **No API key needed**: Uses pytubefix library (scraping-based)
- **Interactive selection**: See title, channel, and duration before choosing
- **Works with all providers**: Combine with `--provider openai/anthropic/openrouter`

### Full Example

```bash
$ youtube-summariser search "machine learning explained" --first --provider anthropic

Using anthropic/claude-sonnet-4-5-20250929
Searching YouTube for: machine learning explained
Auto-selecting: Machine Learning Explained in 100 Seconds

Selected: Machine Learning Explained in 100 Seconds
URL: https://www.youtube.com/watch?v=...

Fetching transcript for ...
Transcript: 1842 characters
Generating summary...

--- Summary ---
...
```

### Requirements

- Python 3.10+
- pytubefix >= 8.0.0 (new dependency, no API key required)

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
