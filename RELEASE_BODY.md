## Channel discovery fix and Obsidian-ready notes

Version 0.7.1 fixes channel video discovery on current YouTube/pytubefix builds and
makes summary markdown ready to drop into an Obsidian vault.

### What's fixed

- `ys channel` no longer fails with `a bytes-like object is required, not 'str'`
  when older `pytubefix` builds return non-string channel entries
- Minimum `pytubefix` raised to `>=10.10.0` for working channel URL extraction

### What's improved

- Summary notes now include YAML frontmatter (`title`, `url`, `video_id`, `channel`,
  `created`, `tags`, `model`)
- Video title is used as the note H1
- The summarization prompt asks for clickable YouTube timestamp deep links,
  topical tags (including `#youtube`), and optional Obsidian callouts
- Invented `[[wikilinks]]` are discouraged so notes stay vault-friendly

Example channel usage:

```bash
youtube-summariser channel "https://www.youtube.com/@CHANNEL" --max-videos 10
```

Point `--output-dir` at an Obsidian vault folder to save notes in place.

See [CHANGELOG.md](CHANGELOG.md) for the complete change list.
