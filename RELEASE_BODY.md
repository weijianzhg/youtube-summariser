## Relationship-ready Obsidian summaries

Version 0.8.0 makes generated notes easier to connect, filter, and visualize in
Obsidian without repeating metadata extraction during post-processing.

### What's improved

- Summary frontmatter now includes normalized topic tags, content type, concepts,
  prerequisites, and explicit series position when available
- Official YouTube publication dates are preserved alongside title and channel metadata
- The official title and channel are provided to the model as classification context
- Model-generated graph metadata is validated before being rendered into YAML
- Machine-readable metadata is removed from the final note body
- Topical tags are emitted as canonical frontmatter tags instead of trailing hashtags
- Search fetches publication details only after a result is selected

### Reliability

- Invalid graph metadata is ignored without interrupting summary generation
- Misplaced graph metadata and duplicate model-generated titles are cleaned automatically
- Incomplete responses are retried once and never saved as empty notes
- Uploader-controlled title, channel, and transcript text remain untrusted user-role data
- Visual details that are absent from a transcript are not inferred
- Existing summary generation continues to work when optional metadata is unavailable

Obsidian Graph View still requires a vault-wide indexing step to turn the structured
concept properties into shared concept notes and `[[wikilinks]]`.

See [CHANGELOG.md](CHANGELOG.md) for the complete change list.
