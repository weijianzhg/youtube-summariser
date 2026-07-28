## Relationship-ready Obsidian summaries

Version 0.8.0 makes generated notes easier to connect, filter, and visualize in
Obsidian without a large post-processing step.

### What's improved

- Summary frontmatter now includes normalized topic tags, content type, concepts,
  prerequisites, and explicit series position when available
- Official YouTube publication dates are preserved alongside title and channel metadata
- The official title and channel are provided to the model as classification context
- Model-generated graph metadata is validated before being rendered into YAML
- Machine-readable metadata is removed from the final note body
- Topical tags are emitted as canonical frontmatter tags instead of trailing hashtags

### Reliability

- Invalid graph metadata is ignored without interrupting summary generation
- Visual details that are absent from a transcript are not inferred
- Existing summary generation continues to work when optional metadata is unavailable

See [CHANGELOG.md](CHANGELOG.md) for the complete change list.
