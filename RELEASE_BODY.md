## American English Support

This release adds support for the American English spelling of the CLI command.

### What's New

Both spellings now work after installation:

```bash
# American English
youtube-summarizer "https://youtube.com/watch?v=VIDEO_ID"

# British English  
youtube-summariser "https://youtube.com/watch?v=VIDEO_ID"
```

Use whichever you prefer - they're completely identical!

### Why?

- **Broader appeal**: Many users expect the American spelling
- **No confusion**: Both work the same way
- **Already supported**: The `summarize` / `summarise` subcommands were already aliased

See [CHANGELOG.md](CHANGELOG.md) for detailed changes.
