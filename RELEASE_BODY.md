## 🚀 Latest AI Models Update

This release updates to the **latest and most powerful AI models** from both OpenAI and Anthropic, bringing enhanced capabilities and performance improvements.

### ✨ What's New

#### Updated AI Models
- **OpenAI GPT-5.2**: Upgraded from GPT-4o to the latest GPT-5.2 (Thinking variant)
  - Enhanced reasoning capabilities with new `xhigh` effort level
  - Improved multimodal capabilities, especially in vision
  - Better code generation for front-end UI
  - Advanced tool calling and context management

- **Anthropic Claude Sonnet 4.5**: Upgraded from Claude Sonnet 4 to 4.5
  - 1M token context window support
  - Advanced tool use capabilities (beta)
  - Improved code understanding and execution

#### Updated Dependencies
- `openai>=1.60.0` (upgraded from 1.0.0)
- `anthropic>=0.40.0` (upgraded from 0.18.0)

### 🔄 Migration

**No breaking changes!** All existing commands and configurations work exactly as before. The new models are drop-in replacements with enhanced capabilities.

To use the new models, simply upgrade:
```bash
pip install --upgrade youtube-summariser
```

Or if installed from source:
```bash
pip install --upgrade openai anthropic
```

### 📦 Requirements

- Python 3.10+
- OpenAI SDK >= 1.60.0
- Anthropic SDK >= 0.40.0

### 🎯 Model Configuration

The default models are now:
- **OpenAI**: `gpt-5.2`
- **Anthropic**: `claude-sonnet-4-5-20250929`

You can customize the model by editing the `config.yaml` file in the package.

See [CHANGELOG.md](CHANGELOG.md) for detailed technical changes.
