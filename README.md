# Video Summarizer - User Guide

This application allows you to summarize YouTube videos by generating concise summaries using AI. The app extracts the transcript from YouTube videos and processes it with your choice of LLM provider (OpenAI or Anthropic) to create structured summaries.

**Two interfaces available:**
- **Web UI** — Browser-based form with real-time summary display
- **CLI** — Terminal command that saves summaries to `.txt` files

![Video Summarizer Landing Page](img/landing.png)

## Prerequisites

Before you can run this application, you need to have the following installed:

- Python 3.11 or higher
- pip (Python package manager)
- pipenv (Python virtual environment and package manager)

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/weijianzhg/youtube-summariser
cd youtube-summariser
```

### 2. Install pipenv (if not already installed)

```bash
pip install pipenv
```

### 3. Install Dependencies

The project uses Pipfile for dependency management. To install all dependencies:

```bash
# Install dependencies from Pipfile
pipenv install
```

This will:
- Create a virtual environment if it doesn't exist
- Install all dependencies specified in the Pipfile
- Create/update Pipfile.lock with exact versions

### 4. Activate the Virtual Environment

After installing dependencies, activate the virtual environment:

```bash
pipenv shell
```

### 5. Configure LLM Provider

The application supports both **OpenAI** and **Anthropic (Claude)** models. Configure your preferred provider in `config.yaml`.


### 6. Set Up Environment Variables

Set the API key for your chosen provider:

| Provider | Environment Variable |
|----------|---------------------|
| OpenAI | `OPENAI_API_KEY` |
| Anthropic | `ANTHROPIC_API_KEY` |

Additionally, for the web interface:
- `SESSION_SECRET`: A secret key for Flask sessions

You can set these variables in your terminal:

```bash
# On Windows (PowerShell)
$env:OPENAI_API_KEY="your_openai_api_key"
$env:SESSION_SECRET="your_secret_key"

# For Anthropic instead:
$env:ANTHROPIC_API_KEY="your_anthropic_api_key"

# On macOS/Linux
export OPENAI_API_KEY=your_openai_api_key
export SESSION_SECRET=your_secret_key

# For Anthropic instead:
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

Alternatively, create a `.env` file in the project root directory:

```
# For OpenAI
OPENAI_API_KEY=your_openai_api_key
SESSION_SECRET=your_secret_key

# For Anthropic (comment out OpenAI and uncomment below)
# ANTHROPIC_API_KEY=your_anthropic_api_key
```

### 7. Running the Application

#### Option A: Web Interface

Start the web server:

```bash
python main.py
```

This will start the Flask development server on `http://0.0.0.0:5001/`.

You can access the application by opening a web browser and navigating to:
- `http://localhost:5001/` (if accessing from the same machine)
- `http://your-ip-address:5001/` (if accessing from another device on the same network)

#### Option B: Command-Line Interface (CLI)

Summarize videos directly from the terminal:

```bash
# Basic usage - saves to auto-generated filename
pipenv run python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Custom output filename
pipenv run python cli.py "https://youtu.be/VIDEO_ID" -o my_summary.txt

# Print to terminal only (no file saved)
pipenv run python cli.py "https://youtube.com/watch?v=VIDEO_ID" --no-save
```

**CLI Options:**

| Flag | Description |
|------|-------------|
| `-o, --output` | Specify output filename (default: `summary_<video_id>_<timestamp>.txt`) |
| `--no-save` | Print summary to terminal without saving to file |

**Output file format:**
```
YouTube Video Summary
=====================
Video URL: https://www.youtube.com/watch?v=VIDEO_ID
Video ID: VIDEO_ID
Generated: 2025-12-31 14:30:00
Model: openai / gpt-4o

## Main Topics
...
```

### 8. Using the Web Application

1. Enter a YouTube URL in the input field
2. Click the "Summarize" button
3. Wait for the application to process the video and generate a summary
4. View the structured summary with clickable timestamps

## Troubleshooting

- **API Key Issues**: Ensure your API key (OpenAI or Anthropic) is valid and has sufficient credits
- **"Configuration file not found"**: Make sure `config.yaml` exists in the project root
- **"Invalid YAML in configuration file"**: Check `config.yaml` syntax is valid
- **YouTube Transcript Errors**: Some videos may not have transcripts available or may have disabled transcript access
- **Port Conflicts**: If port 5001 is already in use, you can specify a different port using the `--port` argument
- **Dependency Issues**: If you encounter any dependency-related issues, try:
  ```bash
  pipenv clean
  pipenv install
  ```
