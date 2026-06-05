#!/usr/bin/env python3
"""
Command-line interface for YouTube Video Summariser.

Both 'youtube-summariser' (British) and 'youtube-summarizer' (American) work.

Usage:
    youtube-summarizer <youtube_url> [--output filename.md]
    youtube-summarizer search <query> [--first]
    youtube-summarizer init

Examples:
    youtube-summarizer "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    youtube-summarizer "https://youtu.be/dQw4w9WgXcQ" -o my_summary.md
    youtube-summarizer search "Python tutorial" --first
    youtube-summarizer init
"""

import argparse
import copy
import os
import re
import sys
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

from . import __version__
from .config_manager import run_init
from .llm_client import SUPPORTED_PROVIDERS, LLMClient, load_config
from .local_llm import DEFAULT_GGUF_MODEL_FILE, DEFAULT_HF_MODEL_ID
from .summarizer import SYSTEM_PROMPT as SYSTEM_PROMPT
from .summarizer import summarize_transcript as summarize_transcript
from .youtube_helper import YouTubeHelper

load_dotenv()


def slugify_filename_component(value: str, max_length: int = 80) -> str:
    """Convert text into a filesystem-safe slug for filenames."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:max_length].rstrip("-")


def generate_output_filename(video_id: str, video_title: Optional[str] = None) -> str:
    """Generate a default date-first output filename."""
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    title_slug = slugify_filename_component(video_title or "")
    if title_slug:
        return f"{date_prefix}__{title_slug}__{video_id}.md"
    return f"{date_prefix}__video-{video_id}.md"


def cmd_init(args):
    """Handle the init subcommand."""
    run_init()


def format_duration(seconds: str) -> str:
    """Convert duration in seconds to human-readable MM:SS or HH:MM:SS format."""
    try:
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        if hours > 0:
            return f"{hours}:{minutes:02d}:{secs:02d}"
        return f"{minutes}:{secs:02d}"
    except (ValueError, TypeError):
        return "??:??"


def cmd_search(args):
    """Handle the search subcommand."""
    # Initialize LLM client first
    try:
        llm = create_llm_from_args(args)
        print(f"Using {llm.provider}/{llm.get_model()}")
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Search for videos
    print(f"Searching YouTube for: {args.query}")
    try:
        results = YouTubeHelper.search_videos(args.query, max_results=args.max_results)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if not results:
        print("No videos found matching your query.", file=sys.stderr)
        sys.exit(1)

    # Select video
    if args.first:
        # Auto-select first result
        selected = results[0]
        print(f"Auto-selecting: {selected['title']}")
    else:
        # Display results and let user pick
        print(f"\nFound {len(results)} video(s):\n")
        for i, video in enumerate(results, 1):
            duration = format_duration(video["duration"])
            print(f"  {i}. {video['title']}")
            print(f"     Channel: {video['channel']} | Duration: {duration}")
            print()

        # Prompt user for selection
        while True:
            try:
                choice = input(f"Select video (1-{len(results)}): ").strip()
                if not choice:
                    print("Cancelled.")
                    sys.exit(0)
                idx = int(choice) - 1
                if 0 <= idx < len(results):
                    selected = results[idx]
                    break
                print(f"Please enter a number between 1 and {len(results)}")
            except ValueError:
                print("Please enter a valid number")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled.")
                sys.exit(0)

    print(f"\nSelected: {selected['title']}")
    print(f"URL: {selected['url']}\n")

    # Process the selected video
    process_video(selected["video_id"], selected["url"], selected.get("title"), args, llm)


def process_video(
    video_id: str,
    video_url: str,
    video_title: Optional[str],
    args,
    llm: LLMClient,
) -> None:
    """
    Shared logic for processing a video: fetch transcript, summarize, and save.

    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
        video_title: Optional video title for filename generation
        args: Parsed CLI arguments (must have output, no_save, no_stream attributes)
        llm: Initialized LLM client
    """
    print(f"Fetching transcript for {video_id}...")
    try:
        transcript = YouTubeHelper.get_transcript(video_id)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"Transcript: {len(transcript)} characters")
    print("Generating summary...")
    try:
        summary = summarize_transcript(
            transcript,
            llm,
            stream=not args.no_stream,
            summary_strategy=getattr(args, "summary_strategy", "auto"),
            allow_truncate=getattr(args, "allow_truncate", False),
        )
    except Exception as e:
        print(f"\nError generating summary: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if args.no_stream:
        print("Done.")

    # Prepare output content for file saving (markdown format)
    output_content = f"""# YouTube Video Summary

| | |
|---|---|
| **Video URL** | <{video_url}> |
| **Video ID** | `{video_id}` |
| **Generated** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Model** | {llm.provider} / {llm.get_model()} |

---

{summary}
"""

    # Output handling
    if args.no_save:
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "-" * 50)
            print(output_content)
    else:
        output_file = args.output or generate_output_filename(video_id, video_title)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_content)

        print(f"Saved to {output_file}")
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "-" * 50)
            print(output_content)


def cmd_summarise(args):
    """Handle the summarise subcommand (or direct URL usage)."""
    # Initialize LLM client
    try:
        llm = create_llm_from_args(args)
        print(f"Using {llm.provider}/{llm.get_model()}")
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Validate URL
    if not YouTubeHelper.validate_url(args.url):
        print("Error: Invalid YouTube URL", file=sys.stderr)
        sys.exit(1)

    # Extract video ID
    video_id = YouTubeHelper.extract_video_id(args.url)
    if not video_id:
        print("Error: Could not extract video ID from URL", file=sys.stderr)
        sys.exit(1)

    video_title = None
    try:
        video_title = YouTubeHelper.get_video_title(video_id)
    except Exception:
        # Title lookup is non-critical; filename fallback still includes video ID.
        video_title = None

    process_video(video_id, args.url, video_title, args, llm)


def create_llm_from_args(args) -> LLMClient:
    """Create an LLM client, applying CLI-only local model overrides."""
    local_model = getattr(args, "local_model", None)
    use_default_local = bool(getattr(args, "local", False))
    if (local_model or use_default_local) and args.provider and args.provider != "local":
        raise ValueError("--local and --local-model can only be used with the local provider.")

    provider = args.provider or ("local" if local_model or use_default_local else None)
    config = None
    if local_model or use_default_local:
        config = copy.deepcopy(load_config())
        local_config = config.setdefault("local", {})
        local_config["model_path"] = local_model or DEFAULT_HF_MODEL_ID
        if local_model:
            local_config.pop("model_file", None)
        elif use_default_local:
            local_config["model_file"] = DEFAULT_GGUF_MODEL_FILE
    if config is None:
        return LLMClient(provider=provider)
    return LLMClient(config=config, provider=provider)


def add_summarise_args(parser):
    """Add common summarise arguments to a parser."""
    parser.add_argument("url", help="YouTube video URL to summarize")
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (default: YYYY-MM-DD__video-title-slug__video-id.md)",
        default=None,
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print summary to stdout without saving to file"
    )
    parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider to use (overrides config)",
        default=None,
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Use the default local Qwen GGUF Q4_K_M model from Hugging Face "
            f"({DEFAULT_HF_MODEL_ID})"
        ),
    )
    parser.add_argument(
        "--local-model",
        help="Path, .gguf file, .tar/.tar.gz archive, or Hugging Face repo ID for a local model",
        default=None,
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response before displaying)",
    )
    parser.add_argument(
        "--summary-strategy",
        choices=["auto", "single", "map-reduce"],
        default="auto",
        help="Summarization strategy when the selected model context is exceeded (default: auto)",
    )
    parser.add_argument(
        "--allow-truncate",
        action="store_true",
        help="Allow explicit partial smoke-test summaries when input exceeds context",
    )


def is_url_like(arg: str) -> bool:
    """Check if an argument looks like a URL."""
    return arg.startswith(("http://", "https://", "www.", "youtube.com", "youtu.be"))


def main():
    # Handle backward compatibility: if first arg looks like a URL, prepend 'summarise'
    if len(sys.argv) > 1 and is_url_like(sys.argv[1]):
        sys.argv.insert(1, "summarise")

    prog_name = os.path.basename(sys.argv[0]) or "youtube-summarizer"

    parser = argparse.ArgumentParser(
        prog=prog_name,
        description="Summarize YouTube videos from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  {prog_name} init
  {prog_name} "https://www.youtube.com/watch?v=VIDEO_ID"
  {prog_name} "https://youtu.be/VIDEO_ID" --output summary.md
  {prog_name} "https://youtube.com/watch?v=VIDEO_ID" --provider openai
  {prog_name} "https://youtube.com/watch?v=VIDEO_ID" --local
  {prog_name} "https://youtube.com/watch?v=VIDEO_ID" --local-model owner/model-name
  {prog_name} "https://youtube.com/watch?v=VIDEO_ID" --local-model ./downloads/model.tar.gz
  {prog_name} search "Python tutorial" --first
        """,
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    # Init subcommand
    init_parser = subparsers.add_parser(
        "init", help="Configure API keys and default settings interactively"
    )
    init_parser.set_defaults(func=cmd_init)

    # Summarise subcommand (explicit)
    summarise_parser = subparsers.add_parser(
        "summarise", help="Summarize a YouTube video", aliases=["summarize"]
    )
    add_summarise_args(summarise_parser)
    summarise_parser.set_defaults(func=cmd_summarise)

    # Search subcommand
    search_parser = subparsers.add_parser("search", help="Search YouTube by title and summarize")
    search_parser.add_argument("query", help="Search query (video title or keywords)")
    search_parser.add_argument(
        "--first",
        "-1",
        action="store_true",
        help="Auto-select first search result without prompting",
    )
    search_parser.add_argument(
        "--max-results",
        type=int,
        default=5,
        help="Number of search results to display (default: 5)",
    )
    search_parser.add_argument(
        "-o",
        "--output",
        help="Output filename (default: YYYY-MM-DD__video-title-slug__video-id.md)",
        default=None,
    )
    search_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to stdout without saving to file",
    )
    search_parser.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        help="LLM provider to use (overrides config)",
        default=None,
    )
    search_parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Use the default local Qwen GGUF Q4_K_M model from Hugging Face "
            f"({DEFAULT_HF_MODEL_ID})"
        ),
    )
    search_parser.add_argument(
        "--local-model",
        help="Path, .gguf file, .tar/.tar.gz archive, or Hugging Face repo ID for a local model",
        default=None,
    )
    search_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response before displaying)",
    )
    search_parser.add_argument(
        "--summary-strategy",
        choices=["auto", "single", "map-reduce"],
        default="auto",
        help="Summarization strategy when the selected model context is exceeded (default: auto)",
    )
    search_parser.add_argument(
        "--allow-truncate",
        action="store_true",
        help="Allow explicit partial smoke-test summaries when input exceeds context",
    )
    search_parser.set_defaults(func=cmd_search)

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
