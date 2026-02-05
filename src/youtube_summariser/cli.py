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
import sys
from datetime import datetime

from dotenv import load_dotenv

from . import __version__
from .config_manager import run_init
from .llm_client import LLMClient
from .youtube_helper import YouTubeHelper

load_dotenv()

SYSTEM_PROMPT = """Summarize this video transcript concisely.

## Output Format (use markdown):

### TL;DR
One paragraph capturing the essence (2-3 sentences).

### Key Takeaways
- Bullet points of the most important insights
- Include timestamps like [MM:SS] where relevant

### Detailed Summary
Comprehensive breakdown. Scale length to video complexity (~50 words per 5 minutes of content).

### Notable Quotes
1-3 memorable quotes with timestamps, if any stand out.

Preserve any timestamps from the transcript. Be concise—omit filler and tangents."""


def summarize_transcript(transcript: str, llm: LLMClient, stream: bool = True) -> str:
    """
    Summarize transcript using the configured LLM.

    Args:
        transcript: The video transcript to summarize
        llm: The LLM client instance
        stream: If True, use streaming and print output incrementally

    Returns:
        The complete summary text
    """
    if stream:
        # Use streaming and collect the full response
        summary_parts = []
        print("\n--- Summary ---\n")
        try:
            for chunk in llm.stream_chat(SYSTEM_PROMPT, transcript):
                print(chunk, end="", flush=True)
                summary_parts.append(chunk)
            print("\n")
            return "".join(summary_parts)
        except KeyboardInterrupt:
            print("\n\nSummary generation interrupted by user.")
            return "".join(summary_parts)
    else:
        # Non-streaming fallback
        return llm.chat(SYSTEM_PROMPT, transcript)


def generate_output_filename(video_id: str) -> str:
    """Generate a default output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"summary_{video_id}_{timestamp}.md"


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
        llm = LLMClient(provider=args.provider)
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
    process_video(selected["video_id"], selected["url"], args, llm)


def process_video(
    video_id: str,
    video_url: str,
    args,
    llm: LLMClient,
) -> None:
    """
    Shared logic for processing a video: fetch transcript, summarize, and save.

    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
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
        summary = summarize_transcript(transcript, llm, stream=not args.no_stream)
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
        output_file = args.output or generate_output_filename(video_id)

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
        llm = LLMClient(provider=args.provider)
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

    process_video(video_id, args.url, args, llm)


def add_summarise_args(parser):
    """Add common summarise arguments to a parser."""
    parser.add_argument("url", help="YouTube video URL to summarize")
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (default: summary_<video_id>_<timestamp>.md)",
        default=None,
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print summary to stdout without saving to file"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider to use (overrides config)",
        default=None,
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response before displaying)",
    )


def is_url_like(arg: str) -> bool:
    """Check if an argument looks like a URL."""
    return arg.startswith(("http://", "https://", "www.", "youtube.com", "youtu.be"))


def main():
    # Handle backward compatibility: if first arg looks like a URL, prepend 'summarise'
    if len(sys.argv) > 1 and is_url_like(sys.argv[1]):
        sys.argv.insert(1, "summarise")

    parser = argparse.ArgumentParser(
        prog="youtube-summarizer",
        description="Summarize YouTube videos from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  youtube-summarizer init
  youtube-summarizer "https://www.youtube.com/watch?v=VIDEO_ID"
  youtube-summarizer "https://youtu.be/VIDEO_ID" --output summary.md
  youtube-summarizer "https://youtube.com/watch?v=VIDEO_ID" --provider openai
  youtube-summarizer search "Python tutorial" --first
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
        help="Output filename (default: summary_<video_id>_<timestamp>.md)",
        default=None,
    )
    search_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to stdout without saving to file",
    )
    search_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider to use (overrides config)",
        default=None,
    )
    search_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response before displaying)",
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
