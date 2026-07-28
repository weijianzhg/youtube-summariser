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
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from . import __version__
from .config_manager import run_init
from .llm_client import LLMClient
from .youtube_helper import YouTubeHelper

load_dotenv()


def build_system_prompt(video_id: str) -> str:
    """Build the summarization prompt for Obsidian-friendly markdown."""
    return f"""Summarize this video transcript concisely for an Obsidian note.

## Output Format (markdown):

Do not include an H1 title — that is added separately.

### TL;DR
One paragraph capturing the essence (2-3 sentences).

### Key Takeaways
- Bullet points of the most important insights
- Where relevant, link timestamps as [MM:SS](https://www.youtube.com/watch?v={video_id}&t=SECONDS)
  (SECONDS = total seconds from the start)

### Detailed Summary
Comprehensive breakdown. Scale length to video complexity (~50 words per 5 minutes of content).

### Notable Quotes
1-3 memorable quotes with linked timestamps, if any stand out.

## Obsidian conventions
- Prefer plain bullets over tables
- You may use Obsidian callouts such as > [!summary] or > [!quote] sparingly
- End with 3-8 topical tags on one line, like #ai #llm #education
- Always include #youtube among the tags
- Do not invent [[wikilinks]]

Preserve meaningful timestamps from the transcript as clickable deep links.
Be concise—omit filler and tangents."""


def yaml_escape(value: str) -> str:
    """Escape a string for inclusion in a double-quoted YAML value."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def format_summary_document(
    *,
    summary: str,
    video_id: str,
    video_url: str,
    video_title: Optional[str],
    llm: LLMClient,
    channel: Optional[str] = None,
    created_at: Optional[datetime] = None,
) -> str:
    """Format a summary as an Obsidian-friendly markdown note with YAML frontmatter."""
    created = created_at or datetime.now()
    title = (video_title or "").strip() or f"YouTube Video {video_id}"
    model = f"{llm.provider}/{llm.get_model()}"

    lines = [
        "---",
        f'title: "{yaml_escape(title)}"',
        f'url: "{yaml_escape(video_url)}"',
        f"video_id: {video_id}",
    ]
    if channel:
        lines.append(f'channel: "{yaml_escape(channel)}"')
    lines.extend(
        [
            f"created: {created.strftime('%Y-%m-%d')}",
            "tags:",
            "  - youtube",
            "  - summary",
            f'model: "{yaml_escape(model)}"',
            "---",
            "",
            f"# {title}",
            "",
            summary.strip(),
            "",
        ]
    )
    return "\n".join(lines)


def summarize_transcript(
    transcript: str,
    llm: LLMClient,
    stream: bool = True,
    video_id: Optional[str] = None,
) -> str:
    """
    Summarize transcript using the configured LLM.

    Args:
        transcript: The video transcript to summarize
        llm: The LLM client instance
        stream: If True, use streaming and print output incrementally
        video_id: Optional YouTube video ID for timestamp deep links

    Returns:
        The complete summary text
    """
    system_prompt = build_system_prompt(video_id or "VIDEO_ID")
    if stream:
        # Use streaming and collect the full response
        summary_parts = []
        print("\n--- Summary ---\n")
        try:
            for chunk in llm.stream_chat(system_prompt, transcript):
                print(chunk, end="", flush=True)
                summary_parts.append(chunk)
            print("\n")
            return "".join(summary_parts)
        except KeyboardInterrupt:
            print("\n\nSummary generation interrupted by user.")
            return "".join(summary_parts)
    else:
        # Non-streaming fallback
        return llm.chat(system_prompt, transcript)


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
    if (
        process_video(
            selected["video_id"],
            selected["url"],
            selected.get("title"),
            args,
            llm,
            channel=selected.get("channel"),
        )
        is False
    ):
        sys.exit(1)


def process_video(
    video_id: str,
    video_url: str,
    video_title: Optional[str],
    args,
    llm: LLMClient,
    channel: Optional[str] = None,
) -> bool:
    """
    Shared logic for processing a video: fetch transcript, summarize, and save.

    Args:
        video_id: YouTube video ID
        video_url: Full YouTube URL
        video_title: Optional video title for filename generation
        args: Parsed CLI arguments (must have no_save and no_stream attributes)
        llm: Initialized LLM client
        channel: Optional channel/author name for Obsidian frontmatter

    Returns:
        True when the video was processed successfully, otherwise False
    """
    print(f"Fetching transcript for {video_id}...")
    try:
        transcript = YouTubeHelper.get_transcript(video_id)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return False

    print(f"Transcript: {len(transcript)} characters")
    print("Generating summary...")
    try:
        summary = summarize_transcript(
            transcript, llm, stream=not args.no_stream, video_id=video_id
        )
    except Exception as e:
        print(f"\nError generating summary: {str(e)}", file=sys.stderr)
        return False

    if args.no_stream:
        print("Done.")

    output_content = format_summary_document(
        summary=summary,
        video_id=video_id,
        video_url=video_url,
        video_title=video_title,
        llm=llm,
        channel=channel,
    )

    # Output handling
    if args.no_save:
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "-" * 50)
            print(output_content)
    else:
        output_file = Path(
            getattr(args, "output", None) or generate_output_filename(video_id, video_title)
        )
        output_dir = getattr(args, "output_dir", None)
        if output_dir:
            output_file = Path(output_dir) / output_file

        try:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(output_content, encoding="utf-8")
        except OSError as e:
            print(f"Error saving summary: {str(e)}", file=sys.stderr)
            return False

        print(f"Saved to {output_file}")
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "-" * 50)
            print(output_content)

    return True


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

    video_title = None
    channel = None
    try:
        metadata = YouTubeHelper.get_video_metadata(video_id)
        video_title = metadata["title"]
        channel = metadata.get("channel") or None
    except Exception:
        # Title lookup is non-critical; filename fallback still includes video ID.
        video_title = None

    if process_video(video_id, args.url, video_title, args, llm, channel=channel) is False:
        sys.exit(1)


def cmd_channel(args):
    """Summarize videos from a YouTube channel."""
    if args.max_videos is not None and args.max_videos < 1:
        print("Error: --max-videos must be at least 1", file=sys.stderr)
        sys.exit(1)

    try:
        llm = LLMClient(provider=args.provider)
        print(f"Using {llm.provider}/{llm.get_model()}")
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if args.max_videos is None:
        print("Finding all videos in the channel...")
    else:
        print(f"Finding the {args.max_videos} most recent channel video(s)...")

    try:
        videos = YouTubeHelper.get_channel_videos(args.url, max_videos=args.max_videos)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print("No videos found in the channel.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(videos)} video(s).")
    succeeded = 0
    failed = []

    for index, video in enumerate(videos, 1):
        video_id = video["video_id"]
        print(f"\n[{index}/{len(videos)}] Processing {video_id}")

        try:
            metadata = YouTubeHelper.get_video_metadata(video_id)
            video_title = metadata["title"]
            print(f"Title: {video_title}")
            channel = metadata.get("channel") or video.get("channel")
        except Exception:
            video_title = None
            channel = video.get("channel")

        if process_video(
            video_id,
            video["url"],
            video_title,
            args,
            llm,
            channel=channel,
        ):
            succeeded += 1
        else:
            failed.append(video_id)

    print(f"\nChannel complete: {succeeded} succeeded, {len(failed)} failed.")
    if failed:
        print(f"Failed video IDs: {', '.join(failed)}", file=sys.stderr)
        sys.exit(1)


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
  {prog_name} search "Python tutorial" --first
  {prog_name} channel "https://youtube.com/@CHANNEL" --max-videos 10
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

    # Channel subcommand
    channel_parser = subparsers.add_parser(
        "channel", help="Summarize videos from a YouTube channel"
    )
    channel_parser.add_argument("url", help="YouTube channel URL")
    channel_parser.add_argument(
        "--max-videos",
        type=int,
        default=None,
        help="Maximum number of recent videos to summarize (default: all)",
    )
    channel_parser.add_argument(
        "--output-dir",
        default="channel-summaries",
        help="Directory for summary files (default: channel-summaries)",
    )
    channel_parser.add_argument(
        "--no-save", action="store_true", help="Print summaries to stdout without saving files"
    )
    channel_parser.add_argument(
        "--provider",
        choices=["openai", "anthropic", "openrouter"],
        help="LLM provider to use (overrides config)",
        default=None,
    )
    channel_parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for each complete response before displaying)",
    )
    channel_parser.set_defaults(func=cmd_channel)

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
