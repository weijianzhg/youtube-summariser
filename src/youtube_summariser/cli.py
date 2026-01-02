#!/usr/bin/env python3
"""
Command-line interface for YouTube Video Summariser.

Usage:
    youtube-summariser <youtube_url> [--output filename.txt]

Examples:
    youtube-summariser "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    youtube-summariser "https://youtu.be/dQw4w9WgXcQ" -o my_summary.txt
"""

import argparse
import sys
from datetime import datetime

from dotenv import load_dotenv

from . import __version__
from .llm_client import LLMClient
from .youtube_helper import YouTubeHelper

load_dotenv()

SYSTEM_PROMPT = """You are a video summarization expert. Create a detailed summary of the \
video transcript with the following sections:

1. Main Topics: List the key topics discussed (2-3 sentences each)
2. Key Points: Highlight important information with timestamps
3. Detailed Summary: A comprehensive breakdown of the content (300-400 words)
4. Notable Quotes: Include 2-3 significant quotes with their timestamps
5. Timestamps for Important Moments: List key moments in the video

For timestamps in brackets like [MM:SS], maintain them in your response."""


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
        print("\n📝 Summary:\n")
        print("=" * 50)
        try:
            for chunk in llm.stream_chat(SYSTEM_PROMPT, transcript):
                print(chunk, end="", flush=True)
                summary_parts.append(chunk)
            print("\n" + "=" * 50 + "\n")
            return "".join(summary_parts)
        except KeyboardInterrupt:
            print("\n\n⚠️  Summary generation interrupted by user")
            return "".join(summary_parts)
    else:
        # Non-streaming fallback
        return llm.chat(SYSTEM_PROMPT, transcript)


def generate_output_filename(video_id: str) -> str:
    """Generate a default output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"summary_{video_id}_{timestamp}.txt"


def main():
    parser = argparse.ArgumentParser(
        prog="youtube-summariser",
        description="Summarize YouTube videos from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  youtube-summariser "https://www.youtube.com/watch?v=VIDEO_ID"
  youtube-summariser "https://youtu.be/VIDEO_ID" --output summary.txt
  youtube-summariser "https://youtube.com/watch?v=VIDEO_ID" -o my_notes.txt
        """,
    )
    parser.add_argument("url", nargs="?", help="YouTube video URL to summarize")
    parser.add_argument(
        "-o",
        "--output",
        help="Output filename (default: summary_<video_id>_<timestamp>.txt)",
        default=None,
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Print summary to stdout without saving to file"
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider to use (overrides config.yaml)",
        default=None,
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming output (wait for complete response before displaying)",
    )
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {__version__}")

    args = parser.parse_args()

    # Check if URL was provided
    if not args.url:
        parser.print_help()
        sys.exit(0)

    # Initialize LLM client
    print("🔧 Initializing LLM client...")
    try:
        llm = LLMClient(provider=args.provider)
        print(f"✅ Using {llm.provider} with model {llm.get_model()}")
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Validate URL
    print(f"🔗 Processing URL: {args.url}")

    if not YouTubeHelper.validate_url(args.url):
        print("Error: Invalid YouTube URL", file=sys.stderr)
        sys.exit(1)

    # Extract video ID
    video_id = YouTubeHelper.extract_video_id(args.url)
    if not video_id:
        print("Error: Could not extract video ID from URL", file=sys.stderr)
        sys.exit(1)

    print(f"📺 Video ID: {video_id}")

    # Fetch transcript
    print("📝 Fetching transcript...")
    try:
        transcript = YouTubeHelper.get_transcript(video_id)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print(f"✅ Transcript fetched ({len(transcript)} characters)")

    # Generate summary
    print("🤖 Generating AI summary...")
    try:
        summary = summarize_transcript(transcript, llm, stream=not args.no_stream)
    except Exception as e:
        print(f"\nError generating summary: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if args.no_stream:
        print("✅ Summary generated")

    # Prepare output content for file saving
    output_content = f"""YouTube Video Summary
=====================
Video URL: {args.url}
Video ID: {video_id}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Model: {llm.provider} / {llm.get_model()}

{summary}
"""

    # Output handling
    if args.no_save:
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "=" * 50)
            print(output_content)
    else:
        output_file = args.output or generate_output_filename(video_id)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_content)

        print(f"💾 Summary saved to: {output_file}")
        if args.no_stream:
            # Only print full formatted output if we haven't already streamed it
            print("\n" + "=" * 50)
            print(output_content)


if __name__ == "__main__":
    main()
