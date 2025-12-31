#!/usr/bin/env python3
"""
Command-line interface for YouTube Video Summarizer.

Usage:
    python cli.py <youtube_url> [--output filename.txt]

Examples:
    python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    python cli.py "https://youtu.be/dQw4w9WgXcQ" -o my_summary.txt
"""

import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from youtube_helper import YouTubeHelper

load_dotenv()


def summarize_transcript(transcript: str, openai_client: OpenAI) -> str:
    """Summarize transcript using OpenAI API."""
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are a video summarization expert. Create a detailed summary of the video transcript with the following sections:

1. Main Topics: List the key topics discussed (2-3 sentences each)
2. Key Points: Highlight important information with timestamps
3. Detailed Summary: A comprehensive breakdown of the content (300-400 words)
4. Notable Quotes: Include 2-3 significant quotes with their timestamps
5. Timestamps for Important Moments: List key moments in the video

For timestamps in brackets like [MM:SS], maintain them in your response."""
            },
            {"role": "user", "content": transcript}
        ],
        max_tokens=1500
    )
    return response.choices[0].message.content


def generate_output_filename(video_id: str) -> str:
    """Generate a default output filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"summary_{video_id}_{timestamp}.txt"


def main():
    parser = argparse.ArgumentParser(
        description="Summarize YouTube videos from the command line",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"
  python cli.py "https://youtu.be/VIDEO_ID" --output summary.txt
  python cli.py "https://youtube.com/watch?v=VIDEO_ID" -o my_notes.txt
        """
    )
    parser.add_argument(
        "url",
        help="YouTube video URL to summarize"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output filename (default: summary_<video_id>_<timestamp>.txt)",
        default=None
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Print summary to stdout without saving to file"
    )

    args = parser.parse_args()

    # Check for API key
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        print("Set it with: export OPENAI_API_KEY='your-api-key'", file=sys.stderr)
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
        openai_client = OpenAI(api_key=api_key)
        summary = summarize_transcript(transcript, openai_client)
    except Exception as e:
        print(f"Error generating summary: {str(e)}", file=sys.stderr)
        sys.exit(1)

    print("✅ Summary generated")

    # Prepare output content
    output_content = f"""YouTube Video Summary
=====================
Video URL: {args.url}
Video ID: {video_id}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

{summary}
"""

    # Output handling
    if args.no_save:
        print("\n" + "=" * 50)
        print(output_content)
    else:
        output_file = args.output or generate_output_filename(video_id)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(output_content)

        print(f"💾 Summary saved to: {output_file}")
        print("\n" + "=" * 50)
        print(output_content)


if __name__ == "__main__":
    main()

