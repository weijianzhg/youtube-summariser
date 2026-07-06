"""Channel summarization: summarize the latest videos of a YouTube channel."""

import glob
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .llm_client import LLMClient
from .youtube_helper import YouTubeHelper

# Rough estimation constants (spoken content is ~150 words/min ≈ 200 tokens/min,
# plus timestamps and prompt overhead).
TOKENS_PER_VIDEO_MINUTE = 225
PROMPT_OVERHEAD_TOKENS = 300
OUTPUT_TOKENS_PER_VIDEO = 800
FALLBACK_VIDEO_MINUTES = 15  # used when a video's duration is unknown
SECONDS_PER_VIDEO_ESTIMATE = (20, 45)

# USD per million tokens (input, output) for common models, matched by substring.
MODEL_PRICES_PER_MTOK = {
    "claude-opus": (5.0, 25.0),
    "claude-sonnet": (3.0, 15.0),
    "claude-haiku": (1.0, 5.0),
    "anthropic/claude-opus": (5.0, 25.0),
    "anthropic/claude-sonnet": (3.0, 15.0),
    "anthropic/claude-haiku": (1.0, 5.0),
}

CHANNEL_SUMMARY_PROMPT = """You are given summaries of the most recent videos from a YouTube
channel. Write an overall channel summary in markdown with these sections:

### Channel Overview
2-3 sentences on what the channel covers, based on these videos.

### Recurring Themes
Bullet points of topics or ideas that appear across multiple videos.

### Video Highlights
One line per video: the title (as given) and its single most important takeaway.

### Who Should Watch
1-2 sentences on the audience that would benefit most.

Base everything strictly on the provided summaries."""

# Keep the roll-up input well within typical context limits.
MAX_ROLLUP_INPUT_CHARS = 300_000


def normalize_channel_url(channel: str) -> str:
    """Turn a handle or partial channel reference into a full YouTube URL."""
    channel = channel.strip()
    if channel.startswith(("http://", "https://")):
        return channel
    if channel.startswith("@"):
        return f"https://www.youtube.com/{channel}"
    if channel.startswith(("youtube.com", "www.youtube.com")):
        return f"https://{channel}"
    # Bare name: assume it's a handle
    return f"https://www.youtube.com/@{channel}"


def default_output_dir(channel_name: str, slugify) -> str:
    """Build the default output folder name for a channel run."""
    date_prefix = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(channel_name) or "channel"
    return f"{date_prefix}__{slug}"


def estimate_run(videos: List[Dict], model: str) -> Dict:
    """
    Estimate token usage, cost, and wall-clock time for summarizing videos.

    Returns a dict with input_tokens, output_tokens, cost (None if the model's
    pricing is unknown), and (min_seconds, max_seconds).
    """
    input_tokens = 0
    for video in videos:
        try:
            minutes = int(video.get("duration") or 0) / 60
        except (TypeError, ValueError):
            minutes = 0
        if minutes <= 0:
            minutes = FALLBACK_VIDEO_MINUTES
        input_tokens += int(minutes * TOKENS_PER_VIDEO_MINUTE) + PROMPT_OVERHEAD_TOKENS
    output_tokens = OUTPUT_TOKENS_PER_VIDEO * len(videos)

    cost = None
    for prefix, (price_in, price_out) in MODEL_PRICES_PER_MTOK.items():
        if model.startswith(prefix):
            cost = (input_tokens * price_in + output_tokens * price_out) / 1_000_000
            break

    lo, hi = SECONDS_PER_VIDEO_ESTIMATE
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "time_range_seconds": (lo * len(videos), hi * len(videos)),
    }


def find_existing_summary(output_dir: str, video_id: str) -> Optional[str]:
    """Return the path of an existing summary file for video_id, if any."""
    matches = sorted(glob.glob(os.path.join(output_dir, f"*__{video_id}.md")))
    return matches[0] if matches else None


def extract_summary_body(file_content: str) -> str:
    """Strip the metadata header from a saved summary file, keeping the summary."""
    marker = "\n---\n"
    idx = file_content.find(marker)
    if idx != -1:
        return file_content[idx + len(marker) :].strip()
    return file_content.strip()


def collect_stream(llm: LLMClient, system_prompt: str, user_message: str) -> str:
    """Run a streaming chat and collect the full response without printing chunks."""
    return "".join(llm.stream_chat(system_prompt, user_message))


def format_seconds(seconds: int) -> str:
    """Format a duration in seconds as a compact human-readable string."""
    minutes = seconds // 60
    if minutes >= 60:
        return f"{minutes // 60}h {minutes % 60}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{seconds}s"


def confirm_run(videos: List[Dict], skipped: int, estimate: Dict, model: str, yes: bool) -> bool:
    """Print the cost/time warning and ask the user to confirm. Returns True to proceed."""
    lo, hi = estimate["time_range_seconds"]
    print(f"\nAbout to summarize {len(videos)} video(s).")
    if skipped:
        print(f"({skipped} video(s) already summarized in the output folder will be skipped.)")
    tokens_in, tokens_out = estimate["input_tokens"], estimate["output_tokens"]
    print(f"  Estimated tokens: ~{tokens_in:,} in / ~{tokens_out:,} out")
    if estimate["cost"] is not None:
        print(f"  Estimated cost:   ~${estimate['cost']:.2f} ({model})")
    else:
        print(f"  Estimated cost:   unknown for model '{model}' (token estimate above)")
    print(f"  Estimated time:   {format_seconds(lo)} - {format_seconds(hi)}")
    print(
        "\nWarning: this makes one LLM request per video and may take a while.\n"
        "You can interrupt at any time (Ctrl-C) — completed summaries are kept,\n"
        "and re-running the same command resumes where it left off."
    )
    if yes:
        return True
    try:
        answer = input("\nProceed? [y/N]: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return False
    return answer in ("y", "yes")


def build_video_summary_file(
    video: Dict, summary: str, channel_name: str, llm: LLMClient
) -> str:
    """Render the markdown content for a single video summary file."""
    return f"""# YouTube Video Summary

| | |
|---|---|
| **Title** | {video.get("title") or "Unknown"} |
| **Channel** | {channel_name} |
| **Video URL** | <{video["url"]}> |
| **Video ID** | `{video["video_id"]}` |
| **Generated** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Model** | {llm.provider} / {llm.get_model()} |

---

{summary}
"""


def build_channel_summary_file(
    channel_name: str,
    channel_url: str,
    summaries: List[Tuple[Dict, str]],
    failed: List[Dict],
    channel_summary: str,
    llm: LLMClient,
    partial: bool,
) -> str:
    """Render the markdown content for the whole-channel summary file."""
    video_lines = "\n".join(
        f"- [{video.get('title') or video['video_id']}](<{video['url']}>)"
        for video, _ in summaries
    )
    failed_section = ""
    if failed:
        failed_lines = "\n".join(
            f"- [{video.get('title') or video['video_id']}](<{video['url']}>)"
            for video in failed
        )
        failed_section = f"\n**Videos that could not be summarized:**\n\n{failed_lines}\n"
    partial_note = (
        "\n> Note: this run was interrupted; the summary covers only the videos listed above.\n"
        if partial
        else ""
    )
    return f"""# Channel Summary: {channel_name}

| | |
|---|---|
| **Channel** | {channel_name} |
| **Channel URL** | <{channel_url}> |
| **Videos covered** | {len(summaries)} |
| **Generated** | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} |
| **Model** | {llm.provider} / {llm.get_model()} |
{partial_note}
**Videos:**

{video_lines}
{failed_section}
---

{channel_summary}
"""


def generate_channel_summary(
    llm: LLMClient, channel_name: str, summaries: List[Tuple[Dict, str]]
) -> str:
    """Generate the roll-up channel summary from per-video summaries."""
    parts = [f"Channel: {channel_name}", ""]
    for video, summary in summaries:
        parts.append(f"## Video: {video.get('title') or video['video_id']}")
        parts.append(summary)
        parts.append("")
    rollup_input = "\n".join(parts)
    if len(rollup_input) > MAX_ROLLUP_INPUT_CHARS:
        rollup_input = rollup_input[:MAX_ROLLUP_INPUT_CHARS] + "\n[truncated]"
    return collect_stream(llm, CHANNEL_SUMMARY_PROMPT, rollup_input)


def run_channel(args) -> None:
    """Handle the channel subcommand."""
    # Imported here to avoid a circular import (cli imports this module's wrapper).
    from .cli import SYSTEM_PROMPT, generate_output_filename, slugify_filename_component

    try:
        llm = LLMClient(provider=args.provider)
        print(f"Using {llm.provider}/{llm.get_model()}")
    except ValueError as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    channel_url = normalize_channel_url(args.channel)
    print(f"Fetching channel info for {channel_url} ...")
    try:
        channel_name, videos = YouTubeHelper.get_channel_videos(channel_url, limit=args.last)
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    if not videos:
        print("No videos found on this channel.", file=sys.stderr)
        sys.exit(1)

    print(f"Channel: {channel_name} — found {len(videos)} video(s) (latest first)")

    output_dir = args.output_dir or default_output_dir(channel_name, slugify_filename_component)

    done: List[Tuple[Dict, str]] = []  # (video, summary) reused from previous runs
    todo: List[Dict] = []
    for video in videos:
        existing = find_existing_summary(output_dir, video["video_id"])
        if existing:
            with open(existing, "r", encoding="utf-8") as f:
                done.append((video, extract_summary_body(f.read())))
        else:
            todo.append(video)

    if not todo:
        print(f"All {len(videos)} video(s) already summarized in {output_dir}/")
    else:
        estimate = estimate_run(todo, llm.get_model())
        if not confirm_run(todo, len(done), estimate, llm.get_model(), args.yes):
            sys.exit(0)

    os.makedirs(output_dir, exist_ok=True)

    summaries: List[Tuple[Dict, str]] = list(done)
    failed: List[Dict] = []
    interrupted = False

    for i, video in enumerate(todo, 1):
        title = video.get("title") or video["video_id"]
        print(f"\n[{i}/{len(todo)}] {title}")
        try:
            print("  Fetching transcript...")
            transcript = YouTubeHelper.get_transcript(video["video_id"])
            print(f"  Summarizing ({len(transcript)} chars)...")
            summary = collect_stream(llm, SYSTEM_PROMPT, transcript)
        except KeyboardInterrupt:
            print("\nInterrupted — keeping completed summaries.")
            interrupted = True
            break
        except Exception as e:
            print(f"  Skipping ({str(e)})", file=sys.stderr)
            failed.append(video)
            continue

        filename = generate_output_filename(video["video_id"], video.get("title"))
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_video_summary_file(video, summary, channel_name, llm))
        print(f"  Saved {path}")
        summaries.append((video, summary))

    if not summaries:
        print("\nNo summaries were generated; skipping channel summary.", file=sys.stderr)
        sys.exit(1)

    if args.skip_channel_summary:
        print(f"\nDone. {len(summaries)} summary file(s) in {output_dir}/")
        return

    print(f"\nGenerating channel summary from {len(summaries)} video summaries...")
    try:
        channel_summary = generate_channel_summary(llm, channel_name, summaries)
    except KeyboardInterrupt:
        print("\nInterrupted — per-video summaries are saved; channel summary skipped.")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating channel summary: {str(e)}", file=sys.stderr)
        print(f"Per-video summaries are saved in {output_dir}/", file=sys.stderr)
        sys.exit(1)

    channel_summary_path = os.path.join(output_dir, "00__channel-summary.md")
    with open(channel_summary_path, "w", encoding="utf-8") as f:
        f.write(
            build_channel_summary_file(
                channel_name,
                channel_url,
                summaries,
                failed,
                channel_summary,
                llm,
                partial=interrupted,
            )
        )

    print(f"\nDone. {len(summaries)} video summary file(s) in {output_dir}/")
    if failed:
        print(f"{len(failed)} video(s) failed (no transcript available?).")
    print(f"Channel summary: {channel_summary_path}")
    if interrupted:
        print("Run the same command again to summarize the remaining videos.")
