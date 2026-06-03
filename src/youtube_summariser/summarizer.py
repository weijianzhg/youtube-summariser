"""Transcript summarization strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SummaryStrategy = Literal["auto", "single", "map-reduce"]

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

Preserve any timestamps from the transcript. Be concise; omit filler and tangents."""

MAP_SYSTEM_PROMPT = """Summarize one chunk of a timestamped YouTube transcript.

Return compact markdown with:
- Time range covered, if visible
- 3-6 important points with timestamps
- Notable quotes or examples
- Any unresolved context that later chunks may need

Preserve timestamps and omit filler."""

REDUCE_SYSTEM_PROMPT = """Synthesize chunk summaries into one final YouTube video summary.

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

Preserve timestamps from the chunk summaries.
Remove duplicates and keep the final answer concise."""

INTERMEDIATE_REDUCE_SYSTEM_PROMPT = """Condense these chunk summaries into a smaller bridge summary.

Keep the main chronology, timestamps, decisions, examples, quotes, and unresolved context.
Remove repetition and keep the result compact."""

UNKNOWN_CONTEXT_PROMPT_BUDGET = 6000
MIN_PROMPT_BUDGET = 64


@dataclass(frozen=True)
class TranscriptChunk:
    """A token-budgeted transcript slice."""

    index: int
    total: int
    text: str
    token_count: int


def summarize_transcript(
    transcript: str,
    llm,
    stream: bool = True,
    summary_strategy: SummaryStrategy = "auto",
    allow_truncate: bool = False,
) -> str:
    """
    Summarize a transcript with single-shot or map-reduce behavior.

    Long local-model prompts are routed through map-reduce by default so the
    backend does not silently discard most of the transcript.
    """
    strategy = _normalize_strategy(summary_strategy)
    _set_truncation_allowed(llm, allow_truncate)

    max_input_tokens = _max_input_tokens(llm)
    full_prompt_tokens = _count_chat_tokens(llm, SYSTEM_PROMPT, transcript)
    prompt_over_budget = max_input_tokens > 0 and full_prompt_tokens > max_input_tokens

    if strategy == "auto":
        strategy = "map-reduce" if prompt_over_budget else "single"

    if strategy == "single":
        if prompt_over_budget and not allow_truncate:
            raise ValueError(
                "This transcript is too long for the selected model context "
                f"({full_prompt_tokens:,} input tokens, limit {max_input_tokens:,}). "
                "Use the default auto mode or pass --summary-strategy map-reduce for full "
                "long-video summarization. For a quick partial smoke test, pass --allow-truncate."
            )
        return _run_chat(llm, SYSTEM_PROMPT, transcript, stream=stream, print_stream=stream)

    return summarize_long_transcript(transcript, llm, stream=stream)


def summarize_long_transcript(transcript: str, llm, stream: bool = True) -> str:
    """Summarize a long transcript through chunk summaries and synthesis."""
    chunk_budget = _chunk_prompt_budget(llm)
    chunks = split_transcript_into_chunks(
        transcript,
        llm,
        system_prompt=MAP_SYSTEM_PROMPT,
        max_prompt_tokens=chunk_budget,
    )
    print(
        f"Long-video mode: transcript split into {len(chunks)} chunk(s); "
        "synthesizing final summary."
    )

    chunk_summaries: list[str] = []
    for chunk in chunks:
        print(f"Summarizing chunk {chunk.index}/{chunk.total}...")
        user_message = f"Transcript chunk {chunk.index}/{chunk.total}:\n\n{chunk.text}"
        chunk_summaries.append(_run_chat(llm, MAP_SYSTEM_PROMPT, user_message, stream=False))

    return _reduce_summaries(chunk_summaries, llm, stream=stream)


def split_transcript_into_chunks(
    transcript: str,
    llm,
    system_prompt: str = MAP_SYSTEM_PROMPT,
    max_prompt_tokens: int | None = None,
) -> list[TranscriptChunk]:
    """Split transcript text on line boundaries while respecting a prompt token budget."""
    prompt_budget = max_prompt_tokens or _chunk_prompt_budget(llm)
    lines = _transcript_lines(transcript)

    chunk_texts: list[str] = []
    current_lines: list[str] = []

    for line in lines:
        candidate_lines = [*current_lines, line]
        candidate_text = "\n".join(candidate_lines)
        if _prompt_fits(llm, system_prompt, candidate_text, prompt_budget):
            current_lines = candidate_lines
            continue

        if current_lines:
            chunk_texts.append("\n".join(current_lines))
            current_lines = []

        if _prompt_fits(llm, system_prompt, line, prompt_budget):
            current_lines = [line]
        else:
            chunk_texts.extend(_split_oversized_line(line, llm, system_prompt, prompt_budget))

    if current_lines:
        chunk_texts.append("\n".join(current_lines))

    total = len(chunk_texts)
    return [
        TranscriptChunk(
            index=index,
            total=total,
            text=text,
            token_count=_count_chat_tokens(llm, system_prompt, text),
        )
        for index, text in enumerate(chunk_texts, start=1)
    ]


def _reduce_summaries(
    summaries: list[str],
    llm,
    stream: bool,
    depth: int = 0,
) -> str:
    if not summaries:
        return ""
    if depth > 8:
        raise ValueError(
            "Chunk summaries are still too large to synthesize safely. "
            "Try increasing local.max_input_tokens or lowering local.max_tokens."
        )

    prompt_budget = _chunk_prompt_budget(llm)
    combined = _format_summaries(summaries)
    if _prompt_fits(llm, REDUCE_SYSTEM_PROMPT, combined, prompt_budget):
        return _run_chat(llm, REDUCE_SYSTEM_PROMPT, combined, stream=stream, print_stream=stream)

    batches = _batch_summaries(summaries, llm, INTERMEDIATE_REDUCE_SYSTEM_PROMPT, prompt_budget)
    intermediate: list[str] = []
    for index, batch in enumerate(batches, start=1):
        print(f"Reducing summary batch {index}/{len(batches)}...")
        intermediate.append(
            _run_chat(
                llm,
                INTERMEDIATE_REDUCE_SYSTEM_PROMPT,
                _format_summaries(batch),
                stream=False,
            )
        )

    return _reduce_summaries(intermediate, llm, stream=stream, depth=depth + 1)


def _batch_summaries(
    summaries: list[str],
    llm,
    system_prompt: str,
    prompt_budget: int,
) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    for summary in summaries:
        candidate = [*current, summary]
        if current and not _prompt_fits(
            llm,
            system_prompt,
            _format_summaries(candidate),
            prompt_budget,
        ):
            batches.append(current)
            current = [summary]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


def _split_oversized_line(
    line: str,
    llm,
    system_prompt: str,
    prompt_budget: int,
) -> list[str]:
    parts: list[str] = []
    current_words: list[str] = []
    for word in line.split():
        candidate_words = [*current_words, word]
        candidate_text = " ".join(candidate_words)
        if _prompt_fits(llm, system_prompt, candidate_text, prompt_budget):
            current_words = candidate_words
            continue

        if current_words:
            parts.append(" ".join(current_words))
            current_words = [word]
        else:
            parts.append(word)
            current_words = []

    if current_words:
        parts.append(" ".join(current_words))
    return parts or [line]


def _run_chat(
    llm,
    system_prompt: str,
    user_message: str,
    stream: bool,
    print_stream: bool = False,
) -> str:
    if stream:
        return _run_streaming_chat(llm, system_prompt, user_message, print_stream=print_stream)

    try:
        return llm.chat(system_prompt, user_message)
    except NotImplementedError:
        return "".join(llm.stream_chat(system_prompt, user_message))


def _run_streaming_chat(llm, system_prompt: str, user_message: str, print_stream: bool) -> str:
    summary_parts: list[str] = []
    if print_stream:
        print("\n--- Summary ---\n")
    try:
        for chunk in llm.stream_chat(system_prompt, user_message):
            if print_stream:
                print(chunk, end="", flush=True)
            summary_parts.append(chunk)
        if print_stream:
            print("\n")
        return "".join(summary_parts)
    except KeyboardInterrupt:
        if print_stream:
            print("\n\nSummary generation interrupted by user.")
        return "".join(summary_parts)


def _format_summaries(summaries: list[str]) -> str:
    return "\n\n".join(
        f"### Chunk summary {index}\n{summary.strip()}"
        for index, summary in enumerate(summaries, start=1)
        if summary.strip()
    )


def _transcript_lines(transcript: str) -> list[str]:
    lines = [line.strip() for line in transcript.splitlines() if line.strip()]
    return lines or [transcript.strip()]


def _prompt_fits(llm, system_prompt: str, user_message: str, max_prompt_tokens: int) -> bool:
    if max_prompt_tokens <= 0:
        return True
    return _count_chat_tokens(llm, system_prompt, user_message) <= max_prompt_tokens


def _chunk_prompt_budget(llm) -> int:
    max_input_tokens = _max_input_tokens(llm)
    if max_input_tokens <= 0:
        return UNKNOWN_CONTEXT_PROMPT_BUDGET
    reserve = min(max(max_input_tokens // 5, 32), 512)
    return max(max_input_tokens - reserve, MIN_PROMPT_BUDGET)


def _max_input_tokens(llm) -> int:
    getter = getattr(llm, "get_max_input_tokens", None)
    if callable(getter):
        return int(getter() or 0)
    return 0


def _count_chat_tokens(llm, system_prompt: str, user_message: str) -> int:
    counter = getattr(llm, "count_chat_tokens", None)
    if callable(counter):
        return int(counter(system_prompt, user_message))
    return _estimate_chat_tokens(system_prompt, user_message)


def _estimate_chat_tokens(system_prompt: str, user_message: str) -> int:
    return max(1, (len(system_prompt) + len(user_message)) // 4 + 16)


def _set_truncation_allowed(llm, allowed: bool) -> None:
    setter = getattr(llm, "set_truncation_allowed", None)
    if callable(setter):
        setter(allowed)


def _normalize_strategy(strategy: str) -> SummaryStrategy:
    if strategy not in {"auto", "single", "map-reduce"}:
        raise ValueError(f"Unsupported summary strategy: {strategy}")
    return strategy  # type: ignore[return-value]
