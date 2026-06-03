"""Tests for long-video summarization strategies."""

import pytest

from youtube_summariser.summarizer import (
    INTERMEDIATE_REDUCE_SYSTEM_PROMPT,
    MAP_SYSTEM_PROMPT,
    REDUCE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    split_transcript_into_chunks,
    summarize_transcript,
)


class FakeLLM:
    """Small deterministic LLM stub with word-based token accounting."""

    provider = "local"

    def __init__(self, max_input_tokens=180):
        self.max_input_tokens = max_input_tokens
        self.truncation_allowed = None
        self.chat_calls = []
        self.stream_calls = []

    def get_max_input_tokens(self):
        return self.max_input_tokens

    def count_chat_tokens(self, system_prompt, user_message):
        return len(f"{system_prompt}\n{user_message}".split())

    def set_truncation_allowed(self, allowed):
        self.truncation_allowed = allowed

    def chat(self, system_prompt, user_message):
        self.chat_calls.append((system_prompt, user_message))
        if system_prompt == MAP_SYSTEM_PROMPT:
            return "chunk summary with [00:01]"
        if system_prompt == INTERMEDIATE_REDUCE_SYSTEM_PROMPT:
            return "bridge summary"
        if system_prompt == REDUCE_SYSTEM_PROMPT:
            return "final long-video summary"
        return "single summary"

    def stream_chat(self, system_prompt, user_message):
        self.stream_calls.append((system_prompt, user_message))
        yield self.chat(system_prompt, user_message)


def _transcript(line_count=16, words_per_line=10):
    lines = []
    for index in range(line_count):
        timestamp = f"[00:{index:02d}]"
        words = " ".join(f"word{index}_{word}" for word in range(words_per_line))
        lines.append(f"{timestamp} {words}")
    return "\n".join(lines)


def test_split_transcript_into_chunks_preserves_lines_under_budget():
    llm = FakeLLM(max_input_tokens=80)
    transcript = "\n".join(
        [
            "[00:00] first point with detail",
            "[00:05] second point with detail",
            "[00:10] third point with detail",
            "[00:15] fourth point with detail",
        ]
    )

    chunks = split_transcript_into_chunks(
        transcript,
        llm,
        system_prompt="map",
        max_prompt_tokens=14,
    )

    assert len(chunks) > 1
    assert all(chunk.token_count <= 14 for chunk in chunks)
    assert chunks[0].text.startswith("[00:00]")
    assert chunks[1].text.startswith("[00:")


def test_auto_uses_single_shot_when_prompt_fits():
    llm = FakeLLM(max_input_tokens=300)

    summary = summarize_transcript(
        "[00:00] short transcript",
        llm,
        stream=False,
        summary_strategy="auto",
    )

    assert summary == "single summary"
    assert llm.truncation_allowed is False
    assert llm.chat_calls == [(SYSTEM_PROMPT, "[00:00] short transcript")]


def test_auto_uses_map_reduce_when_prompt_is_too_large():
    llm = FakeLLM(max_input_tokens=130)

    summary = summarize_transcript(
        _transcript(line_count=12, words_per_line=12),
        llm,
        stream=False,
        summary_strategy="auto",
    )

    map_calls = [call for call in llm.chat_calls if call[0] == MAP_SYSTEM_PROMPT]
    reduce_calls = [call for call in llm.chat_calls if call[0] == REDUCE_SYSTEM_PROMPT]
    assert summary == "final long-video summary"
    assert len(map_calls) > 1
    assert len(reduce_calls) == 1


def test_single_strategy_rejects_over_budget_prompt_without_truncate():
    llm = FakeLLM(max_input_tokens=130)

    with pytest.raises(ValueError) as exc_info:
        summarize_transcript(
            _transcript(line_count=12, words_per_line=12),
            llm,
            stream=False,
            summary_strategy="single",
        )

    assert "too long for the selected model context" in str(exc_info.value)
    assert "--allow-truncate" in str(exc_info.value)
    assert llm.truncation_allowed is False
    assert llm.chat_calls == []


def test_single_strategy_allows_explicit_truncation_smoke_test():
    llm = FakeLLM(max_input_tokens=130)

    summary = summarize_transcript(
        _transcript(line_count=12, words_per_line=12),
        llm,
        stream=False,
        summary_strategy="single",
        allow_truncate=True,
    )

    assert summary == "single summary"
    assert llm.truncation_allowed is True
    assert llm.chat_calls[0][0] == SYSTEM_PROMPT


def test_reduce_recurses_when_chunk_summaries_exceed_budget():
    class VerboseMapLLM(FakeLLM):
        def chat(self, system_prompt, user_message):
            self.chat_calls.append((system_prompt, user_message))
            if system_prompt == MAP_SYSTEM_PROMPT:
                return " ".join(f"detail{index}" for index in range(40))
            if system_prompt == INTERMEDIATE_REDUCE_SYSTEM_PROMPT:
                return "bridge summary"
            if system_prompt == REDUCE_SYSTEM_PROMPT:
                return "final reduced summary"
            return "single summary"

    llm = VerboseMapLLM(max_input_tokens=130)

    summary = summarize_transcript(
        _transcript(line_count=18, words_per_line=14),
        llm,
        stream=False,
        summary_strategy="map-reduce",
    )

    intermediate_calls = [
        call for call in llm.chat_calls if call[0] == INTERMEDIATE_REDUCE_SYSTEM_PROMPT
    ]
    assert summary == "final reduced summary"
    assert intermediate_calls
