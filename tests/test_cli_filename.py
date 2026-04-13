"""Tests for default filename generation and title propagation."""

import re
from argparse import Namespace

from youtube_summariser import cli


class FakeLLMClient:
    """Minimal LLM stub for CLI command tests."""

    def __init__(self, provider=None):
        self.provider = provider or "anthropic"

    def get_model(self):
        return "test-model"


class TestGenerateOutputFilename:
    """Test default filename creation behavior."""

    def test_generate_output_filename_slugifies_title(self):
        """Should include a date prefix and sanitized title slug."""
        filename = cli.generate_output_filename("abc123", "How to Use LLMs, Effectively!")
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}__how-to-use-llms-effectively__abc123\.md",
            filename,
        )

    def test_generate_output_filename_falls_back_when_title_missing(self):
        """Should fallback to video-id-based filename when title is unavailable."""
        filename = cli.generate_output_filename("abc123", "")
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}__video-abc123\.md", filename)


class TestTitlePlumbing:
    """Test that title metadata is passed through command handlers."""

    def test_cmd_search_passes_selected_title_to_process_video(self, monkeypatch):
        """Search mode should reuse selected video title for filename generation."""
        monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(
            cli.YouTubeHelper,
            "search_videos",
            lambda query, max_results=5: [
                {
                    "video_id": "abc123",
                    "title": "Readable Video Title",
                    "url": "https://www.youtube.com/watch?v=abc123",
                    "duration": "300",
                    "channel": "Example Channel",
                }
            ],
        )

        captured = {}

        def fake_process_video(video_id, video_url, video_title, args, llm):
            captured["video_id"] = video_id
            captured["video_url"] = video_url
            captured["video_title"] = video_title
            captured["args"] = args
            captured["llm"] = llm

        monkeypatch.setattr(cli, "process_video", fake_process_video)

        args = Namespace(
            provider=None,
            query="test query",
            max_results=5,
            first=True,
            output=None,
            no_save=False,
            no_stream=True,
        )

        cli.cmd_search(args)

        assert captured["video_id"] == "abc123"
        assert captured["video_title"] == "Readable Video Title"
        assert captured["args"] is args
        assert isinstance(captured["llm"], FakeLLMClient)

    def test_cmd_summarise_fetches_title_and_passes_to_process_video(self, monkeypatch):
        """Direct URL mode should fetch and pass video title."""
        monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(cli.YouTubeHelper, "validate_url", lambda url: True)
        monkeypatch.setattr(cli.YouTubeHelper, "extract_video_id", lambda url: "abc123")
        monkeypatch.setattr(
            cli.YouTubeHelper,
            "get_video_title",
            lambda video_id: "Title from Metadata",
        )

        captured = {}

        def fake_process_video(video_id, video_url, video_title, args, llm):
            captured["video_id"] = video_id
            captured["video_url"] = video_url
            captured["video_title"] = video_title
            captured["args"] = args
            captured["llm"] = llm

        monkeypatch.setattr(cli, "process_video", fake_process_video)

        args = Namespace(
            provider=None,
            url="https://www.youtube.com/watch?v=abc123",
            output=None,
            no_save=False,
            no_stream=True,
        )

        cli.cmd_summarise(args)

        assert captured["video_id"] == "abc123"
        assert captured["video_url"] == args.url
        assert captured["video_title"] == "Title from Metadata"
        assert captured["args"] is args
        assert isinstance(captured["llm"], FakeLLMClient)

    def test_cmd_summarise_uses_none_title_when_lookup_fails(self, monkeypatch):
        """Direct URL mode should continue when title lookup fails."""
        monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(cli.YouTubeHelper, "validate_url", lambda url: True)
        monkeypatch.setattr(cli.YouTubeHelper, "extract_video_id", lambda url: "abc123")

        def raise_lookup_error(video_id):
            raise Exception("lookup failed")

        monkeypatch.setattr(cli.YouTubeHelper, "get_video_title", raise_lookup_error)

        captured = {}

        def fake_process_video(video_id, video_url, video_title, args, llm):
            captured["video_title"] = video_title

        monkeypatch.setattr(cli, "process_video", fake_process_video)

        args = Namespace(
            provider=None,
            url="https://www.youtube.com/watch?v=abc123",
            output=None,
            no_save=False,
            no_stream=True,
        )

        cli.cmd_summarise(args)

        assert captured["video_title"] is None
