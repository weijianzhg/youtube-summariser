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

        def fake_process_video(video_id, video_url, video_title, args, llm, channel=None):
            captured["video_id"] = video_id
            captured["video_url"] = video_url
            captured["video_title"] = video_title
            captured["channel"] = channel
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
        assert captured["channel"] == "Example Channel"
        assert captured["args"] is args
        assert isinstance(captured["llm"], FakeLLMClient)

    def test_cmd_summarise_fetches_title_and_passes_to_process_video(self, monkeypatch):
        """Direct URL mode should fetch and pass video title."""
        monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(cli.YouTubeHelper, "validate_url", lambda url: True)
        monkeypatch.setattr(cli.YouTubeHelper, "extract_video_id", lambda url: "abc123")
        monkeypatch.setattr(
            cli.YouTubeHelper,
            "get_video_metadata",
            lambda video_id: {"title": "Title from Metadata", "channel": "Example Channel"},
        )

        captured = {}

        def fake_process_video(video_id, video_url, video_title, args, llm, channel=None):
            captured["video_id"] = video_id
            captured["video_url"] = video_url
            captured["video_title"] = video_title
            captured["channel"] = channel
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
        assert captured["channel"] == "Example Channel"
        assert captured["args"] is args
        assert isinstance(captured["llm"], FakeLLMClient)

    def test_cmd_summarise_uses_none_title_when_lookup_fails(self, monkeypatch):
        """Direct URL mode should continue when title lookup fails."""
        monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
        monkeypatch.setattr(cli.YouTubeHelper, "validate_url", lambda url: True)
        monkeypatch.setattr(cli.YouTubeHelper, "extract_video_id", lambda url: "abc123")

        def raise_lookup_error(video_id):
            raise Exception("lookup failed")

        monkeypatch.setattr(cli.YouTubeHelper, "get_video_metadata", raise_lookup_error)

        captured = {}

        def fake_process_video(video_id, video_url, video_title, args, llm, channel=None):
            captured["video_title"] = video_title
            captured["channel"] = channel

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
        assert captured["channel"] is None


class TestObsidianSummaryFormat:
    """Test Obsidian-oriented summary document formatting."""

    def test_format_summary_document_includes_frontmatter_and_title_heading(self):
        """Saved notes should use YAML frontmatter and the video title as H1."""
        llm = FakeLLMClient()
        created = __import__("datetime").datetime(2026, 7, 28, 15, 30, 0)

        document = cli.format_summary_document(
            summary="### TL;DR\nA crisp summary.\n\n#youtube #ai",
            video_id="abc123",
            video_url="https://www.youtube.com/watch?v=abc123",
            video_title='How to "ship" notes',
            llm=llm,
            channel="Example Channel",
            created_at=created,
        )

        assert document.startswith("---\n")
        assert 'title: "How to \\"ship\\" notes"' in document
        assert 'url: "https://www.youtube.com/watch?v=abc123"' in document
        assert "video_id: abc123" in document
        assert 'channel: "Example Channel"' in document
        assert "created: 2026-07-28" in document
        assert "tags:\n  - youtube\n  - summary" in document
        assert 'model: "anthropic/test-model"' in document
        assert "# How to \"ship\" notes\n" in document
        assert "### TL;DR\nA crisp summary." in document

    def test_format_summary_document_falls_back_without_title(self):
        """Missing titles should still produce a usable note heading."""
        document = cli.format_summary_document(
            summary="Body",
            video_id="abc123",
            video_url="https://www.youtube.com/watch?v=abc123",
            video_title=None,
            llm=FakeLLMClient(),
        )

        assert 'title: "YouTube Video abc123"' in document
        assert "# YouTube Video abc123\n" in document
        assert "channel:" not in document

    def test_build_system_prompt_includes_deep_link_template(self):
        """The prompt should ask for clickable YouTube timestamp links."""
        prompt = cli.build_system_prompt("abc123")
        assert "https://www.youtube.com/watch?v=abc123&t=SECONDS" in prompt
        assert "Do not invent [[wikilinks]]" in prompt
        assert "#youtube" in prompt
        assert "Do not include an H1 title" in prompt
