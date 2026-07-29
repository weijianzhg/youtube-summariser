"""Tests for channel-wide summarization."""

from argparse import Namespace

import pytest

from youtube_summariser import cli


class FakeLLMClient:
    """Minimal LLM stub for channel command tests."""

    instances = []

    def __init__(self, provider=None):
        self.provider = provider or "anthropic"
        self.instances.append(self)

    def get_model(self):
        return "test-model"


def channel_args(**overrides):
    """Build channel command arguments with sensible defaults."""
    values = {
        "provider": None,
        "url": "https://www.youtube.com/@example",
        "max_videos": 2,
        "output_dir": "channel-summaries",
        "no_save": False,
        "no_stream": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cmd_channel_reuses_one_llm_and_processes_each_video(monkeypatch):
    """A channel batch should initialize one LLM client and reuse the video pipeline."""
    FakeLLMClient.instances = []
    monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)

    videos = [
        {"video_id": "one", "url": "https://www.youtube.com/watch?v=one"},
        {"video_id": "two", "url": "https://www.youtube.com/watch?v=two"},
    ]
    requested = {}

    def fake_get_channel_videos(url, max_videos=None):
        requested["url"] = url
        requested["max_videos"] = max_videos
        return videos

    monkeypatch.setattr(cli.YouTubeHelper, "get_channel_videos", fake_get_channel_videos)
    monkeypatch.setattr(
        cli.YouTubeHelper,
        "get_video_metadata",
        lambda video_id: {"title": f"Title {video_id}", "channel": "Example Channel"},
    )

    processed = []

    def fake_process_video(
        video_id,
        video_url,
        video_title,
        args,
        llm,
        channel=None,
        published_at=None,
    ):
        processed.append((video_id, video_url, video_title, args, llm, published_at))
        return True

    monkeypatch.setattr(cli, "process_video", fake_process_video)
    args = channel_args()

    cli.cmd_channel(args)

    assert requested == {"url": args.url, "max_videos": 2}
    assert len(FakeLLMClient.instances) == 1
    assert [item[0] for item in processed] == ["one", "two"]
    assert [item[2] for item in processed] == ["Title one", "Title two"]
    assert all(item[3] is args for item in processed)
    assert all(item[4] is FakeLLMClient.instances[0] for item in processed)


def test_cmd_channel_continues_after_individual_failure(monkeypatch):
    """One unavailable transcript should not stop later videos from being summarized."""
    FakeLLMClient.instances = []
    monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)
    monkeypatch.setattr(
        cli.YouTubeHelper,
        "get_channel_videos",
        lambda url, max_videos=None: [
            {"video_id": "fails", "url": "https://www.youtube.com/watch?v=fails"},
            {"video_id": "works", "url": "https://www.youtube.com/watch?v=works"},
        ],
    )
    monkeypatch.setattr(
        cli.YouTubeHelper,
        "get_video_metadata",
        lambda video_id: {"title": video_id, "channel": ""},
    )

    processed = []

    def fake_process_video(
        video_id,
        video_url,
        video_title,
        args,
        llm,
        channel=None,
        published_at=None,
    ):
        processed.append(video_id)
        return video_id == "works"

    monkeypatch.setattr(cli, "process_video", fake_process_video)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_channel(channel_args())

    assert exc_info.value.code == 1
    assert processed == ["fails", "works"]


def test_cmd_channel_rejects_non_positive_limit_before_initializing_llm(monkeypatch):
    """An invalid batch limit should fail before any API client is created."""
    FakeLLMClient.instances = []
    monkeypatch.setattr(cli, "LLMClient", FakeLLMClient)

    with pytest.raises(SystemExit) as exc_info:
        cli.cmd_channel(channel_args(max_videos=0))

    assert exc_info.value.code == 1
    assert FakeLLMClient.instances == []


def test_process_video_saves_channel_summary_in_output_directory(tmp_path, monkeypatch):
    """The shared video pipeline should place channel output under its batch directory."""
    monkeypatch.setattr(cli.YouTubeHelper, "get_transcript", lambda video_id: "[00:00] Hello")
    monkeypatch.setattr(
        cli,
        "summarize_transcript",
        lambda transcript, llm, stream=True, video_id=None, video_title=None, channel=None: (
            "## TL;DR\nA summary.\n"
            "## Key Takeaways\n- One\n"
            "## Detailed Summary\nDetails.\n"
            '<!-- knowledge-graph\n{"content_type":"talk"}\n-->'
        ),
    )
    args = channel_args(output_dir=tmp_path)
    llm = FakeLLMClient()

    succeeded = cli.process_video(
        "abc123",
        "https://www.youtube.com/watch?v=abc123",
        "Example Video",
        args,
        llm,
        channel="Example Channel",
    )

    assert succeeded is True
    output_files = list(tmp_path.glob("*__example-video__abc123.md"))
    assert len(output_files) == 1
    content = output_files[0].read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert 'title: "Example Video"' in content
    assert 'channel: "Example Channel"' in content
    assert "# Example Video\n" in content
    assert "## TL;DR\nA summary." in content
    assert "https://www.youtube.com/watch?v=abc123" in content


def test_process_video_retries_one_incomplete_summary(tmp_path, monkeypatch):
    """A malformed first response should be retried before writing the note."""
    monkeypatch.setattr(cli.YouTubeHelper, "get_transcript", lambda video_id: "[00:00] Hello")
    responses = iter(
        [
            '<!-- knowledge-graph\n{"content_type":"talk"}\n-->',
            "## TL;DR\nA summary.\n"
            "## Key Takeaways\n- One\n"
            "## Detailed Summary\nDetails.\n"
            '<!-- knowledge-graph\n{"content_type":"talk"}\n-->',
        ]
    )
    calls = []

    def fake_summarize(
        transcript,
        llm,
        stream=True,
        video_id=None,
        video_title=None,
        channel=None,
    ):
        calls.append((video_id, video_title, channel))
        return next(responses)

    monkeypatch.setattr(cli, "summarize_transcript", fake_summarize)
    args = channel_args(output_dir=tmp_path)

    assert cli.process_video(
        "abc123",
        "https://www.youtube.com/watch?v=abc123",
        "Example Video",
        args,
        FakeLLMClient(),
        channel="Example Channel",
    )
    assert calls == [
        ("abc123", "Example Video", "Example Channel"),
        ("abc123", "Example Video", "Example Channel"),
    ]
    assert len(list(tmp_path.glob("*.md"))) == 1


def test_process_video_rejects_two_incomplete_summaries(tmp_path, monkeypatch, capsys):
    """Repeated incomplete responses should fail without saving a broken note."""
    monkeypatch.setattr(cli.YouTubeHelper, "get_transcript", lambda video_id: "[00:00] Hello")
    calls = []

    def fake_summarize(*args, **kwargs):
        calls.append(1)
        return '<!-- knowledge-graph\n{"content_type":"talk"}\n-->'

    monkeypatch.setattr(cli, "summarize_transcript", fake_summarize)
    args = channel_args(output_dir=tmp_path)

    succeeded = cli.process_video(
        "abc123",
        "https://www.youtube.com/watch?v=abc123",
        "Example Video",
        args,
        FakeLLMClient(),
    )

    assert succeeded is False
    assert calls == [1, 1]
    assert list(tmp_path.glob("*.md")) == []
    assert "incomplete response twice" in capsys.readouterr().err


def test_process_video_does_not_retry_after_keyboard_interrupt(tmp_path, monkeypatch, capsys):
    """Cancelling a stream should not start another paid LLM request."""
    monkeypatch.setattr(cli.YouTubeHelper, "get_transcript", lambda video_id: "[00:00] Hello")
    calls = []

    def fake_summarize(*args, **kwargs):
        calls.append(1)
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "summarize_transcript", fake_summarize)
    args = channel_args(output_dir=tmp_path)

    succeeded = cli.process_video(
        "abc123",
        "https://www.youtube.com/watch?v=abc123",
        "Example Video",
        args,
        FakeLLMClient(),
    )

    assert succeeded is False
    assert calls == [1]
    assert list(tmp_path.glob("*.md")) == []
    assert "cancelled" in capsys.readouterr().err
