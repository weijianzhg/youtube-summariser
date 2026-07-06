"""Tests for the channel subcommand helpers."""

import os

from youtube_summariser.channel import (
    OUTPUT_TOKENS_PER_VIDEO,
    PROMPT_OVERHEAD_TOKENS,
    TOKENS_PER_VIDEO_MINUTE,
    default_output_dir,
    estimate_run,
    extract_summary_body,
    find_existing_summary,
    format_seconds,
    normalize_channel_url,
)
from youtube_summariser.cli import slugify_filename_component


class TestNormalizeChannelUrl:
    """Test channel URL normalization."""

    def test_full_url_unchanged(self):
        url = "https://www.youtube.com/@mkbhd"
        assert normalize_channel_url(url) == url

    def test_http_url_unchanged(self):
        url = "http://youtube.com/channel/UC123"
        assert normalize_channel_url(url) == url

    def test_handle_gets_prefixed(self):
        assert normalize_channel_url("@mkbhd") == "https://www.youtube.com/@mkbhd"

    def test_bare_name_treated_as_handle(self):
        assert normalize_channel_url("mkbhd") == "https://www.youtube.com/@mkbhd"

    def test_domain_without_scheme(self):
        assert (
            normalize_channel_url("youtube.com/@mkbhd")
            == "https://youtube.com/@mkbhd"
        )

    def test_whitespace_stripped(self):
        assert normalize_channel_url("  @mkbhd  ") == "https://www.youtube.com/@mkbhd"


class TestEstimateRun:
    """Test the token/cost/time estimator."""

    def test_known_duration_token_math(self):
        videos = [{"duration": "600"}]  # 10 minutes
        estimate = estimate_run(videos, "claude-sonnet-4-5-20250929")
        assert estimate["input_tokens"] == 10 * TOKENS_PER_VIDEO_MINUTE + PROMPT_OVERHEAD_TOKENS
        assert estimate["output_tokens"] == OUTPUT_TOKENS_PER_VIDEO

    def test_unknown_duration_uses_fallback(self):
        estimate = estimate_run([{"duration": "0"}], "claude-sonnet-4-5")
        assert estimate["input_tokens"] > PROMPT_OVERHEAD_TOKENS

    def test_invalid_duration_uses_fallback(self):
        estimate = estimate_run([{"duration": "n/a"}, {"duration": None}], "claude-sonnet-4-5")
        assert estimate["input_tokens"] > 2 * PROMPT_OVERHEAD_TOKENS

    def test_known_model_has_cost(self):
        estimate = estimate_run([{"duration": "600"}], "claude-sonnet-4-5-20250929")
        assert estimate["cost"] is not None
        assert estimate["cost"] > 0

    def test_cost_uses_sonnet_pricing(self):
        estimate = estimate_run([{"duration": "600"}], "claude-sonnet-4-5")
        expected = (
            estimate["input_tokens"] * 3.0 + estimate["output_tokens"] * 15.0
        ) / 1_000_000
        assert abs(estimate["cost"] - expected) < 1e-9

    def test_unknown_model_has_no_cost(self):
        estimate = estimate_run([{"duration": "600"}], "some-unknown-model")
        assert estimate["cost"] is None

    def test_time_scales_with_video_count(self):
        one = estimate_run([{"duration": "60"}], "claude-sonnet-4-5")
        three = estimate_run([{"duration": "60"}] * 3, "claude-sonnet-4-5")
        assert three["time_range_seconds"][0] == 3 * one["time_range_seconds"][0]


class TestOutputDir:
    """Test default output folder naming."""

    def test_contains_channel_slug(self):
        name = default_output_dir("My Cool Channel!", slugify_filename_component)
        assert name.endswith("__my-cool-channel")

    def test_empty_name_falls_back(self):
        name = default_output_dir("!!!", slugify_filename_component)
        assert name.endswith("__channel")


class TestFindExistingSummary:
    """Test resume detection via existing summary files."""

    def test_finds_matching_file(self, tmp_path):
        f = tmp_path / "2026-07-06__some-title__abc123.md"
        f.write_text("content")
        found = find_existing_summary(str(tmp_path), "abc123")
        assert found == str(f)

    def test_no_match_returns_none(self, tmp_path):
        (tmp_path / "2026-07-06__some-title__abc123.md").write_text("content")
        assert find_existing_summary(str(tmp_path), "zzz999") is None

    def test_missing_dir_returns_none(self, tmp_path):
        assert find_existing_summary(os.path.join(str(tmp_path), "nope"), "abc") is None


class TestExtractSummaryBody:
    """Test stripping the metadata header from saved summary files."""

    def test_strips_header(self):
        content = "# YouTube Video Summary\n\n| a | b |\n\n---\n\n### TL;DR\nGreat video."
        assert extract_summary_body(content) == "### TL;DR\nGreat video."

    def test_no_marker_returns_all(self):
        assert extract_summary_body("  just a summary  ") == "just a summary"


class TestFormatSeconds:
    """Test compact duration formatting."""

    def test_seconds(self):
        assert format_seconds(45) == "45s"

    def test_minutes(self):
        assert format_seconds(300) == "5m"

    def test_hours(self):
        assert format_seconds(5400) == "1h 30m"
