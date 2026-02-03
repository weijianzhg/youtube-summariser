"""Tests for CLI search command and related functions."""

from youtube_summariser.cli import format_duration


class TestFormatDuration:
    """Test the format_duration helper function."""

    def test_format_duration_seconds_only(self):
        """Should format small durations as M:SS."""
        assert format_duration("45") == "0:45"
        assert format_duration("0") == "0:00"

    def test_format_duration_minutes_and_seconds(self):
        """Should format medium durations as MM:SS."""
        assert format_duration("90") == "1:30"
        assert format_duration("300") == "5:00"
        assert format_duration("3599") == "59:59"

    def test_format_duration_hours(self):
        """Should format long durations as H:MM:SS."""
        assert format_duration("3600") == "1:00:00"
        assert format_duration("3661") == "1:01:01"
        assert format_duration("7200") == "2:00:00"

    def test_format_duration_invalid_input(self):
        """Should return placeholder for invalid input."""
        assert format_duration("not-a-number") == "??:??"
        assert format_duration("") == "??:??"

    def test_format_duration_none_input(self):
        """Should handle None input gracefully."""
        assert format_duration(None) == "??:??"
