"""Tests for YouTubeHelper class, including search functionality."""

from unittest.mock import MagicMock, patch

import pytest

from youtube_summariser.youtube_helper import YouTubeHelper


class TestYouTubeHelperSearch:
    """Test the search_videos functionality."""

    def test_search_videos_returns_correct_structure(self):
        """search_videos should return list of dicts with expected keys."""
        # Create mock video objects
        mock_video1 = MagicMock()
        mock_video1.video_id = "abc123"
        mock_video1.title = "Test Video 1"
        mock_video1.watch_url = "https://www.youtube.com/watch?v=abc123"
        mock_video1.length = 300
        mock_video1.author = "Test Channel"

        mock_video2 = MagicMock()
        mock_video2.video_id = "def456"
        mock_video2.title = "Test Video 2"
        mock_video2.watch_url = "https://www.youtube.com/watch?v=def456"
        mock_video2.length = 600
        mock_video2.author = "Another Channel"

        mock_search = MagicMock()
        mock_search.videos = [mock_video1, mock_video2]

        with patch("pytubefix.Search", return_value=mock_search):
            results = YouTubeHelper.search_videos("test query", max_results=5)

        assert len(results) == 2
        assert results[0]["video_id"] == "abc123"
        assert results[0]["title"] == "Test Video 1"
        assert results[0]["url"] == "https://www.youtube.com/watch?v=abc123"
        assert results[0]["duration"] == "300"
        assert results[0]["channel"] == "Test Channel"

        assert results[1]["video_id"] == "def456"
        assert results[1]["title"] == "Test Video 2"

    def test_search_videos_respects_max_results(self):
        """search_videos should limit results to max_results."""
        mock_videos = []
        for i in range(10):
            mv = MagicMock()
            mv.video_id = f"vid{i}"
            mv.title = f"Video {i}"
            mv.watch_url = f"https://www.youtube.com/watch?v=vid{i}"
            mv.length = 100 * i
            mv.author = f"Channel {i}"
            mock_videos.append(mv)

        mock_search = MagicMock()
        mock_search.videos = mock_videos

        with patch("pytubefix.Search", return_value=mock_search):
            results = YouTubeHelper.search_videos("test query", max_results=3)

        assert len(results) == 3
        assert results[0]["video_id"] == "vid0"
        assert results[2]["video_id"] == "vid2"

    def test_search_videos_empty_query_raises_error(self):
        """search_videos should raise ValueError for empty query."""
        with pytest.raises(ValueError) as exc_info:
            YouTubeHelper.search_videos("")

        assert "empty" in str(exc_info.value).lower()

    def test_search_videos_whitespace_query_raises_error(self):
        """search_videos should raise ValueError for whitespace-only query."""
        with pytest.raises(ValueError) as exc_info:
            YouTubeHelper.search_videos("   ")

        assert "empty" in str(exc_info.value).lower()

    def test_search_videos_handles_missing_duration(self):
        """search_videos should handle videos with no duration."""
        mock_video = MagicMock()
        mock_video.video_id = "abc123"
        mock_video.title = "Test Video"
        mock_video.watch_url = "https://www.youtube.com/watch?v=abc123"
        mock_video.length = None
        mock_video.author = "Test Channel"

        mock_search = MagicMock()
        mock_search.videos = [mock_video]

        with patch("pytubefix.Search", return_value=mock_search):
            results = YouTubeHelper.search_videos("test", max_results=1)

        assert results[0]["duration"] == "0"

    def test_search_videos_handles_missing_author(self):
        """search_videos should handle videos with no author."""
        mock_video = MagicMock()
        mock_video.video_id = "abc123"
        mock_video.title = "Test Video"
        mock_video.watch_url = "https://www.youtube.com/watch?v=abc123"
        mock_video.length = 300
        mock_video.author = None

        mock_search = MagicMock()
        mock_search.videos = [mock_video]

        with patch("pytubefix.Search", return_value=mock_search):
            results = YouTubeHelper.search_videos("test", max_results=1)

        assert results[0]["channel"] == "Unknown"

    def test_search_videos_handles_search_exception(self):
        """search_videos should wrap exceptions from pytubefix."""
        with patch(
            "pytubefix.Search",
            side_effect=Exception("Network error"),
        ):
            with pytest.raises(Exception) as exc_info:
                YouTubeHelper.search_videos("test query")

        assert "Failed to search YouTube" in str(exc_info.value)

    def test_search_videos_returns_empty_list_when_no_results(self):
        """search_videos should return empty list when no videos found."""
        mock_search = MagicMock()
        mock_search.videos = []

        with patch("pytubefix.Search", return_value=mock_search):
            results = YouTubeHelper.search_videos("very obscure query xyz123")

        assert results == []


class TestYouTubeHelperExtractVideoId:
    """Test the extract_video_id functionality."""

    def test_extract_video_id_from_standard_url(self):
        """Should extract ID from standard youtube.com/watch URL."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        assert YouTubeHelper.extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_from_short_url(self):
        """Should extract ID from youtu.be short URL."""
        url = "https://youtu.be/dQw4w9WgXcQ"
        assert YouTubeHelper.extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_from_embed_url(self):
        """Should extract ID from embed URL."""
        url = "https://www.youtube.com/embed/dQw4w9WgXcQ"
        assert YouTubeHelper.extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_with_extra_params(self):
        """Should extract ID even with extra query parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s"
        assert YouTubeHelper.extract_video_id(url) == "dQw4w9WgXcQ"

    def test_extract_video_id_invalid_url_returns_none(self):
        """Should return None for invalid URLs."""
        assert YouTubeHelper.extract_video_id("not-a-url") is None
        assert YouTubeHelper.extract_video_id("https://google.com") is None


class TestYouTubeHelperValidateUrl:
    """Test the validate_url functionality."""

    def test_validate_url_accepts_valid_youtube_url(self):
        """Should return True for valid YouTube URLs."""
        assert YouTubeHelper.validate_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert YouTubeHelper.validate_url("https://youtu.be/dQw4w9WgXcQ")

    def test_validate_url_rejects_non_youtube_url(self):
        """Should return False for non-YouTube URLs."""
        assert not YouTubeHelper.validate_url("https://vimeo.com/12345")
        assert not YouTubeHelper.validate_url("https://google.com")

    def test_validate_url_rejects_youtube_url_without_video_id(self):
        """Should return False for YouTube URLs without video ID."""
        assert not YouTubeHelper.validate_url("https://www.youtube.com/")
        assert not YouTubeHelper.validate_url("https://www.youtube.com/feed")
