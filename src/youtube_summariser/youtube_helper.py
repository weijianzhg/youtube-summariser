"""YouTube helper utilities for extracting video IDs and transcripts."""

import logging
import re
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import YouTubeTranscriptApi

logger = logging.getLogger(__name__)


class YouTubeHelper:
    """Helper class for YouTube video operations."""

    @staticmethod
    def format_timestamp(seconds: float) -> str:
        """Convert seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def extract_video_id(url: str) -> Optional[str]:
        """Extract YouTube video ID from various URL formats."""
        try:
            parsed_url = urlparse(url)

            # Handle youtu.be URLs
            if "youtu.be" in parsed_url.netloc:
                return parsed_url.path.strip("/")

            # Handle youtube.com URLs
            if "youtube.com" in parsed_url.netloc:
                # Try to get v parameter from query string
                if "v" in parse_qs(parsed_url.query):
                    return parse_qs(parsed_url.query)["v"][0]

                # Handle embed and direct video URLs
                path_parts = parsed_url.path.split("/")
                if "embed" in path_parts or "v" in path_parts:
                    return path_parts[-1]

            # If above methods fail, try regex pattern matching
            patterns = [
                r"(?:youtube\.com\/watch\?v=|youtu\.be\/)([\w-]+)",
                r"(?:youtube\.com\/embed\/)([\w-]+)",
                r"(?:youtube\.com\/v\/)([\w-]+)",
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            return None

        except Exception as e:
            logger.error(f"Error extracting video ID: {str(e)}")
            return None

    @staticmethod
    def get_transcript(video_id: str) -> str:
        """
        Get video transcript using youtube_transcript_api.

        Args:
            video_id: The YouTube video ID

        Returns:
            The formatted transcript text with timestamps

        Raises:
            Exception: If transcript cannot be retrieved or processed
        """
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.fetch(video_id).to_raw_data()

            # Format transcript with timestamps
            formatted_parts = []
            for entry in transcript_list:
                timestamp = YouTubeHelper.format_timestamp(entry["start"])
                text = entry["text"]
                formatted_parts.append(f"[{timestamp}] {text}")

            transcript_text = "\n".join(formatted_parts)

            if not transcript_text:
                raise Exception("Empty transcript received")

            return transcript_text

        except Exception as e:
            logger.error(f"Error getting transcript for video {video_id}: {str(e)}")
            raise Exception(f"Failed to get video transcript: {str(e)}")

    @staticmethod
    def validate_url(url: str) -> bool:
        """Validate if the given URL is a valid YouTube URL."""
        try:
            parsed_url = urlparse(url)

            # Check if domain contains youtube.com or youtu.be (substring match)
            # This handles subdomains like m.youtube.com, music.youtube.com, etc.
            is_youtube = "youtube.com" in parsed_url.netloc or "youtu.be" in parsed_url.netloc
            if not is_youtube:
                return False

            # Check if video ID can be extracted
            video_id = YouTubeHelper.extract_video_id(url)
            if not video_id:
                return False

            return True

        except Exception as e:
            logger.error(f"Error validating URL: {str(e)}")
            return False

    @staticmethod
    def get_available_transcript_languages(video_id: str) -> List[Dict[str, str]]:
        """Get list of available transcript languages for a video."""
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            available_transcripts = []

            for transcript in transcript_list:
                available_transcripts.append(
                    {"language_code": transcript.language_code, "language": transcript.language}
                )

            return available_transcripts

        except Exception as e:
            logger.error(f"Error getting available transcripts for video {video_id}: {str(e)}")
            raise Exception("Failed to get available transcripts")
