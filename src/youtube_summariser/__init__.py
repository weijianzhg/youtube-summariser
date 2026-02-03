"""
YouTube Summariser - Summarize YouTube videos using AI.

A command-line tool that extracts transcripts from YouTube videos
and generates structured summaries using OpenAI, Anthropic, or OpenRouter models.
"""

__version__ = "0.5.0"
__author__ = "Weijian Zhang"

from .llm_client import LLMClient
from .youtube_helper import YouTubeHelper

__all__ = ["LLMClient", "YouTubeHelper", "__version__"]
