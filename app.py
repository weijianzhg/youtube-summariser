import os
import logging
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_helper import YouTubeHelper
from llm_client import LLMClient
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET")

# Initialize LLM client
llm = LLMClient()

SYSTEM_PROMPT = """You are a video summarization expert. Create a detailed summary of the video transcript with the following sections:

1. Main Topics: List the key topics discussed (2-3 sentences each)
2. Key Points: Highlight important information with timestamps
3. Detailed Summary: A comprehensive breakdown of the content (300-400 words)
4. Notable Quotes: Include 2-3 significant quotes with their timestamps
5. Timestamps for Important Moments: List key moments in the video

For timestamps in brackets like [MM:SS], maintain them in your response. Create clickable timestamps by formatting them as [MM:SS](https://youtu.be/VIDEO_ID?t=XXs) where XXs is the time in seconds."""


def summarize_transcript(transcript):
    """Summarize transcript using the configured LLM."""
    try:
        return llm.chat(SYSTEM_PROMPT, transcript)
    except Exception as e:
        logger.error(f"Error summarizing transcript: {str(e)}")
        raise Exception("Failed to generate summary")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        url = data.get('url')

        if not url:
            return jsonify({'error': 'No URL provided'}), 400

        # Use YouTubeHelper to extract video ID and get transcript
        video_id = YouTubeHelper.extract_video_id(url)
        if not video_id:
            return jsonify({'error': 'Invalid YouTube URL'}), 400

        # Validate URL before proceeding
        if not YouTubeHelper.validate_url(url):
            return jsonify({'error': 'Invalid YouTube URL format'}), 400

        # Get transcript using the helper class
        transcript = YouTubeHelper.get_transcript(video_id)
        if not transcript:
            return jsonify({'error': 'No transcript available for this video'}), 400

        # Generate summary
        summary = summarize_transcript(transcript)

        return jsonify({
            'summary': summary,
            'video_id': video_id
        })
    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
