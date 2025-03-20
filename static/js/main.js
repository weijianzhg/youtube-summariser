document.addEventListener('DOMContentLoaded', function() {
    // Initialize Feather icons
    feather.replace();

    const form = document.getElementById('summaryForm');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const errorMessage = document.getElementById('errorMessage');
    const summaryResult = document.getElementById('summaryResult');
    const summaryText = document.getElementById('summaryText');

    function showLoading() {
        loadingSpinner.classList.remove('d-none');
        errorMessage.classList.add('d-none');
        summaryResult.classList.add('d-none');
    }

    function hideLoading() {
        loadingSpinner.classList.add('d-none');
    }

    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.classList.remove('d-none');
    }

    function formatTimestampLinks(summary, videoId) {
        // Replace timestamps with clickable links
        return summary.replace(/\[(\d{2}:\d{2})\]/g, (match, timestamp) => {
            const [minutes, seconds] = timestamp.split(':').map(Number);
            const totalSeconds = minutes * 60 + seconds;
            return `<a href="https://youtu.be/${videoId}?t=${totalSeconds}" target="_blank" class="timestamp-link">${match}</a>`;
        });
    }

    function showSummary(summary, videoId) {
        // Format the summary with clickable timestamps
        const formattedSummary = formatTimestampLinks(summary, videoId);

        // Convert markdown-style sections to HTML with proper styling
        const htmlSummary = formattedSummary
            .replace(/^(#+)\s(.*?)$/gm, (match, hashes, title) => {
                const level = hashes.length + 2; // Start at h3 for better hierarchy
                return `<h${level}>${title}</h${level}>`;
            });

        summaryText.innerHTML = htmlSummary;
        summaryResult.classList.remove('d-none');
    }

    form.addEventListener('submit', async function(e) {
        e.preventDefault();

        const youtubeUrl = document.getElementById('youtubeUrl').value.trim();

        if (!youtubeUrl) {
            showError('Please enter a YouTube URL');
            return;
        }

        showLoading();

        try {
            const response = await fetch('/summarize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ url: youtubeUrl })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || 'Failed to generate summary');
            }

            showSummary(data.summary, data.video_id);
        } catch (error) {
            showError(error.message);
        } finally {
            hideLoading();
        }
    });
});