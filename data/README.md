# Fine-Tuning Data

This directory contains the first curated data pass for a local YouTube summarization model.

## Collected Source

| Dataset | License | Rows | Why it is included |
| --- | --- | ---: | --- |
| [ClarityClips/youtube-video-summarization](https://huggingface.co/datasets/ClarityClips/youtube-video-summarization) | MIT | 109 source, 103 usable | Direct YouTube video `transcript` to `summarize` pairs. Small, but the schema matches the app's transcript summarization task closely. |

The processed files are chat-style JSONL:

- `processed/youtube_summary_sft_train.jsonl`
- `processed/youtube_summary_sft_eval.jsonl`
- `processed/youtube_summary_sft_manifest.json`

The Tuned Tensor-compatible upload file is:

- `tuned_tensor/youtube_summary_tuned_tensor.jsonl`
- `tuned_tensor/youtube_summary_tuned_tensor_manifest.json`
- `tuned_tensor/youtube_summary_tuned_tensor_sources.jsonl`

The recommended Tuned Tensor upload file is the capped variant:

- `tuned_tensor/youtube_summary_tuned_tensor_capped.jsonl`
- `tuned_tensor/youtube_summary_tuned_tensor_capped_manifest.json`
- `tuned_tensor/youtube_summary_tuned_tensor_capped_sources.jsonl`

Each row contains:

- `messages`: system, user, and assistant messages for supervised fine tuning.
- `source`: dataset id, dataset URL, declared license, source commit SHA, source row, and video metadata.

Each Tuned Tensor upload row contains only:

- `input`: title, URL, categories, and transcript.
- `output`: target markdown summary.

Use the system prompt in `tuned_tensor/youtube_summary_tuned_tensor_manifest.json` when creating the Tuned Tensor behaviour spec. Tuned Tensor applies the behaviour spec system prompt at run time, so the upload JSONL intentionally does not include chat `messages`.

The capped file keeps rows under a fixed character budget (`16,000` input chars, `6,000` output chars). This exists because the first Qwen 4B run failed during baseline evaluation when a single example produced `7,937` input tokens plus `256` reserved output tokens, exceeding the active `8,192` token serving window by one token.

Regenerate the files with:

```bash
uv run --no-project --with pyarrow python scripts/prepare_finetune_dataset.py
```

Upload to Tuned Tensor with:

```bash
tt datasets upload data/tuned_tensor/youtube_summary_tuned_tensor_capped.jsonl \
  --name "youtube-summary-mit-seed-capped" \
  --description "MIT-licensed YouTube transcript-summary seed dataset capped for 8k-context eval"
```

## Candidate Sources Not Collected Yet

| Dataset | License signal | Status | Reason |
| --- | --- | --- | --- |
| [PleIAs/YouTube-Commons](https://huggingface.co/datasets/PleIAs/YouTube-Commons) | CC-BY-4.0 | Candidate domain corpus | Large transcript-only corpus from CC-BY YouTube videos. Useful for continued pretraining or summary generation with a local teacher, but not direct SFT pairs. |
| [jamescalam/youtube-transcriptions](https://huggingface.co/datasets/jamescalam/youtube-transcriptions) | AFL-3.0 | Candidate domain corpus | English technical YouTube transcript chunks. No summaries, and rows need to be merged by video before use. |
| [AndresR2909/youtube_transcriptions_summaries_gpt4](https://huggingface.co/datasets/AndresR2909/youtube_transcriptions_summaries_gpt4) | No declared license found | Skipped | Larger transcript-summary set, but not suitable for a public Hugging Face upload without license clearance. |
| [Svngoku/youtube-summarization-sft](https://huggingface.co/datasets/Svngoku/youtube-summarization-sft) | No declared license found | Skipped | Already in SFT format, but no license was declared in the dataset metadata checked on 2026-06-02. |
| [emirunlu26/turkish-youtube-text-summarization](https://huggingface.co/datasets/emirunlu26/turkish-youtube-text-summarization) | Apache-2.0 | Not collected | Good permissive dataset, but Turkish rather than the app's current English-first target. |

## Publication Note

The selected dataset has a declared MIT license in its Hugging Face metadata. Before publishing a derived dataset, keep the source attribution and license metadata, and do a final provenance check on the underlying YouTube transcript rights.
