#!/usr/bin/env python3
"""Prepare a small supervised fine-tuning dataset for YouTube summarization."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import urllib.request
from pathlib import Path

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover - exercised by users without pyarrow installed.
    pq = None


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "ClarityClips__youtube-video-summarization"
PROCESSED_DIR = ROOT / "data" / "processed"
TUNED_TENSOR_DIR = ROOT / "data" / "tuned_tensor"
RAW_FILE = RAW_DIR / "train-00000-of-00001.parquet"
TRAIN_FILE = PROCESSED_DIR / "youtube_summary_sft_train.jsonl"
EVAL_FILE = PROCESSED_DIR / "youtube_summary_sft_eval.jsonl"
MANIFEST_FILE = PROCESSED_DIR / "youtube_summary_sft_manifest.json"
TUNED_TENSOR_FILE = TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor.jsonl"
TUNED_TENSOR_SOURCES_FILE = TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor_sources.jsonl"
TUNED_TENSOR_MANIFEST_FILE = TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor_manifest.json"
CAPPED_TUNED_TENSOR_FILE = TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor_capped.jsonl"
CAPPED_TUNED_TENSOR_SOURCES_FILE = (
    TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor_capped_sources.jsonl"
)
CAPPED_TUNED_TENSOR_MANIFEST_FILE = (
    TUNED_TENSOR_DIR / "youtube_summary_tuned_tensor_capped_manifest.json"
)

DEFAULT_MAX_INPUT_CHARS = 16_000
DEFAULT_MAX_OUTPUT_CHARS = 6_000

SOURCE_DATASET = "ClarityClips/youtube-video-summarization"
SOURCE_URL = f"https://huggingface.co/datasets/{SOURCE_DATASET}"
SOURCE_FILE_URL = (
    "https://huggingface.co/datasets/"
    "ClarityClips/youtube-video-summarization/resolve/main/"
    "data/train-00000-of-00001.parquet"
)
SOURCE_SHA = "ca33e3f950fc45e5a804d090c4cea2e43cb8584f"
SOURCE_LICENSE = "mit"

SYSTEM_PROMPT = """Summarize this video transcript concisely.

## Output Format (use markdown):

### TL;DR
One paragraph capturing the essence (2-3 sentences).

### Key Takeaways
- Bullet points of the most important insights
- Include timestamps like [MM:SS] where relevant

### Detailed Summary
Comprehensive breakdown. Scale length to video complexity (~50 words per 5 minutes of content).

### Notable Quotes
1-3 memorable quotes with timestamps, if any stand out.

Preserve any timestamps from the transcript. Be concise - omit filler and tangents."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eval-ratio",
        type=float,
        default=0.1,
        help="Fraction of rows to reserve for evaluation. Default: 0.1.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/eval split. Default: 42.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download the source Parquet file even if it already exists.",
    )
    parser.add_argument(
        "--max-input-chars",
        type=int,
        default=DEFAULT_MAX_INPUT_CHARS,
        help=(
            "Maximum characters for capped Tuned Tensor input rows. "
            f"Default: {DEFAULT_MAX_INPUT_CHARS}."
        ),
    )
    parser.add_argument(
        "--max-output-chars",
        type=int,
        default=DEFAULT_MAX_OUTPUT_CHARS,
        help=(
            "Maximum characters for capped Tuned Tensor output rows. "
            f"Default: {DEFAULT_MAX_OUTPUT_CHARS}."
        ),
    )
    return parser.parse_args()


def download_source(force: bool) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists() and not force:
        return

    print(f"Downloading {SOURCE_DATASET} to {RAW_FILE}")
    with urllib.request.urlopen(SOURCE_FILE_URL) as response:
        RAW_FILE.write_bytes(response.read())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_string(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def as_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [as_string(item) for item in value if as_string(item)]
    return [as_string(value)] if as_string(value) else []


def load_rows() -> tuple[int, list[dict]]:
    if pq is None:
        raise RuntimeError(
            "pyarrow is required. Run: "
            "uv run --with pyarrow scripts/prepare_finetune_dataset.py"
        )

    table = pq.read_table(RAW_FILE)
    rows = table.to_pylist()
    raw_count = len(rows)
    clean_rows = []
    for index, row in enumerate(rows):
        transcript = as_string(row.get("transcript"))
        summary = as_string(row.get("summarize"))
        title = as_string(row.get("title"))
        link = as_string(row.get("link"))
        if not transcript or not summary:
            continue

        categories = as_list(row.get("category"))
        user_content = "\n".join(
            part
            for part in [
                f"Title: {title}" if title else "",
                f"Video URL: {link}" if link else "",
                f"Categories: {', '.join(categories)}" if categories else "",
                "Transcript:",
                transcript,
            ]
            if part
        )
        clean_rows.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": summary},
                ],
                "source": {
                    "dataset": SOURCE_DATASET,
                    "dataset_url": SOURCE_URL,
                    "license": SOURCE_LICENSE,
                    "source_sha": SOURCE_SHA,
                    "source_row": index,
                    "video_url": link,
                    "title": title,
                    "categories": categories,
                },
            }
        )
    return raw_count, clean_rows


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_tuned_tensor_record(record: dict) -> dict[str, str]:
    """Return the flat schema accepted by `tt datasets upload`."""
    messages = record["messages"]
    return {
        "input": messages[1]["content"],
        "output": messages[2]["content"],
    }


def truncate_text(value: str, max_chars: int, marker: str) -> tuple[str, bool]:
    if len(value) <= max_chars:
        return value, False

    marker_with_break = f"\n\n{marker}"
    keep = max(0, max_chars - len(marker_with_break))
    truncated = value[:keep].rstrip()
    return f"{truncated}{marker_with_break}", True


def build_capped_tuned_tensor_record(
    record: dict,
    *,
    max_input_chars: int,
    max_output_chars: int,
) -> tuple[dict[str, str], dict[str, int | bool]]:
    flat = build_tuned_tensor_record(record)
    capped_input, input_truncated = truncate_text(
        flat["input"],
        max_input_chars,
        "[Transcript truncated to fit the model evaluation context.]",
    )
    capped_output, output_truncated = truncate_text(
        flat["output"],
        max_output_chars,
        "[Summary truncated for capped training/evaluation.]",
    )
    return (
        {"input": capped_input, "output": capped_output},
        {
            "input_chars_before": len(flat["input"]),
            "input_chars_after": len(capped_input),
            "output_chars_before": len(flat["output"]),
            "output_chars_after": len(capped_output),
            "input_truncated": input_truncated,
            "output_truncated": output_truncated,
        },
    )


def length_stats(records: list[dict[str, str]]) -> dict[str, int]:
    return {
        "max_input_chars": max(len(row["input"]) for row in records),
        "max_output_chars": max(len(row["output"]) for row in records),
        "avg_input_chars": round(
            sum(len(row["input"]) for row in records) / len(records)
        ),
        "avg_output_chars": round(
            sum(len(row["output"]) for row in records) / len(records)
        ),
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.eval_ratio < 1:
        raise ValueError("--eval-ratio must be between 0 and 1")
    if args.max_input_chars <= 0:
        raise ValueError("--max-input-chars must be positive")
    if args.max_output_chars <= 0:
        raise ValueError("--max-output-chars must be positive")

    download_source(args.force_download)
    raw_count, records = load_rows()
    if not records:
        raise RuntimeError("No usable rows found in source dataset")

    rng = random.Random(args.seed)
    rng.shuffle(records)
    eval_count = max(1, round(len(records) * args.eval_ratio))
    eval_records = records[:eval_count]
    train_records = records[eval_count:]
    tuned_tensor_records = [build_tuned_tensor_record(record) for record in records]
    tuned_tensor_sources = [
        {"row": index, **record["source"]} for index, record in enumerate(records)
    ]
    capped_pairs = [
        build_capped_tuned_tensor_record(
            record,
            max_input_chars=args.max_input_chars,
            max_output_chars=args.max_output_chars,
        )
        for record in records
    ]
    capped_tuned_tensor_records = [record for record, _ in capped_pairs]
    capped_tuned_tensor_sources = [
        {
            "row": index,
            **records[index]["source"],
            "cap": cap_metadata,
        }
        for index, (_, cap_metadata) in enumerate(capped_pairs)
    ]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TUNED_TENSOR_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(TRAIN_FILE, train_records)
    write_jsonl(EVAL_FILE, eval_records)
    write_jsonl(TUNED_TENSOR_FILE, tuned_tensor_records)
    write_jsonl(TUNED_TENSOR_SOURCES_FILE, tuned_tensor_sources)
    write_jsonl(CAPPED_TUNED_TENSOR_FILE, capped_tuned_tensor_records)
    write_jsonl(CAPPED_TUNED_TENSOR_SOURCES_FILE, capped_tuned_tensor_sources)

    manifest = {
        "created_by": "scripts/prepare_finetune_dataset.py",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "source_sha": SOURCE_SHA,
        "raw_file": str(RAW_FILE.relative_to(ROOT)),
        "raw_file_sha256": sha256_file(RAW_FILE),
        "format": "chat messages JSONL",
        "system_prompt": SYSTEM_PROMPT,
        "seed": args.seed,
        "eval_ratio": args.eval_ratio,
        "counts": {
            "source_rows": raw_count,
            "usable_rows": len(records),
            "train_rows": len(train_records),
            "eval_rows": len(eval_records),
        },
        "outputs": {
            "train": str(TRAIN_FILE.relative_to(ROOT)),
            "eval": str(EVAL_FILE.relative_to(ROOT)),
        },
    }
    MANIFEST_FILE.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tuned_tensor_manifest = {
        "created_by": "scripts/prepare_finetune_dataset.py",
        "source_dataset": SOURCE_DATASET,
        "source_url": SOURCE_URL,
        "source_license": SOURCE_LICENSE,
        "source_sha": SOURCE_SHA,
        "raw_file": str(RAW_FILE.relative_to(ROOT)),
        "raw_file_sha256": sha256_file(RAW_FILE),
        "format": "Tuned Tensor input/output JSONL",
        "schema": {"input": "string", "output": "string"},
        "system_prompt_for_behavior_spec": SYSTEM_PROMPT,
        "file_size_bytes": TUNED_TENSOR_FILE.stat().st_size,
        "max_upload_size_bytes": 100 * 1024 * 1024,
        "counts": {
            "source_rows": raw_count,
            "usable_rows": len(tuned_tensor_records),
        },
        "lengths": length_stats(tuned_tensor_records),
        "outputs": {
            "dataset": str(TUNED_TENSOR_FILE.relative_to(ROOT)),
            "source_sidecar": str(TUNED_TENSOR_SOURCES_FILE.relative_to(ROOT)),
        },
        "upload_command": (
            "tt datasets upload "
            f"{TUNED_TENSOR_FILE.relative_to(ROOT)} "
            '--name "youtube-summary-mit-seed" '
            '--description "MIT-licensed YouTube transcript-summary seed dataset"'
        ),
    }
    TUNED_TENSOR_MANIFEST_FILE.write_text(
        json.dumps(tuned_tensor_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    capped_manifest = {
        **tuned_tensor_manifest,
        "format": "Tuned Tensor input/output JSONL capped for 8k-context eval",
        "file_size_bytes": CAPPED_TUNED_TENSOR_FILE.stat().st_size,
        "caps": {
            "max_input_chars": args.max_input_chars,
            "max_output_chars": args.max_output_chars,
            "target_serving_context_tokens": 8192,
            "target_eval_output_tokens": 256,
            "rationale": (
                "The first Qwen 4B run failed during baseline evaluation when "
                "one prompt had 7937 input tokens and reserved 256 output tokens, "
                "exceeding the active 8192-token serving context by one token."
            ),
        },
        "counts": {
            **tuned_tensor_manifest["counts"],
            "input_rows_truncated": sum(
                1
                for source in capped_tuned_tensor_sources
                if source["cap"]["input_truncated"]
            ),
            "output_rows_truncated": sum(
                1
                for source in capped_tuned_tensor_sources
                if source["cap"]["output_truncated"]
            ),
        },
        "lengths": length_stats(capped_tuned_tensor_records),
        "outputs": {
            "dataset": str(CAPPED_TUNED_TENSOR_FILE.relative_to(ROOT)),
            "source_sidecar": str(CAPPED_TUNED_TENSOR_SOURCES_FILE.relative_to(ROOT)),
        },
        "upload_command": (
            "tt datasets upload "
            f"{CAPPED_TUNED_TENSOR_FILE.relative_to(ROOT)} "
            '--name "youtube-summary-mit-seed-capped" '
            '--description "MIT-licensed YouTube transcript-summary seed dataset capped for 8k-context eval"'
        ),
    }
    CAPPED_TUNED_TENSOR_MANIFEST_FILE.write_text(
        json.dumps(capped_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {len(train_records)} train rows to {TRAIN_FILE}")
    print(f"Wrote {len(eval_records)} eval rows to {EVAL_FILE}")
    print(f"Wrote manifest to {MANIFEST_FILE}")
    print(f"Wrote {len(tuned_tensor_records)} Tuned Tensor rows to {TUNED_TENSOR_FILE}")
    print(f"Wrote Tuned Tensor source sidecar to {TUNED_TENSOR_SOURCES_FILE}")
    print(f"Wrote Tuned Tensor manifest to {TUNED_TENSOR_MANIFEST_FILE}")
    print(
        f"Wrote {len(capped_tuned_tensor_records)} capped Tuned Tensor rows "
        f"to {CAPPED_TUNED_TENSOR_FILE}"
    )
    print(f"Wrote capped Tuned Tensor source sidecar to {CAPPED_TUNED_TENSOR_SOURCES_FILE}")
    print(f"Wrote capped Tuned Tensor manifest to {CAPPED_TUNED_TENSOR_MANIFEST_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
