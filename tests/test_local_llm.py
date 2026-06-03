"""Tests for local model path and archive handling."""

import tarfile

import pytest

from youtube_summariser.local_llm import LocalTransformersClient


def test_archive_model_path_extracts_to_cache(tmp_path, monkeypatch):
    """A local .tar.gz model archive should be extracted to the configured cache."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    source_dir = tmp_path / "source-model"
    source_dir.mkdir()
    (source_dir / "config.json").write_text("{}", encoding="utf-8")
    (source_dir / "tokenizer.json").write_text("{}", encoding="utf-8")

    archive_path = tmp_path / "model.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        archive.add(source_dir / "config.json", arcname="config.json")
        archive.add(source_dir / "tokenizer.json", arcname="tokenizer.json")

    cache_dir = tmp_path / "cache"
    client = LocalTransformersClient(
        {"model_path": str(archive_path), "cache_dir": str(cache_dir)}
    )

    model_dir = client._resolve_model_dir()

    assert model_dir.exists()
    assert (model_dir / "config.json").exists()
    assert (model_dir / ".complete").exists()
    assert model_dir.parent == cache_dir


def test_archive_extraction_rejects_path_traversal(tmp_path, monkeypatch):
    """Archives must not be able to write outside the cache directory."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    unsafe_file = tmp_path / "unsafe.txt"
    unsafe_file.write_text("nope", encoding="utf-8")

    archive_path = tmp_path / "unsafe.tar"
    with tarfile.open(archive_path, "w") as archive:
        archive.add(unsafe_file, arcname="../unsafe.txt")

    client = LocalTransformersClient(
        {"model_path": str(archive_path), "cache_dir": str(tmp_path / "cache")}
    )

    with pytest.raises(ValueError) as exc_info:
        client._resolve_model_dir()

    assert "Unsafe path" in str(exc_info.value)
