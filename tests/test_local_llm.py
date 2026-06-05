"""Tests for local model path and archive handling."""

import sys
import tarfile
import types

import pytest

from youtube_summariser.local_llm import (
    DEFAULT_GGUF_MODEL_FILE,
    DEFAULT_GGUF_MODEL_ID,
    DEFAULT_HF_MODEL_ID,
    DEFAULT_TRANSFORMERS_HF_MODEL_ID,
    LocalTransformersClient,
)


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
    client = LocalTransformersClient({"model_path": str(archive_path), "cache_dir": str(cache_dir)})

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


def test_hf_repo_id_downloads_to_configured_cache(tmp_path, monkeypatch):
    """A Hugging Face repo ID should be downloaded into the local model cache."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    downloaded_dir = tmp_path / "downloaded-model"
    calls = {}

    def fake_snapshot_download(**kwargs):
        calls.update(kwargs)
        downloaded_dir.mkdir()
        (downloaded_dir / "config.json").write_text("{}", encoding="utf-8")
        return str(downloaded_dir)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    cache_dir = tmp_path / "cache"
    client = LocalTransformersClient(
        {"model_path": DEFAULT_TRANSFORMERS_HF_MODEL_ID, "cache_dir": str(cache_dir)}
    )

    model_dir = client._resolve_model_dir()

    assert model_dir == downloaded_dir
    assert calls["repo_id"] == DEFAULT_TRANSFORMERS_HF_MODEL_ID
    assert calls["cache_dir"] == str(cache_dir / "huggingface")
    assert calls["local_files_only"] is False


def test_default_gguf_repo_downloads_q4_file(tmp_path, monkeypatch):
    """The default Hub model should download only the Q4_K_M GGUF artifact."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    downloaded_file = tmp_path / DEFAULT_GGUF_MODEL_FILE
    downloaded_file.write_text("gguf", encoding="utf-8")
    calls = {}

    def fake_hf_hub_download(**kwargs):
        calls.update(kwargs)
        return str(downloaded_file)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: object(), hf_hub_download=fake_hf_hub_download),
    )

    cache_dir = tmp_path / "cache"
    client = LocalTransformersClient(
        {"model_path": DEFAULT_HF_MODEL_ID, "cache_dir": str(cache_dir)}
    )

    model_path = client._resolve_gguf_model_path()

    assert model_path == downloaded_file
    assert calls["repo_id"] == DEFAULT_GGUF_MODEL_ID
    assert calls["filename"] == DEFAULT_GGUF_MODEL_FILE
    assert calls["cache_dir"] == str(cache_dir / "huggingface")
    assert calls["local_files_only"] is False


def test_generic_gguf_repo_quant_selector_lists_matching_file(tmp_path, monkeypatch):
    """repo:Q4_K_M should select a matching GGUF file from arbitrary GGUF repos."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    downloaded_file = tmp_path / "other-model-q4.gguf"
    downloaded_file.write_text("gguf", encoding="utf-8")
    calls = {}

    class FakeApi:
        def list_repo_files(self, **kwargs):
            calls["list"] = kwargs
            return ["README.md", "other-model.Q5_K_M.gguf", "other-model.Q4_K_M.gguf"]

    def fake_hf_hub_download(**kwargs):
        calls["download"] = kwargs
        return str(downloaded_file)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=FakeApi, hf_hub_download=fake_hf_hub_download),
    )

    client = LocalTransformersClient(
        {"model_path": "someone/other-model-GGUF:Q4_K_M", "cache_dir": str(tmp_path / "cache")}
    )

    model_path = client._resolve_gguf_model_path()

    assert model_path == downloaded_file
    assert calls["list"]["repo_id"] == "someone/other-model-GGUF"
    assert calls["download"]["filename"] == "other-model.Q4_K_M.gguf"


def test_local_gguf_file_is_used_directly(tmp_path, monkeypatch):
    """A local .gguf path should be accepted without archive extraction."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    model_file = tmp_path / "model.Q4_K_M.gguf"
    model_file.write_text("gguf", encoding="utf-8")

    client = LocalTransformersClient({"model_path": str(model_file)})

    assert client._resolve_gguf_model_path() == model_file
    assert client.model_label() == model_file.name


def test_gguf_context_size_reserves_largest_phase_output_cap(tmp_path, monkeypatch):
    """llama.cpp context should include the largest configured generation cap."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    model_file = tmp_path / "model.Q4_K_M.gguf"
    model_file.write_text("gguf", encoding="utf-8")

    client = LocalTransformersClient(
        {
            "model_path": str(model_file),
            "max_input_tokens": 1536,
            "max_tokens": 512,
            "map_max_tokens": 256,
            "intermediate_max_tokens": 256,
            "final_max_tokens": 1024,
        }
    )

    assert client._gguf_context_size() == 1536 + 1024 + 64


def test_gguf_repo_id_is_used_as_model_label(monkeypatch):
    """Repo IDs should be visible in output metadata."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)

    client = LocalTransformersClient({"model_path": DEFAULT_HF_MODEL_ID})

    assert client.model_label() == f"{DEFAULT_HF_MODEL_ID}:{DEFAULT_GGUF_MODEL_FILE}"


def test_gguf_chat_uses_llama_cpp_and_cleans_think_tags(tmp_path, monkeypatch):
    """GGUF generation should call llama.cpp and return cleaned assistant text."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    model_file = tmp_path / "model.gguf"
    model_file.write_text("gguf", encoding="utf-8")
    calls = {}

    class FakeLlama:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def tokenize(self, text, **kwargs):
            calls["tokenize_kwargs"] = kwargs
            return list(range(max(1, len(text) // 8)))

        def __call__(self, prompt, **kwargs):
            calls["prompt"] = prompt
            calls["generation"] = kwargs
            return {"choices": [{"text": "<think>\nnotes</think>\n\nFinal summary"}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", types.SimpleNamespace(Llama=FakeLlama))

    client = LocalTransformersClient(
        {"model_path": str(model_file), "max_input_tokens": 128, "max_tokens": 16}
    )

    assert client.count_chat_tokens("system", "user") > 0
    assert client.chat("system", "user") == "Final summary"
    assert calls["init"]["model_path"] == str(model_file)
    assert calls["generation"]["stream"] is False
    assert "<|im_start|>system" in calls["prompt"]


def test_missing_local_path_still_raises_clear_error(tmp_path, monkeypatch):
    """Nonexistent local paths should not be treated as model directories."""
    monkeypatch.delenv("YOUTUBE_SUMMARISER_LOCAL_MODEL", raising=False)
    missing_path = tmp_path / "missing-model"

    with pytest.raises(ValueError) as exc_info:
        LocalTransformersClient({"model_path": str(missing_path)})

    assert "Local model path does not exist" in str(exc_info.value)
    assert DEFAULT_HF_MODEL_ID in str(exc_info.value)
