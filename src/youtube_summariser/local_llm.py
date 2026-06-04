"""Local Transformers backend for summarization."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import tarfile
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

LOCAL_MODEL_ENV = "YOUTUBE_SUMMARISER_LOCAL_MODEL"
DEFAULT_HF_MODEL_ID = "weijianzhg/youtube-summariser-qwen3.5-4b"
DEFAULT_CACHE_DIR = "~/.cache/youtube-summariser/models"
ARCHIVE_SUFFIXES = (".tar.gz", ".tgz", ".tar")


class LocalTransformersClient:
    """Lazy local model runner backed by Hugging Face Transformers."""

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.model_reference = self._configured_model_reference()
        self.model_path = self._configured_model_path(self.model_reference)
        self._model_dir: Path | None = None
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._uses_device_map = False

    def model_label(self) -> str:
        """Return a human-readable model label for output metadata."""
        configured_label = self.config.get("model")
        if configured_label:
            return str(configured_label)
        if self.model_path is not None:
            return _strip_archive_suffix(self.model_path.name)
        return self.model_reference

    def get_max_input_tokens(self) -> int:
        """Return the configured prompt-token limit."""
        return int(self.config.get("max_input_tokens", 1536) or 0)

    def set_truncation_allowed(self, allowed: bool) -> None:
        """Control whether over-budget prompts may be truncated."""
        self.config["allow_truncate"] = bool(allowed)

    def count_chat_tokens(self, system_prompt: str, user_message: str) -> int:
        """Count chat-template tokens for a local prompt without truncating it."""
        inputs = self._encode_chat_inputs(system_prompt, user_message)
        return int(inputs["input_ids"].shape[-1])

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Generate a complete local response."""
        model, tokenizer, torch = self._ensure_loaded()
        inputs = self._prepare_inputs(system_prompt, user_message)
        with torch.inference_mode():
            outputs = model.generate(**inputs, **self._generation_kwargs(tokenizer))

        prompt_length = inputs["input_ids"].shape[-1]
        generated_ids = outputs[0][prompt_length:]
        return _clean_response(tokenizer.decode(generated_ids, skip_special_tokens=True))

    def stream_chat(self, system_prompt: str, user_message: str) -> Iterator[str]:
        """Stream a local response as it is generated."""
        model, tokenizer, _torch = self._ensure_loaded()
        inputs = self._prepare_inputs(system_prompt, user_message)

        try:
            from transformers import TextIteratorStreamer
        except ImportError as exc:
            raise ValueError(_missing_local_dependencies_message()) from exc

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = {
            **inputs,
            **self._generation_kwargs(tokenizer),
            "streamer": streamer,
        }
        errors: list[BaseException] = []

        def generate_in_background() -> None:
            try:
                model.generate(**generation_kwargs)
            except BaseException as exc:  # pragma: no cover - exercised via caller behavior
                errors.append(exc)
                streamer.on_finalized_text("", stream_end=True)

        thread = threading.Thread(target=generate_in_background, daemon=True)
        thread.start()
        for text in streamer:
            yield text
        thread.join()

        if errors:
            raise RuntimeError(f"Local model generation failed: {errors[0]}") from errors[0]

    def _configured_model_reference(self) -> str:
        raw_reference = (
            os.environ.get(LOCAL_MODEL_ENV)
            or self.config.get("model_path")
            or self.config.get("path")
        )
        if not raw_reference or not str(raw_reference).strip():
            raise ValueError(
                "Local model path not configured. Pass --local to download the default model, "
                "set YOUTUBE_SUMMARISER_LOCAL_MODEL, add local.model_path to config.yaml, "
                "or pass --local-model."
            )
        return str(raw_reference).strip()

    def _configured_model_path(self, reference: str) -> Path | None:
        model_path = _expand_path(reference)
        if not model_path.exists():
            if _is_hf_repo_id(reference):
                return None
            raise ValueError(
                f"Local model path does not exist: {model_path}. "
                f"To download from Hugging Face, pass a repo ID such as {DEFAULT_HF_MODEL_ID}."
            )
        if model_path.is_file() and not _is_archive(model_path):
            raise ValueError(
                "Local model path must be a Hugging Face repo ID, an extracted model directory, "
                "or a .tar/.tar.gz archive: "
                f"{model_path}"
            )
        return model_path

    def _ensure_loaded(self):
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return self._model, self._tokenizer, self._torch

        try:
            import torch
            import transformers
        except ImportError as exc:
            raise ValueError(_missing_local_dependencies_message()) from exc

        tokenizer = self._ensure_tokenizer_loaded()
        model_dir = self._model_dir or self._resolve_model_dir()
        model = self._load_model(transformers, model_dir, torch)
        self._move_model_if_needed(model, torch)
        model.eval()

        self._model_dir = model_dir
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        return self._model, self._tokenizer, self._torch

    def _ensure_tokenizer_loaded(self):
        if self._tokenizer is not None:
            return self._tokenizer

        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ValueError(_missing_local_dependencies_message()) from exc

        model_dir = self._model_dir or self._resolve_model_dir()
        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            local_files_only=True,
            trust_remote_code=self._trust_remote_code(),
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        self._model_dir = model_dir
        self._tokenizer = tokenizer
        return tokenizer

    def _resolve_model_dir(self) -> Path:
        if self.model_path is None:
            return self._find_model_root(self._download_hf_model(self.model_reference))
        if self.model_path.is_dir():
            return self._find_model_root(self.model_path)
        return self._find_model_root(self._extract_model_archive(self.model_path))

    def _download_hf_model(self, repo_id: str) -> Path:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ValueError(_missing_local_dependencies_message()) from exc

        cache_root = _expand_path(str(self.config.get("cache_dir", DEFAULT_CACHE_DIR)))
        cache_root.mkdir(parents=True, exist_ok=True)
        snapshot_kwargs: dict[str, Any] = {
            "repo_id": repo_id,
            "cache_dir": str(cache_root / "huggingface"),
            "local_files_only": bool(self.config.get("local_files_only", False)),
        }
        revision = self.config.get("revision")
        if revision:
            snapshot_kwargs["revision"] = str(revision)

        logger.info("Downloading local model %s from Hugging Face Hub", repo_id)
        try:
            model_dir = Path(snapshot_download(**snapshot_kwargs))
        except Exception as exc:
            raise ValueError(
                f"Could not download local model '{repo_id}' from Hugging Face. "
                "Check the repo ID and your network connection. If you meant a local path, "
                f"make sure it exists. Details: {exc}"
            ) from exc
        return model_dir

    def _extract_model_archive(self, archive_path: Path) -> Path:
        cache_root = _expand_path(str(self.config.get("cache_dir", DEFAULT_CACHE_DIR)))
        cache_root.mkdir(parents=True, exist_ok=True)
        target = cache_root / _archive_cache_name(archive_path)
        complete_marker = target / ".complete"

        if complete_marker.exists():
            try:
                self._find_model_root(target)
                return target
            except ValueError:
                pass

        temp_target = cache_root / f"{target.name}.tmp"
        if temp_target.exists():
            shutil.rmtree(temp_target)
        temp_target.mkdir(parents=True)

        logger.info("Extracting local model archive %s to %s", archive_path, target)
        try:
            _safe_extract_tar(archive_path, temp_target)
            complete_marker_tmp = temp_target / ".complete"
            complete_marker_tmp.write_text(str(archive_path), encoding="utf-8")
            if target.exists():
                shutil.rmtree(target)
            temp_target.replace(target)
        except Exception:
            if temp_target.exists():
                shutil.rmtree(temp_target)
            raise

        return target

    def _find_model_root(self, model_path: Path) -> Path:
        if (model_path / "config.json").exists():
            return model_path

        config_files = list(model_path.glob("*/config.json"))
        if len(config_files) == 1:
            return config_files[0].parent

        raise ValueError(
            "Could not find config.json in local model path. "
            f"Expected a Hugging Face model directory: {model_path}"
        )

    def _load_model(self, transformers_module, model_dir: Path, torch_module):
        model_kwargs: dict[str, Any] = {
            "local_files_only": True,
            "trust_remote_code": self._trust_remote_code(),
        }
        torch_dtype = self._torch_dtype(torch_module)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        device_map = self.config.get("device_map")
        if device_map:
            model_kwargs["device_map"] = device_map
        elif self._device_setting() == "auto" and torch_module.cuda.is_available():
            model_kwargs["device_map"] = "auto"
        self._uses_device_map = "device_map" in model_kwargs

        errors: list[str] = []
        for class_name in self._model_class_candidates():
            model_class = getattr(transformers_module, class_name, None)
            if model_class is None:
                continue
            try:
                return model_class.from_pretrained(model_dir, **model_kwargs)
            except Exception as exc:
                errors.append(f"{class_name}: {exc}")

        joined_errors = "; ".join(errors) if errors else "no compatible auto model class found"
        raise ValueError(
            "Unable to load local model with Transformers. "
            "Make sure the local optional dependencies are current enough for this checkpoint. "
            f"Details: {joined_errors}"
        )

    def _model_class_candidates(self) -> list[str]:
        configured = self.config.get("model_class")
        if configured:
            if isinstance(configured, str):
                return [configured]
            return [str(item) for item in configured]
        return ["AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForVision2Seq"]

    def _prepare_inputs(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        _model, _tokenizer, torch = self._ensure_loaded()
        inputs = self._encode_chat_inputs(system_prompt, user_message)

        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        inputs = self._truncate_inputs(inputs, torch)
        input_device = self._input_device()
        return {
            key: value.to(input_device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _encode_chat_inputs(self, system_prompt: str, user_message: str) -> dict[str, Any]:
        tokenizer = self._ensure_tokenizer_loaded()
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        template_kwargs = {
            "add_generation_prompt": True,
            "return_tensors": "pt",
            "tokenize": True,
        }
        try:
            encoded = tokenizer.apply_chat_template(
                messages,
                return_dict=True,
                enable_thinking=False,
                **template_kwargs,
            )
        except TypeError:
            try:
                encoded = tokenizer.apply_chat_template(
                    messages,
                    return_dict=True,
                    **template_kwargs,
                )
            except TypeError:
                encoded = tokenizer.apply_chat_template(messages, **template_kwargs)

        if isinstance(encoded, Mapping):
            inputs = dict(encoded)
        else:
            inputs = {"input_ids": encoded}
        return inputs

    def _truncate_inputs(self, inputs: dict[str, Any], torch_module) -> dict[str, Any]:
        max_input_tokens = self.get_max_input_tokens()
        input_ids = inputs["input_ids"]
        input_length = input_ids.shape[-1]
        if max_input_tokens <= 0 or input_length <= max_input_tokens:
            return inputs
        if not bool(self.config.get("allow_truncate", False)):
            raise ValueError(
                "Local model input is too long for the configured context "
                f"({input_length:,} input tokens, limit {max_input_tokens:,}). "
                "Use long-video mode or pass --allow-truncate for a partial smoke test."
            )

        head_tokens = max_input_tokens // 2
        tail_tokens = max_input_tokens - head_tokens
        truncated: dict[str, Any] = {}
        for key, value in inputs.items():
            if (
                hasattr(value, "shape")
                and len(value.shape) >= 2
                and value.shape[-1] == input_length
            ):
                truncated[key] = torch_module.cat(
                    (value[..., :head_tokens], value[..., -tail_tokens:]),
                    dim=-1,
                )
            else:
                truncated[key] = value

        logger.warning(
            "Local model input was truncated from %s to %s tokens",
            input_length,
            max_input_tokens,
        )
        return truncated

    def _generation_kwargs(self, tokenizer) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.config.get("max_tokens", 512)),
            "do_sample": bool(self.config.get("do_sample", False)),
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        if tokenizer.eos_token_id is not None:
            kwargs["eos_token_id"] = tokenizer.eos_token_id
        if kwargs["do_sample"]:
            kwargs["temperature"] = float(self.config.get("temperature", 0.2))
            kwargs["top_p"] = float(self.config.get("top_p", 0.95))
        repetition_penalty = self.config.get("repetition_penalty")
        if repetition_penalty is not None:
            kwargs["repetition_penalty"] = float(repetition_penalty)
        return kwargs

    def _input_device(self):
        if self._model is None:
            return "cpu"
        device = getattr(self._model, "device", None)
        if device is not None and str(device) != "meta":
            return device
        try:
            return next(self._model.parameters()).device
        except StopIteration:
            return "cpu"

    def _move_model_if_needed(self, model, torch_module) -> None:
        if self._uses_device_map:
            return

        device = self._device_setting()
        if device == "auto":
            if (
                getattr(torch_module.backends, "mps", None)
                and torch_module.backends.mps.is_available()
            ):
                device = "mps"
            elif torch_module.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        if device != "cpu":
            model.to(device)

    def _torch_dtype(self, torch_module):
        dtype = str(self.config.get("torch_dtype", "auto")).lower()
        if dtype == "auto":
            if self._device_setting() == "auto" and (
                getattr(torch_module.backends, "mps", None)
                and torch_module.backends.mps.is_available()
            ):
                return torch_module.float16
            return "auto"
        mapping = {
            "float16": torch_module.float16,
            "fp16": torch_module.float16,
            "bfloat16": torch_module.bfloat16,
            "bf16": torch_module.bfloat16,
            "float32": torch_module.float32,
            "fp32": torch_module.float32,
        }
        return mapping.get(dtype, dtype)

    def _device_setting(self) -> str:
        return str(self.config.get("device", "auto")).lower()

    def _trust_remote_code(self) -> bool:
        return bool(self.config.get("trust_remote_code", False))


def _missing_local_dependencies_message() -> str:
    return (
        "Local provider requires optional dependencies. Install them with "
        "`pip install -e .[local]` from this repo, or "
        "`pip install 'youtube-summariser[local]'` when installed from a package."
    )


def _expand_path(path_value: str) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(path_value)))
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def _is_archive(path: Path) -> bool:
    return any(str(path).endswith(suffix) for suffix in ARCHIVE_SUFFIXES)


def _is_hf_repo_id(reference: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9][\w.-]*/[A-Za-z0-9][\w.-]*", reference))


def _strip_archive_suffix(name: str) -> str:
    for suffix in ARCHIVE_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _archive_cache_name(archive_path: Path) -> str:
    stat = archive_path.stat()
    identity = f"{archive_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return f"{_strip_archive_suffix(archive_path.name)}-{digest}"


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive_path, "r:*") as archive:
        members = archive.getmembers()
        for member in members:
            member_path = (destination / member.name).resolve()
            if not _is_relative_to(member_path, destination_resolved):
                raise ValueError(f"Unsafe path in local model archive: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not supported in local model archives: {member.name}")
        archive.extractall(destination, members=members)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _clean_response(text: str) -> str:
    cleaned = text.strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].lstrip()
    if cleaned.startswith("<think>"):
        cleaned = cleaned.removeprefix("<think>").lstrip()
    return cleaned
