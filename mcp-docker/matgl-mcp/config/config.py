"""MatGL model configuration for the MCP service.

The old implementation loaded pinned local model directories that depended on
legacy DGL-era exports. This module now resolves official Hugging Face model
IDs through ``matgl.load_model`` and caches the live model objects in-process.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

import torch

import matgl


class MatGLConfig:
    """Model registry and loader helpers for MatGL-backed tools."""

    def __init__(self) -> None:
        self._model_cache: dict[str, Any] = {}
        self._cache_lock = threading.Lock()
        self._device = self._resolve_device()

        self.MODEL_REFS = {
            "PES": os.getenv("MATGL_PES_MODEL_REF", "materialyze/TensorNet-PES-MatPES-PBE-2025.2"),
            "QET_PES": os.getenv("MATGL_QET_PES_MODEL_REF", "materialyze/QET-PES-MatPES-PBE-2025.2"),
            "QET_PES_PBE": os.getenv("MATGL_QET_PES_MODEL_REF", "materialyze/QET-PES-MatPES-PBE-2025.2"),
            "QET_PES_R2SCAN": os.getenv(
                "MATGL_QET_PES_R2SCAN_MODEL_REF",
                "materialyze/QET-PES-MatPES-r2SCAN-2025.2",
            ),
            "EFORM": os.getenv("MATGL_EFORM_MODEL_REF", "materialyze/M3GNet-Eform-MP-2018.6.1"),
            "BAND_GAP_MFI": os.getenv(
                "MATGL_BAND_GAP_MODEL_REF",
                "materialyze/MEGNet-BandGap-mfi-MP-2019.4.1",
            ),
            "CHGNET_MPTRJ": os.getenv(
                "MATGL_CHGNET_MODEL_REF",
                str(Path(__file__).resolve().parents[1] / "models" / "CHGNet-MPtrj-2024.2.13-11M-PES"),
            ),
        }

    def resolve_model_ref(self, model_key_or_ref: str) -> str:
        return self.MODEL_REFS.get(model_key_or_ref, model_key_or_ref)

    @staticmethod
    def _resolve_device() -> torch.device:
        raw = MatGLConfig._normalize_device_env(os.getenv("MATGL_DEVICE", "cuda:0"), "cuda:0")
        if raw.startswith("cuda") and not torch.cuda.is_available():
            raw = "cpu"
        try:
            return torch.device(raw)
        except RuntimeError:
            fallback = "cuda:0" if torch.cuda.is_available() else "cpu"
            return torch.device(fallback)

    @staticmethod
    def _normalize_device_env(raw: str | None, default: str) -> str:
        value = (raw or default).strip()
        if " - " in value:
            value = value.split(" - ", 1)[0].strip()
        return value or default

    def _pin_model_to_device(self, model: Any) -> Any:
        for candidate in (model, getattr(model, "model", None)):
            if candidate is not None and hasattr(candidate, "to"):
                candidate.to(self._device)
        return model

    def load_model(self, model_key: str) -> Any:
        model_ref = self.resolve_model_ref(model_key)
        cached = self._model_cache.get(model_ref)
        if cached is not None:
            return cached

        with self._cache_lock:
            cached = self._model_cache.get(model_ref)
            if cached is None:
                load_path = self._resolve_local_model_path(model_ref) or model_ref
                cached = matgl.load_model(load_path)
                cached = self._pin_model_to_device(cached)
                self._model_cache[model_ref] = cached
        return cached

    @staticmethod
    def _resolve_local_model_path(model_ref: str) -> str | None:
        path = Path(model_ref)
        if path.exists():
            return str(path)
        if "/" not in model_ref:
            return None

        owner, name = model_ref.split("/", 1)
        repo_dir_name = f"models--{owner}--{name}"
        cache_roots = MatGLConfig._hf_cache_roots()
        for root in cache_roots:
            repo_dir = root / repo_dir_name
            snapshot = MatGLConfig._snapshot_dir(repo_dir)
            if snapshot is not None:
                return str(snapshot)
        return None

    @staticmethod
    def _hf_cache_roots() -> list[Path]:
        roots: list[Path] = []
        for raw in (
            os.getenv("MATGL_HF_CACHE"),
            os.getenv("HF_HUB_CACHE"),
            str(Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "hub"),
            "/root/.cache/huggingface/hub",
        ):
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path not in roots:
                roots.append(path)
        return roots

    @staticmethod
    def _snapshot_dir(repo_dir: Path) -> Path | None:
        required = ("model.pt", "state.pt", "model.json")
        ref_path = repo_dir / "refs" / "main"
        candidates: list[Path] = []
        if ref_path.exists():
            commit = ref_path.read_text(encoding="utf-8").strip()
            if commit:
                candidates.append(repo_dir / "snapshots" / commit)
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.exists():
            candidates.extend(sorted((p for p in snapshots_dir.iterdir() if p.is_dir()), reverse=True))
        for candidate in candidates:
            if all((candidate / filename).exists() for filename in required):
                return candidate
        return None

    def get_model_ref(self, model_key: str) -> str:
        if model_key not in self.MODEL_REFS:
            raise ValueError(f"Unknown model key: {model_key}. Available keys: {list(self.MODEL_REFS.keys())}")
        return self.MODEL_REFS[model_key]

    def clear_model_cache(self) -> int:
        with self._cache_lock:
            count = len(self._model_cache)
            self._model_cache.clear()
        return count

    @property
    def device(self) -> torch.device:
        return self._device


matgl_config = MatGLConfig()

MODEL_REFS = matgl_config.MODEL_REFS


def load_model(model_key: str) -> Any:
    return matgl_config.load_model(model_key)


def get_model_ref(model_key: str) -> str:
    return matgl_config.get_model_ref(model_key)


def resolve_model_ref(model_key_or_ref: str) -> str:
    return matgl_config.resolve_model_ref(model_key_or_ref)
