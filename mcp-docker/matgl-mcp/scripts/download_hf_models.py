"""Download MatGL Hugging Face model artifacts into the runtime cache."""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_REFS = (
    "materialyze/TensorNet-PES-MatPES-PBE-2025.2",
    "materialyze/QET-PES-MatPES-PBE-2025.2",
    "materialyze/QET-PES-MatPES-r2SCAN-2025.2",
    "materialyze/M3GNet-Eform-MP-2018.6.1",
    "materialyze/MEGNet-BandGap-mfi-MP-2019.4.1",
)


def _model_refs() -> list[str]:
    refs = [
        os.getenv("MATGL_PES_MODEL_REF"),
        os.getenv("MATGL_QET_PES_MODEL_REF"),
        os.getenv("MATGL_QET_PES_R2SCAN_MODEL_REF"),
        os.getenv("MATGL_EFORM_MODEL_REF"),
        os.getenv("MATGL_BAND_GAP_MODEL_REF"),
    ]
    return list(dict.fromkeys(ref for ref in refs if ref)) or list(DEFAULT_MODEL_REFS)


def main() -> None:
    cache_dir = Path(
        os.getenv("MATGL_HF_CACHE")
        or os.getenv("HF_HUB_CACHE")
        or Path(os.getenv("HF_HOME", "/root/.cache/huggingface")) / "hub"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(f"cache_dir={cache_dir}", flush=True)
    for ref in _model_refs():
        print(f"downloading {ref}", flush=True)
        path = snapshot_download(repo_id=ref, cache_dir=str(cache_dir))
        print(f"cached {ref} -> {path}", flush=True)


if __name__ == "__main__":
    main()
