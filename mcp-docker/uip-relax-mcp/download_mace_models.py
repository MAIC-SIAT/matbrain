#!/usr/bin/env python3
"""Pre-download and warm up MACE-MP models into the mounted cache.

MACE stores auto-downloaded models under the XDG cache root. In Docker we set
XDG_CACHE_HOME=/app/model-cache and mount that path from the host so rollout
does not block on first use.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def _purge_mace_cache(cache_dir: Path) -> None:
    mace_cache = cache_dir / "mace"
    if mace_cache.exists():
        print(f"Removing possibly corrupted MACE cache: {mace_cache}", flush=True)
        shutil.rmtree(mace_cache)


def main() -> int:
    cache_dir = Path(os.getenv("XDG_CACHE_HOME", "/app/model-cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    os.environ.setdefault("MACE_CACHE_DIR", str(cache_dir))

    models = [
        item.strip()
        for item in os.getenv("MACE_MODELS_TO_DOWNLOAD", "small,medium").split(",")
        if item.strip()
    ]
    device = os.getenv("UIP_WARMUP_DEVICE", os.getenv("UIP_DEVICE", "cpu"))
    dtype = os.getenv("UIP_DTYPE", "float32")

    from mace.calculators import mace_mp

    print(f"XDG_CACHE_HOME={os.environ['XDG_CACHE_HOME']}")
    print(f"MACE_CACHE_DIR={os.environ['MACE_CACHE_DIR']}")
    print(f"Downloading/warming MACE-MP models: {models}")
    for model in models:
        print(f"Loading model={model} device={device} dtype={dtype}", flush=True)
        try:
            _ = mace_mp(model=model, device=device, default_dtype=dtype)
        except Exception as exc:
            message = str(exc)
            should_purge = os.getenv("MACE_PURGE_CACHE_ON_ERROR", "1") == "1"
            corrupt_markers = [
                "failed finding central directory",
                "PytorchStreamReader failed reading zip archive",
                "Model download failed",
                "no local model found",
            ]
            if should_purge and any(marker in message for marker in corrupt_markers):
                print(f"Model load failed, likely due to a corrupt cached checkpoint: {exc}", flush=True)
                _purge_mace_cache(cache_dir)
                print(f"Retrying model={model}", flush=True)
                _ = mace_mp(model=model, device=device, default_dtype=dtype)
            else:
                raise
        print(f"Loaded model={model}", flush=True)

    print("Cache contents:")
    for path in sorted(cache_dir.rglob("*")):
        if path.is_file():
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
