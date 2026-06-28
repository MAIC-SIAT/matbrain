#!/usr/bin/env bash
set -euo pipefail

# Download/warm MACE-MP models from the host Python environment, then let Docker
# mount the same cache directory. This avoids downloading inside Docker.
#
# Usage:
#   cd "$(git rev-parse --show-toplevel)/mcp-docker/uip-relax-mcp"
#   bash download_mace_models_host.sh
#
# Optional:
#   HOST_MACE_CACHE_DIR=/path/to/model-cache MACE_MODELS_TO_DOWNLOAD=small,medium bash download_mace_models_host.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
export HOST_MACE_CACHE_DIR="${HOST_MACE_CACHE_DIR:-${SCRIPT_DIR}/model-cache}"
export MACE_MODELS_TO_DOWNLOAD="${MACE_MODELS_TO_DOWNLOAD:-small,medium}"
export UIP_WARMUP_DEVICE="${UIP_WARMUP_DEVICE:-cpu}"
export UIP_DTYPE="${UIP_DTYPE:-float32}"

mkdir -p "${HOST_MACE_CACHE_DIR}"
chmod u+rwX "${HOST_MACE_CACHE_DIR}"

export XDG_CACHE_HOME="${HOST_MACE_CACHE_DIR}"
export MACE_CACHE_DIR="${HOST_MACE_CACHE_DIR}"
export MPLCONFIGDIR="${HOST_MACE_CACHE_DIR}/matplotlib"
mkdir -p "${MPLCONFIGDIR}"

"${PYTHON_BIN}" - <<'PY'
import importlib.util
missing = [pkg for pkg in ["torch", "mace"] if importlib.util.find_spec(pkg) is None]
if missing:
    raise SystemExit(
        "Missing host Python packages: "
        + ", ".join(missing)
        + "\nInstall them in the current environment first, for example:\n"
        + '  python -m pip install "mace-torch>=0.3.12"\n'
        + "If torch is not installed, install the CUDA/CPU torch build appropriate for this host."
    )
PY

cd "${SCRIPT_DIR}"
"${PYTHON_BIN}" download_mace_models.py
