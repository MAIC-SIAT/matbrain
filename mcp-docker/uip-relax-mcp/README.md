# UIP Relax MCP

This service wraps MACE-MP through ASE for single-point energy, relaxation, and
dense discovery-reward signals.

## Pre-download MACE models

MACE auto-downloads model files under the XDG cache root. This service sets:

```text
XDG_CACHE_HOME=/app/model-cache
MACE_CACHE_DIR=/app/model-cache
```

and mounts `HOST_MACE_CACHE_DIR` from the host. The default is the local service
directory cache using relative paths:

```text
HOST_MACE_CACHE_DIR=./model-cache
HOST_UIP_LOG_DIR=./logs
```

Create these directories as the host user before warmup so Docker does not try
to create them:

```bash
cd "$(git rev-parse --show-toplevel)/mcp-docker/uip-relax-mcp"
mkdir -p model-cache logs
chmod u+rwX model-cache logs
```

### Option A: host-side download

If the host Python/conda environment already has `mace-torch` and `torch`, this
is the simplest route:

```bash
cd "$(git rev-parse --show-toplevel)/mcp-docker/uip-relax-mcp"
bash download_mace_models_host.sh
```

If needed:

```bash
python -m pip install "mace-torch>=0.3.12"
```

Make sure `.env` points Docker to the same directory:

```text
HOST_MACE_CACHE_DIR=./model-cache
```

### Option B: Docker-side download

Pre-download inside Docker before RL rollout:

```bash
cd "$(git rev-parse --show-toplevel)/mcp-docker/uip-relax-mcp"
sudo docker compose --profile warmup run --rm mace-model-warmup
```

By default this downloads:

```text
MACE_MODELS_TO_DOWNLOAD=small,medium
```

Change `.env` if you also want `large`.

For safer low-memory warmup, keep:

```text
UIP_WARMUP_DEVICE=cpu
```

Runtime inference can still use:

```text
UIP_DEVICE=cuda
NVIDIA_VISIBLE_DEVICES=0
```
