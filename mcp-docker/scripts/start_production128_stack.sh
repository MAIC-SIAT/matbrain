#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PROFILE="${PROFILE:-production128}"
MATGL_SERVER_URL="${MATGL_SERVER_URL:-http://localhost:5668/sse}"
MATGL_PREWARM_MODELS="${MATGL_PREWARM_MODELS:-EFORM,BAND_GAP_MFI}"
MATGL_PREWARM_CONCURRENCY="${MATGL_PREWARM_CONCURRENCY:-64}"
UIP_PREWARM_MODELS="${UIP_PREWARM_MODELS:-small,medium}"
UIP_PREWARM_STRICT="${UIP_PREWARM_STRICT:-0}"
VERIFY_TIMEOUT="${VERIFY_TIMEOUT:-90}"
PREWARM_TIMEOUT="${PREWARM_TIMEOUT:-180}"
STACK_SCOPE="${STACK_SCOPE:-teacher-trace-ready}"
USE_SUDO="${USE_SUDO:-0}"
BUILD_IMAGES="${BUILD_IMAGES:-1}"
RECREATE_SHARED_NETWORK="${RECREATE_SHARED_NETWORK:-0}"
VERIFY_MODE="${VERIFY_MODE:-auto}"

cd "$ROOT_DIR"

stack_cmd=(python3 mcp-docker/mat_mcp_stack.py)
if [[ "$USE_SUDO" == "1" ]]; then
  stack_cmd+=(--sudo)
fi
if [[ "$BUILD_IMAGES" == "1" ]]; then
  stack_cmd+=(--build)
fi

printf '[0/3] ensuring shared docker network\n'
if [[ "$RECREATE_SHARED_NETWORK" == "1" ]]; then
  "${stack_cmd[@]}" recreate-shared-network --no-proxy
else
  "${stack_cmd[@]}" ensure-shared-network --no-proxy
fi

stack_cmd+=(up --profile "$PROFILE" --no-proxy)
if [[ "$STACK_SCOPE" == "gpu-only" ]]; then
  stack_cmd+=(--gpu-only)
fi

printf '[1/3] starting stack with profile=%s\n' "$PROFILE"
"${stack_cmd[@]}"

printf '[2/3] prewarming MatGL models: %s\n' "$MATGL_PREWARM_MODELS"
.venv/bin/python mcp-docker/scripts/prewarm_matgl_stack.py \
  --server-url "$MATGL_SERVER_URL" \
  --model-refs "$MATGL_PREWARM_MODELS" \
  --concurrency "$MATGL_PREWARM_CONCURRENCY" \
  --timeout "$PREWARM_TIMEOUT"

if [[ -n "$UIP_PREWARM_MODELS" ]]; then
  printf '[2b/3] prewarming UIP MACE models: %s\n' "$UIP_PREWARM_MODELS"
  if ! PYTHON_BIN=.venv/bin/python MACE_MODELS_TO_DOWNLOAD="$UIP_PREWARM_MODELS" \
    bash mcp-docker/uip-relax-mcp/download_mace_models_host.sh; then
    if [[ "$UIP_PREWARM_STRICT" == "1" ]]; then
      printf 'UIP prewarm failed and UIP_PREWARM_STRICT=1; aborting.\n' >&2
      exit 1
    fi
    printf 'WARN: UIP prewarm failed; continuing startup. UIP tools may later fail with missing local model.\n' >&2
  fi
fi

printf '[3/3] running quick verification\n'
effective_verify_mode="$VERIFY_MODE"
if [[ "$effective_verify_mode" == "auto" ]]; then
  if [[ "$STACK_SCOPE" == "gpu-only" ]]; then
    effective_verify_mode="gpu-only"
  else
    effective_verify_mode="teacher-trace-ready"
  fi
fi

if [[ "$effective_verify_mode" == "gpu-only" ]]; then
  tmp_verify_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_verify_dir"' EXIT
  python3 mcp-docker/scripts/make_remote_tool_configs.py \
    --host 127.0.0.1 \
    --output-dir "$tmp_verify_dir" \
    --inputs mcp-docker/configs/structure_property_tools.yaml mcp-docker/configs/evidence_tools.yaml
  local_main_config="$tmp_verify_dir/structure_property_tools.127.0.0.1.yaml"
  local_evidence_config="$tmp_verify_dir/evidence_tools.127.0.0.1.yaml"
  curl -fsS http://localhost:5668/health >/dev/null
  curl -fsS http://localhost:5669/health >/dev/null
  curl -fsS http://localhost:5679/health >/dev/null
  python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
    --main-config "$local_main_config" \
    --evidence-config "$local_evidence_config" \
    --tags matgl \
    --timeout "$VERIFY_TIMEOUT"
  python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
    --main-config "$local_main_config" \
    --evidence-config "$local_evidence_config" \
    --tags mace \
    --timeout "$VERIFY_TIMEOUT"
elif [[ "$effective_verify_mode" == "teacher-trace-ready" ]]; then
  tmp_verify_dir="$(mktemp -d)"
  trap 'rm -rf "$tmp_verify_dir"' EXIT
  python3 mcp-docker/scripts/make_remote_tool_configs.py \
    --host 127.0.0.1 \
    --output-dir "$tmp_verify_dir" \
    --inputs mcp-docker/configs/structure_property_tools.yaml mcp-docker/configs/evidence_tools.yaml
  local_main_config="$tmp_verify_dir/structure_property_tools.127.0.0.1.yaml"
  local_evidence_config="$tmp_verify_dir/evidence_tools.127.0.0.1.yaml"
  for url in \
    http://127.0.0.1:5668/health \
    http://127.0.0.1:5669/health \
    http://127.0.0.1:5672/pymatgen/health \
    http://127.0.0.1:5674/health \
    http://127.0.0.1:5675/health \
    http://127.0.0.1:5676/health \
    http://127.0.0.1:5679/health \
    http://127.0.0.1:5685/health
  do
    curl -fsS "$url" >/dev/null
  done
  python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
    --coverage-only \
    --main-config "$local_main_config" \
    --evidence-config "$local_evidence_config"
  python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
    --main-config "$local_main_config" \
    --evidence-config "$local_evidence_config" \
    --tags matgl \
    --timeout "$VERIFY_TIMEOUT"
  python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
    --main-config "$local_main_config" \
    --evidence-config "$local_evidence_config" \
    --tags mace \
    --timeout "$VERIFY_TIMEOUT"
else
  python3 mcp-docker/mat_mcp_stack.py verify --tags quick --timeout "$VERIFY_TIMEOUT"
fi

printf 'Stack ready.\n'
