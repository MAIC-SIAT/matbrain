#!/usr/bin/env bash
set -euo pipefail

# Prepare remote Mat-MCP tool configs for verl 0.9 native multi-turn rollout.
# Run inside the verl/SGLang Docker container from any directory.

ROOT_DIR="${ROOT_DIR:-/workspace/matbrain}"
MCP_HOST="${MCP_HOST:-127.0.0.1}"
TIMEOUT="${TIMEOUT:-180}"
RUN_ROOT="${RUN_ROOT:-/workspace/run}"
if [[ ! -d "$RUN_ROOT" || ! -w "$RUN_ROOT" ]]; then
  RUN_ROOT="${RUN_ROOT_FALLBACK:-/tmp}"
fi
WORK_DIR="${WORK_DIR:-$RUN_ROOT/matbrain_verl09_mcp}"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# MCP calls are cluster-local. Do not route them through an HTTP proxy.
export HTTP_PROXY=
export HTTPS_PROXY=
export ALL_PROXY=
export http_proxy=
export https_proxy=
export all_proxy=
export NO_PROXY="$MCP_HOST,localhost,127.0.0.1,0.0.0.0,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"

mkdir -p "$WORK_DIR/remote" "$WORK_DIR/verl09_native"
mkdir -p "$WORK_DIR/verl09_router"

remote_main="$WORK_DIR/remote/structure_property_tools.${MCP_HOST}.yaml"
remote_core="$WORK_DIR/remote/structure_property_core_tools.${MCP_HOST}.yaml"
remote_evidence="$WORK_DIR/remote/evidence_tools.${MCP_HOST}.yaml"
native_config="$WORK_DIR/verl09_native/mat_mcp_structure_property_${MCP_HOST//./_}_native.yaml"
core_native_config="$WORK_DIR/verl09_native/mat_mcp_structure_property_${MCP_HOST//./_}_core_native.yaml"
router_config="$WORK_DIR/verl09_router/mat_mcp_structure_property_${MCP_HOST//./_}_router.yaml"
router_catalog="$WORK_DIR/verl09_router/mat_mcp_structure_property_${MCP_HOST//./_}_catalog.json"
MAIN_CONFIG_PATH="${MAIN_CONFIG_PATH:-mcp-docker/configs/structure_property_tools.yaml}"
CORE_CONFIG_PATH="${CORE_CONFIG_PATH:-mcp-docker/configs/structure_property_core_tools.yaml}"
EVIDENCE_CONFIG_PATH="${EVIDENCE_CONFIG_PATH:-mcp-docker/configs/evidence_tools.yaml}"
MATGL_SERVER_URL="${MATGL_SERVER_URL:-http://${MCP_HOST}:5668/sse}"
MATGL_PREWARM_MODELS="${MATGL_PREWARM_MODELS:-EFORM,BAND_GAP_MFI}"
MATGL_PREWARM_CONCURRENCY="${MATGL_PREWARM_CONCURRENCY:-64}"

python3 mcp-docker/scripts/make_remote_tool_configs.py \
  --host "$MCP_HOST" \
  --inputs "$MAIN_CONFIG_PATH" "$EVIDENCE_CONFIG_PATH" \
  --output-dir "$WORK_DIR/remote"

python3 mcp-docker/scripts/make_remote_tool_configs.py \
  --host "$MCP_HOST" \
  --inputs "$CORE_CONFIG_PATH" \
  --output-dir "$WORK_DIR/remote"

python3 mcp-docker/scripts/check_mat_mcp_stack.py \
  --main-config "$remote_main" \
  --evidence-config "$remote_evidence" \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
  --main-config "$remote_main" \
  --evidence-config "$remote_evidence" \
  --tags quick \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/make_verl09_native_mcp_tool_config.py \
  --main-config "$remote_main" \
  --output "$native_config" \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/make_verl09_native_mcp_tool_config.py \
  --main-config "$remote_core" \
  --output "$core_native_config" \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$native_config"

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$core_native_config"

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$native_config" \
  --call-tool validate_formula_verifier \
  --arguments '{"formula":"Fe2O3"}'

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$core_native_config" \
  --call-tool validate_formula_verifier \
  --arguments '{"formula":"Fe2O3"}'

python3 mcp-docker/scripts/make_verl09_router_mcp_tool_config.py \
  --main-config "$remote_main" \
  --output "$router_config" \
  --catalog-output "$router_catalog" \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/prewarm_matgl_stack.py \
  --server-url "$MATGL_SERVER_URL" \
  --model-refs "$MATGL_PREWARM_MODELS" \
  --concurrency "$MATGL_PREWARM_CONCURRENCY" \
  --timeout "$TIMEOUT"

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$router_config" \
  --call-tool mat_mcp_list_tool_domains \
  --arguments '{}'

python3 mcp-docker/scripts/smoke_test_verl09_native_tools.py \
  --tool-config "$router_config" \
  --call-tool mat_mcp_call_tool \
  --arguments '{"tool_name":"validate_formula_verifier","arguments":{"formula":"Fe2O3"}}'

printf 'Native verl 0.9 tool config ready: %s\n' "$native_config"
printf 'Core native verl 0.9 tool config ready: %s\n' "$core_native_config"
printf 'Router verl 0.9 tool config ready: %s\n' "$router_config"
printf 'Use it for training with: export TOOL_CONFIG=%s\n' "$native_config"
printf 'Use compact direct-tool training with: export TOOL_CONFIG=%s\n' "$core_native_config"
printf 'Use router training with: export TOOL_CONFIG=%s\n' "$router_config"
