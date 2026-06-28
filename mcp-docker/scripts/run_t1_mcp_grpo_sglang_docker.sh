#!/usr/bin/env bash
set -euo pipefail
set -x

# Run from /workspace/matbrain inside the verl SGLang Docker container.

export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep MCP traffic off the HTTP proxy. Add external proxy variables outside this
# script only for model/package download steps.
export MCP_HOST="${MCP_HOST:-127.0.0.1}"
export NO_PROXY="$MCP_HOST,localhost,127.0.0.1,0.0.0.0,::1,${NO_PROXY:-}"
export no_proxy="$NO_PROXY"
export HTTP_PROXY="${HTTP_PROXY:-}"
export HTTPS_PROXY="${HTTPS_PROXY:-}"
export ALL_PROXY="${ALL_PROXY:-}"
export http_proxy="${http_proxy:-}"
export https_proxy="${https_proxy:-}"
export all_proxy="${all_proxy:-}"

ROOT_DIR="${ROOT_DIR:-/workspace/matbrain}"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
RUN_ROOT="${RUN_ROOT:-/workspace/run}"
mkdir -p "$RUN_ROOT/triton_cache" "$RUN_ROOT/cuda_cache"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$RUN_ROOT/triton_cache}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$RUN_ROOT/cuda_cache}"

TRAIN_FILE="${TRAIN_FILE:-$ROOT_DIR/dataset/rl/t1_mcp/t1_mcp_train.parquet}"
VAL_FILE="${VAL_FILE:-$ROOT_DIR/dataset/rl/t1_mcp/t1_mcp_eval.parquet}"
DEFAULT_MODEL_PATH="${HOME}/.cache/modelscope/hub/models/Qwen/Qwen3-14B"
HF_QWEN_SNAPSHOT_DIR="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots"
if [[ -d "$HF_QWEN_SNAPSHOT_DIR" ]]; then
  for candidate in "$HF_QWEN_SNAPSHOT_DIR"/*; do
    if [[ -f "$candidate/config.json" ]]; then
      DEFAULT_MODEL_PATH="$candidate"
      break
    fi
  done
fi
MODEL_PATH="${MODEL_PATH:-$DEFAULT_MODEL_PATH}"
DEFAULT_TOOL_CONFIG="$ROOT_DIR/train/rl/tool_configs/verl09_native/mat_mcp_structure_property_172_24_2_23_core_native.yaml"
RUN_TOOL_CONFIG="$RUN_ROOT/matbrain_verl09_mcp/verl09_native/mat_mcp_structure_property_172_24_2_23_native.yaml"
RUN_CORE_TOOL_CONFIG="$RUN_ROOT/matbrain_verl09_mcp/verl09_native/mat_mcp_structure_property_172_24_2_23_core_native.yaml"
RUN_ROUTER_TOOL_CONFIG="$RUN_ROOT/matbrain_verl09_mcp/verl09_router/mat_mcp_structure_property_172_24_2_23_router.yaml"
TMP_TOOL_CONFIG="/tmp/matbrain_verl09_mcp/verl09_native/mat_mcp_structure_property_172_24_2_23_native.yaml"
TMP_CORE_TOOL_CONFIG="/tmp/matbrain_verl09_mcp/verl09_native/mat_mcp_structure_property_172_24_2_23_core_native.yaml"
TMP_ROUTER_TOOL_CONFIG="/tmp/matbrain_verl09_mcp/verl09_router/mat_mcp_structure_property_172_24_2_23_router.yaml"
TOOL_CONFIG_MODE="${TOOL_CONFIG_MODE:-core}"
if [[ "$TOOL_CONFIG_MODE" == "router" ]]; then
  tool_config_candidates=("$RUN_ROUTER_TOOL_CONFIG" "$TMP_ROUTER_TOOL_CONFIG")
elif [[ "$TOOL_CONFIG_MODE" == "core" ]]; then
  tool_config_candidates=("$RUN_CORE_TOOL_CONFIG" "$TMP_CORE_TOOL_CONFIG")
else
  tool_config_candidates=("$RUN_TOOL_CONFIG" "$TMP_TOOL_CONFIG")
fi
for candidate_tool_config in "${tool_config_candidates[@]}"; do
  if [[ -f "$candidate_tool_config" ]]; then
    DEFAULT_TOOL_CONFIG="$candidate_tool_config"
    break
  fi
done
TOOL_CONFIG="${TOOL_CONFIG:-$DEFAULT_TOOL_CONFIG}"
REWARD_FN="${REWARD_FN:-$ROOT_DIR/train/rl/reward_fn/reward.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$RUN_ROOT/checkpoints/t1_mcp_grpo_sglang}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$RUN_ROOT/rollouts/t1_mcp_grpo_sglang}"

if [[ ! -f "$TOOL_CONFIG" ]]; then
  echo "ERROR: TOOL_CONFIG not found: $TOOL_CONFIG" >&2
  if [[ "$TOOL_CONFIG_MODE" == "router" ]]; then
    echo "Router mode is enabled, but no router tool config was found." >&2
  elif [[ "$TOOL_CONFIG_MODE" == "core" ]]; then
    echo "Core mode is enabled, but no compact native tool config was found." >&2
  else
    echo "Native mode is enabled, but no native tool config was found." >&2
  fi
  echo "Run first: bash train/rl/setup_verl09_native_mcp.sh" >&2
  exit 2
fi

export MATBRAIN_REWARD_TOKENIZER="$MODEL_PATH"
export MATBRAIN_TOOL_CONFIG="$TOOL_CONFIG"

echo "Using TOOL_CONFIG_MODE=$TOOL_CONFIG_MODE"
echo "Using TOOL_CONFIG=$TOOL_CONFIG"

python3 "$ROOT_DIR/train/rl/check_t1_rl_prompt_contract.py" \
  "$TRAIN_FILE" \
  "$VAL_FILE"

PROJECT_NAME="${PROJECT_NAME:-matbrain_t1}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_14b_t1_mcp_grpo_sglang}"

MAX_TURNS="${MAX_TURNS:-3}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
MAX_TOOL_RESPONSE_LENGTH="${MAX_TOOL_RESPONSE_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-1}"
N_RESP_PER_PROMPT_VAL="${N_RESP_PER_PROMPT_VAL:-1}"
INFER_TP="${INFER_TP:-2}"
TRAIN_SP="${TRAIN_SP:-4}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.35}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENFORCE_EAGER="${ENFORCE_EAGER:-False}"
ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-True}"
ROLLOUT_QUANTIZATION="${ROLLOUT_QUANTIZATION:-}"
VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
VLLM_CALCULATE_KV_SCALES="${VLLM_CALCULATE_KV_SCALES:-}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async}"
ROLLOUT_NAME="${ROLLOUT_NAME:-sglang}"
USE_LORA="${USE_LORA:-True}"
LORA_RANK="${LORA_RANK:-64}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]}"
LORA_MERGE="${LORA_MERGE:-True}"
ENABLE_SGLANG_LORA_MERGE_COMPAT="${ENABLE_SGLANG_LORA_MERGE_COMPAT:-False}"
ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-safetensors}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-True}"
ROLLOUT_LOG_PROB_DYNAMIC_BSZ="${ROLLOUT_LOG_PROB_DYNAMIC_BSZ:-True}"
REF_LOG_PROB_DYNAMIC_BSZ="${REF_LOG_PROB_DYNAMIC_BSZ:-True}"
ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING="${ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING:-True}"
ACTOR_ENTROPY_CHECKPOINTING="${ACTOR_ENTROPY_CHECKPOINTING:-False}"
PPO_MAX_TOKEN_LEN_PER_GPU="${PPO_MAX_TOKEN_LEN_PER_GPU:-}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:-30}"
TEST_FREQ="${TEST_FREQ:-30}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
LOGGER="${LOGGER:-['console']}"

if [[ "$USE_LORA" == "True" || "$USE_LORA" == "true" || "$USE_LORA" == "1" ]]; then
  ACTOR_LR="${ACTOR_LR:-3e-5}"
  ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
  ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
  REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
  if [[ -z "${FREE_CACHE_ENGINE+x}" ]]; then
    if [[ "$ROLLOUT_NAME" == "vllm" ]]; then
      FREE_CACHE_ENGINE="True"
    else
      FREE_CACHE_ENGINE="False"
    fi
  fi
else
  ACTOR_LR="${ACTOR_LR:-1e-6}"
  ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
  ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
  REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
  FREE_CACHE_ENGINE="${FREE_CACHE_ENGINE:-True}"
fi

actor_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-$actor_max_token_len_per_gpu}"

quant_args=()
if [[ -n "$ROLLOUT_QUANTIZATION" ]]; then
  quant_args=("actor_rollout_ref.rollout.quantization=$ROLLOUT_QUANTIZATION")
fi

rollout_engine_args=()
if [[ "$ROLLOUT_NAME" == "vllm" ]]; then
  rollout_engine_args=(
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=True"
    "+actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes"
  )
  if [[ -n "$VLLM_KV_CACHE_DTYPE" ]]; then
    rollout_engine_args+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.kv_cache_dtype=$VLLM_KV_CACHE_DTYPE")
  fi
  if [[ -n "$VLLM_CALCULATE_KV_SCALES" ]]; then
    rollout_engine_args+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.calculate_kv_scales=$VLLM_CALCULATE_KV_SCALES")
  fi
fi

lora_args=()
if [[ "$USE_LORA" == "True" || "$USE_LORA" == "true" || "$USE_LORA" == "1" ]]; then
  lora_args=(
    "actor_rollout_ref.model.lora_rank=$LORA_RANK"
    "actor_rollout_ref.model.lora_alpha=$LORA_ALPHA"
    "actor_rollout_ref.model.target_modules=$LORA_TARGET_MODULES"
    "actor_rollout_ref.model.lora.merge=$LORA_MERGE"
    "actor_rollout_ref.rollout.load_format=$ROLLOUT_LOAD_FORMAT"
    "actor_rollout_ref.rollout.layered_summon=$ROLLOUT_LAYERED_SUMMON"
  )
  if [[ "$ROLLOUT_NAME" == "sglang" && ( "$LORA_MERGE" == "True" || "$LORA_MERGE" == "true" || "$LORA_MERGE" == "1" ) ]]; then
    if [[ "$ENABLE_SGLANG_LORA_MERGE_COMPAT" == "True" || "$ENABLE_SGLANG_LORA_MERGE_COMPAT" == "true" || "$ENABLE_SGLANG_LORA_MERGE_COMPAT" == "1" ]]; then
      export MATBRAIN_SGLANG_LORA_MERGE_COMPAT=1
      lora_args+=("actor_rollout_ref.model.external_lib=train.rl.sglang_lora_merge_compat")
    fi
  fi
fi

python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="['$TRAIN_FILE']" \
  data.val_files="['$VAL_FILE']" \
  data.return_raw_chat=True \
  data.seed=42 \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.custom_cls.path="$ROOT_DIR/train/rl/t1_mcp_dataset.py" \
  data.custom_cls.name=CustomRLHFDataset \
  custom_reward_function.path="$REWARD_FN" \
  custom_reward_function.name=compute_score \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_shm=True \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  "${lora_args[@]}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$actor_max_token_len_per_gpu" \
  actor_rollout_ref.actor.entropy_from_logits_with_chunking="$ACTOR_ENTROPY_FROM_LOGITS_WITH_CHUNKING" \
  actor_rollout_ref.actor.entropy_checkpointing="$ACTOR_ENTROPY_CHECKPOINTING" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="$TRAIN_SP" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="$ROLLOUT_LOG_PROB_DYNAMIC_BSZ" \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$log_prob_max_token_len_per_gpu" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz="$REF_LOG_PROB_DYNAMIC_BSZ" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$log_prob_max_token_len_per_gpu" \
  actor_rollout_ref.ref.fsdp_config.param_offload="$REF_PARAM_OFFLOAD" \
  actor_rollout_ref.rollout.name="$ROLLOUT_NAME" \
  actor_rollout_ref.rollout.mode="$ROLLOUT_MODE" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$INFER_TP" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$MAX_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.max_tool_response_length="$MAX_TOOL_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.multi_turn.tool_response_truncate_side=middle \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.multi_turn.format=hermes \
  actor_rollout_ref.rollout.gpu_memory_utilization="$GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.free_cache_engine="$FREE_CACHE_ENGINE" \
  actor_rollout_ref.rollout.enforce_eager="$ENFORCE_EAGER" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$ENABLE_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.n="$N_RESP_PER_PROMPT" \
  actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
  actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n="$N_RESP_PER_PROMPT_VAL" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$MAX_NUM_SEQS" \
  "${rollout_engine_args[@]}" \
  "${quant_args[@]}" \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.log_val_generations=10 \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  "$@"
