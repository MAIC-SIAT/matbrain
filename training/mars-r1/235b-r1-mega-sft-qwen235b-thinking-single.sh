#!/usr/bin/env bash
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-235B-A22B}
MAT_R1_DATASET=${MAT_R1_DATASET:-../../data/train-data/mat-r1/mat_r1_train_sample_200.jsonl}
LOG_DIR=${LOG_DIR:-megatron_logs}
mkdir -p "$LOG_DIR"

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
megatron sft \
    --load "$MODEL_PATH" \
    --system 'You are MARS-R1, Created by Material AI Center (MAIC), Shenzhen Institutes of Advanced Technology (SIAT), a professional assistant in materials science.' \
    --dataset "$MAT_R1_DATASET" \
              'swift/self-cognition#50000' \
    --model_name 'MARS-R1' 'MARS-R1' \
    --model_author '材料人工智能研究中心，深圳先进技术研究院' 'Material AI Center (MAIC), Shenzhen Institutes of Advanced Technology (SIAT)' \
    --train_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --split_dataset_ratio 0.001 \
    --moe_permute_fusion true \
    --tensor_model_parallel_size 4 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --save megatron_output/Qwen3-235B-A22B-Thinking-2507-mars-r1 \
    --eval_interval 200 \
    --save_interval 200 \
    --packing true \
    --max_length 10240 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --wandb_exp_name 'qwen3-235b-a22b-thinking-2507-mars-r1' \
    --wandb_project 'qwen3-235b-a22b-thinking-2507-mars-r1' \
    --distributed_backend 'nccl' \
    --no_initialization false \
    > "$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log" 2>&1
