#!/usr/bin/env bash
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-30B-A3B}
MAT_R1_DATASET=${MAT_R1_DATASET:-../../data/train-data/mat-r1/mat_r1_train_sample_200.jsonl}

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=8 \
megatron sft \
    --load "$MODEL_PATH" \
    --system 'You are Mat-R1, Created by Material AI Center (MAIC), Shenzhen Institutes of Advanced Technology (SIAT), a professional assistant in materials science.' \
    --dataset "$MAT_R1_DATASET" \
              'swift/self-cognition#50000' \
    --model_name 'Mat-R1' 'Mat-R1' \
    --model_author '材料人工智能研究中心，深圳先进技术研究院' 'Material AI Center (MAIC), Shenzhen Institutes of Advanced Technology (SIAT)' \
    --split_dataset_ratio 0.001 \
    --tensor_model_parallel_size 2 \
    --expert_tensor_parallel_size 1 \
    --expert_model_parallel_size 8 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 16 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --max_epochs 2 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 2e-5 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-6 \
    --save megatron_output/Mat-R1-Qwen3-30B-A3B-Thinking \
    --eval_interval 200 \
    --save_interval 200 \
    --packing true \
    --max_length 15360 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --sequence_parallel true \
    --attention_backend flash \
    --distributed_backend 'nccl' \
    --report_to swanlab \
    --swanlab_project Mat-R1-Qwen3-30B-A3B
    # --wandb_exp_name 'Mat-R1-Qwen3-30B-A3B' \
    # --wandb_project 'Mat-R1-Qwen3-30B-A3B' \
