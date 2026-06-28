# Mat-R1 Training Notes

This directory contains public supervised fine-tuning entry points for Mat-R1,
the analytical reasoning module in MatBrain.

The scripts are provided for reproducibility and inspection. They require a
Megatron-Swift-compatible environment, the selected base model checkpoint, and
the full training corpus. The public repository includes only a 200-example
sample dataset.

## Main Files

| Path | Purpose |
| --- | --- |
| `30b-r1-mega-sft-single.sh` | Full-parameter SFT script for the 30B analytical model. |
| `235b-r1-mega-sft-qwen235b-thinking-single.sh` | Optional large-backbone LoRA SFT reference script. |
| `data-sample/mars_r1_train_sample_200.jsonl` | Public 200-example Mat-R1 training sample. |

## Public Sample Data

The public sample is also mirrored at:

```text
data/train-data/mat-r1/mat_r1_train_sample_200.jsonl
```

Each row retains only the fields needed for inspection:

- `mp_id`
- `question_type`
- `messages`

The full Mat-252K-SFT corpus and trained Mat-R1 checkpoint are not included in
this repository.

## 30B SFT Configuration

The 30B script uses:

| Parameter | Value |
| --- | ---: |
| Global batch size | 16 |
| Maximum epochs | 2 |
| Peak learning rate | 2e-5 |
| Minimum learning rate | 1e-6 |
| Warmup fraction | 0.05 |
| Maximum sequence length | 15,360 tokens |
| Tensor parallel size | 2 |
| Expert model parallel size | 8 |
| Attention backend | Flash attention |

Local model paths, dataset paths, output directories, and logging backends
should be configured by the user before launching a training run.
