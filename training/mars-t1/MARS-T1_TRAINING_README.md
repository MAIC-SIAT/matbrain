# Mat-T1 Training Notes

This directory contains the public training entry points and reward code for
Mat-T1, the tool-planning module in MatBrain.

The files are provided for reproducibility and inspection. They are not a
turn-key training package unless the required base model, full training data,
MCP services, and distributed training environment are available locally.

## Main Files

| Path | Purpose |
| --- | --- |
| `run_qwen3-14b_dapo.sh` | Main verl/DAPO-style RL launch script. |
| `run_qwen3-14b_dapo_sync.sh` | Synchronous variant of the RL launch script. |
| `mars.py` | Custom dataset class used by verl. |
| `mars_reward_fn/mars_rewards.py` | Reward components for format, syntax, reasoning length, and interaction turns. |
| `mars_mcp_tools/mars_tool_config.yaml` | Public MCP tool configuration template for Mat-T1 rollouts. |
| `mars_mcp_tools/mars_tools.py` | SSE-based MCP tool wrapper used by the verl rollout loop. |
| `test_code/` | Lightweight tool-wrapper checks. |

## Training Setup

The public launch script uses environment variables so local paths can be
provided without modifying the script:

```bash
export MODEL_PATH=/path/to/base-model
export MARS_TRAIN_DATA=/path/to/train.parquet
export MARS_TEST_DATA=/path/to/test.parquet
export MARS_TOOL_CONFIG=/path/to/mars_tool_config.yaml

bash training/mars-t1/run_qwen3-14b_dapo.sh
```

The default public script references a Qwen3-14B-compatible base model name and
expects verl-compatible parquet data. Model weights and the full internal
training dataset are not included in this repository.

## Key Hyperparameters

| Parameter | Value |
| --- | ---: |
| Algorithm | GRPO/DAPO-style policy optimization |
| KL coefficient | 0.0 |
| Maximum prompt length | 16,384 tokens |
| Maximum response length | 3,072 tokens |
| Maximum user turns | 4 |
| Maximum assistant turns | 4 |
| Actor learning rate | 1e-6 |
| Train batch size | 8 |
| Rollout engine | vLLM |
| Tensor parallelism for rollout | 4 |
| Sequence parallelism for training | 4 |

The hard limits on prompt length, response length, and multi-turn interactions
are part of the public training configuration and prevent unbounded growth of
reasoning traces or tool-use sequences during RL rollouts.

## Reward Code

The reward implementation is in `mars_reward_fn/mars_rewards.py`. The public
configuration includes bounded process-level shaping for executable tool
interaction:

- tool-call syntax validity;
- required response format;
- tanh-saturated reasoning-length reward;
- capped multi-turn interaction reward.

The reward code is intended to be read together with the launch scripts because
the hard rollout budgets and bounded reward design jointly define the Mat-T1 RL
training protocol.

## Public Data

Only sample and audit data are included in this repository. The full Mat-T1 RL
training data and trained checkpoints are excluded from the public release.
