<div align="center">

# MatBrain

**A lightweight dual-model collaborative agent for autonomous crystal materials research**

[![Paper](https://img.shields.io/badge/Paper-Under%20Review-red)]()
[![Code](https://img.shields.io/badge/Code-Public%20Release-green)]()
[![MCP](https://img.shields.io/badge/Mat--MCP-Dockerized-blue)]()
[![Data](https://img.shields.io/badge/Data-Samples%20%26%20Audit-orange)]()
[![Model](https://img.shields.io/badge/Models-Weights%20Not%20Included-lightgrey)]()

</div>

MatBrain is a dual-model collaborative agent for crystal-materials research. It
separates two functions that are difficult to optimize with a single model:

- **Mat-T1**: an executive tool-planning module that selects and calls
  materials-science tools.
- **Mat-R1**: an analytical reasoning module that reads the tool observations
  and produces the final scientific answer.

The public repository contains the lightweight agent implementation, MCP tool
server code, selected training scripts, public sample data, and data-leakage
audit artifacts. Model weights and full internal training datasets are not
included in this release.

## Repository Layout

```text
.
├── agent/
│   ├── app.py                     # Gradio UI for the MatBrain agent
│   ├── scripts/                   # CLI entry point
│   ├── requirements.txt           # agent runtime dependencies
│   ├── .env.example               # empty runtime configuration template
│   └── matbrain/                  # LangGraph agent implementation
│       ├── graph.py               # executor -> reasoner -> router graph
│       ├── llm.py                 # provider-compatible model wrapper
│       ├── mcp_client.py          # SSE MCP client and tool registry
│       ├── prompts.py             # Mat-T1 / Mat-R1 system prompts
│       └── nodes/                 # executor, reasoner, finalizer nodes
├── mcp-docker/                    # MCP tool servers and stack utilities
├── training/                      # Mat-T1 RL and Mat-R1 SFT scripts/notes
├── figure-data/                   # selected figure-level source materials
└── data/
    ├── train-data/                # public training-data samples
    ├── eval-data/                 # leakage-controlled evaluation samples
    └── data-audit/                # leakage audit flags and split IDs
```

## Agent Topology

```text
START -> executor (Mat-T1) -> reasoner (Mat-R1) -> route
             ^                                      |
             | pending_instruction                  | terminated
             +--------------------------------------+
                                                    v
                                              finalizer -> END
```

- The executor emits either `<think>...<tool_call>{...}</tool_call>` or
  `<think>...<answer>...</answer>`.
- Tool calls are parsed, validated against MCP tool schemas, and dispatched
  through SSE.
- The reasoner reads the full execution history and emits either
  `<next_instruction>...</next_instruction>` or `<answer>...</answer>`.
- `MAX_ITERATIONS` provides a hard ceiling for the collaborative loop.

## Quick Start

Create an environment and install the public agent dependencies:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r agent/requirements.txt
```

Create a local runtime configuration:

```bash
cp agent/.env.example agent/.env
```

Then fill in the local model endpoint settings and MCP server endpoints in
`agent/.env`. The template intentionally contains no credential values or model
identifiers; the MCP endpoint entries are local-host defaults you should adjust
for your own deployment.

Run one query from the CLI:

```bash
.venv/bin/python agent/scripts/run_query.py "查询 CsPbBr3 的晶体结构并预测其形成能"
```

Launch the Web UI:

```bash
.venv/bin/python agent/app.py
```

The UI exposes a chat panel, live executor/reasoner trace, tool inventory, and
stop/reset controls.

## MCP Tool Stack

The MCP services are under `mcp-docker/`. The public copy includes tool servers
for structure query, structure generation, property prediction, crystallographic
analysis, chemistry checks, descriptor calculation, synthesis-route utilities,
and validation.

Useful entry points:

```bash
cd mcp-docker
python3 mat_mcp_stack.py --help
```

See `mcp-docker/README.md`, `mcp-docker/MAT_MCP_MANIFEST.md`, and
`mcp-docker/MAT_MCP_STACK_OPS.md` for service-level details.

## Public Data

The public data directory contains only sample and audit artifacts prepared for
repository release.

| Path | Description |
| --- | --- |
| `data/train-data/mat-r1/mat_r1_train_sample_200.jsonl` | 200 Mat-R1 training examples with `mp_id`, `question_type`, and `messages` fields. |
| `data/eval-data/matbrain_eval_sample_200_leakage_controlled.jsonl` | 200 leakage-controlled evaluation examples. |
| `data/data-audit/scripts/` | Leakage-audit scripts for identity, exact-CIF, composition, protostructure-label, and StructureMatcher-confirmed structure-similarity checks. |
| `data/data-audit/leakage_row_flags.jsonl` | Row-level leakage audit flags for 2,000 held-out examples. |
| `data/data-audit/leakage_metric_summary_table.csv` | Summary table for identity, CIF-hash, composition, protostructure-label, and structure-similarity audits. |
| `data/data-audit/clean-tests/mp_cif_design_property_test_no_identity_structure_full_protostructure_overlap.jsonl` | Primary 1,662-example leakage-controlled held-out set. |
| `data/data-audit/clean-tests/mp_cif_design_property_test_no_any_audit_overlap.jsonl` | Conservative 1,308-example clean set after also excluding raw-formula overlaps. |
| `data/data-audit/audit_run_log.md` | Reproducibility notes and command sequence for the audit artifacts. |
| `data/data-audit/train_mp_ids.json` | Time-split training MP IDs. |
| `data/data-audit/val_mp_ids.json` | Time-split validation MP IDs. |
| `data/data-audit/test_mp_ids.json` | Time-split test MP IDs. |
| `figure-data/figure5/matbrain_session_20260525_105145.md` | MatBrain conversation transcript used as supporting source material for the Figure 5 mixed-halide perovskite case. |

The leakage audit reports no MP-ID overlap and no exact CIF-hash overlap between
training and held-out evaluation entries. Prototype-level and
StructureMatcher-confirmed near-duplicate flags are provided so users can
reproduce leakage-controlled subsets.

## Training Scripts

Training scripts are provided for reproducibility and inspection. They are not
turn-key training recipes unless the required model checkpoints, datasets, and
distributed training environment are available locally.

```text
training/
├── mars-t1/
│   ├── MARS-T1_TRAINING_README.md
│   ├── run_qwen3-14b_dapo.sh
│   ├── run_qwen3-14b_dapo_sync.sh
│   ├── mars.py
│   ├── mars_mcp_tools/
│   └── mars_reward_fn/
└── mars-r1/
    ├── MARS-R1_TRAINING_README.md
    ├── 30b-r1-mega-sft-single.sh
    ├── 235b-r1-mega-sft-qwen235b-thinking-single.sh
    └── data-sample/
```

The Mat-T1 RL scripts use explicit hard budgets:

- `max_prompt_length=16384`
- `max_response_length=3072`
- `max_turns=4`

The Mat-T1 reward code is in
`training/mars-t1/mars_reward_fn/mars_rewards.py`. The Mat-R1 SFT scripts are in
`training/mars-r1/`.

## Model Weights

This repository does not include trained model weights. The agent code can be
connected to compatible local or remote chat-completion endpoints through the
environment variables in `agent/.env.example`.

## Notes

- The public release intentionally excludes private run logs, full checkpoints,
  full internal datasets, `.env` files, and large model artifacts.
- Model badges use a conservative "weights not included" status until public
  model cards are released.
- Some third-party tool code under `mcp-docker/` retains original architecture
  class names and upstream references.
- Materials Project-derived examples and labels remain subject to the Materials
  Project terms and citation requirements.
