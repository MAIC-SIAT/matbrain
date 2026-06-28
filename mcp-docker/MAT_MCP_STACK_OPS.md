# Mat-MCP Stack Operations

This document is the operational entry point for the current Mat-T1/Mat-R1 MCP
Docker stack. It covers the services exposed by:

- `configs/structure_property_tools.yaml`
- `configs/evidence_tools.yaml`

The machine-readable inventory is `mat_mcp_stack.yaml`. The CLI wrapper is
`mat_mcp_stack.py`.

## Current Service Set

The default stack includes 16 services:

| Service | Port | Purpose |
| --- | ---: | --- |
| `matgl-mcp` | 5668 | MatGL/CHGNet property prediction and relaxation |
| `crystallm-mcp` | 5669 | CrystaLLM structure generation |
| `pymatgen-mcp` | 5672 | pymatgen structure analysis, XRD, slabs, VASP I/O, magnetism |
| `smact-mcp` | 5673 | oxidation state and charge-neutrality checks |
| `matminer-mcp` | 5674 | composition and structure descriptors |
| `pyxtal-mcp` | 5675 | symmetry-constrained crystal generation |
| `structure-verifier-mcp` | 5676 | reward-ready formula/structure/route validation |
| `materials-db-router-mcp` | 5677 | MP/OPTIMADE lookup and novelty evidence |
| `synthesis-kb-mcp` | 5678 | offline synthesis route and precursor KB |
| `uip-relax-mcp` | 5679 | MACE-MP single point, relaxation, and UIP rewards |
| `literature-evidence-mcp` | 5680 | OpenAlex/Crossref/arXiv/web/PubChem evidence search |
| `reaction-thermo-mcp` | 5682 | reaction balancing and MP thermodynamic evidence |
| `characterization-mcp` | 5683 | XRD simulation, matching, and phase-purity scoring |
| `precursor-chemistry-mcp` | 5684 | RDKit/PubChem precursor and safety checks |
| `tool-registry-mcp` | 5685 | tool capability registry and task-conditioned selection |
| `defect-interface-mcp` | 5686 | defects, interfaces, and diffusion paths |

`mat-query-mcp` and `mattergen-mcp` are recorded in `mat_mcp_stack.yaml` but
excluded by default. They are not exposed by the current RL tool configs.

## One-Time Static Check

Run this after editing `.env`, ports, service directories, or tool YAMLs:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 mcp-docker/mat_mcp_stack.py check-config
```

One-command bringup for the production host:

```bash
bash mcp-docker/scripts/start_production128_stack.sh
```

This checks:

- service directories, `Dockerfile`, `.env`, and `docker-compose.yml`
- duplicate ports
- `.env` `HOST_PORT` / `SERVER_PORT` against the inventory
- `/pymatgen` path prefix
- training YAML server URLs against enabled stack services
- obvious optional-key gaps such as empty MP/Bing/Tavily keys

## Start / Stop

The stack launcher uses sizing profiles:

- `--profile single`: router + backend-1 for every service.
- `--profile smoke`: moderate backend fanout for validation.
- `--profile production32`: high concurrency.
- `--profile production128`: aggressive fanout for 128-core / 4-GPU hosts.

Build and start the default stack in single-backend mode:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build up --profile single
```

Build and start the production stack:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build up --profile production128
```

Start without rebuilding:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo up --profile production128
```

Stop:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo down
```

Restart:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo restart --profile production128
```

Show compose status:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo ps
```

Preview the exact commands without changing state:

```bash
python3 mcp-docker/mat_mcp_stack.py --dry-run up --profile production128
```

If your user is already in the Docker group, omit `--sudo`.

## MACE Model Warmup

`uip-relax-mcp` mounts the host cache at:

```text
./mcp-docker/uip-relax-mcp/model-cache
```

Warm or download the configured MACE-MP models:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo warmup-mace
```

The current cache has the `small` and `medium` MACE-MP model files. The service
uses `MACE_MP_MODEL=medium` and reports `device=cuda` in `/health`.

## MatGL Prewarm

MatGL uses an offline Hugging Face cache under:

```text
./mcp-docker/matgl-mcp/hf-cache/hub
```

Prewarm the common MatGL property models across the backend pool:

```bash
python3 mcp-docker/scripts/prewarm_matgl_stack.py \
  --server-url http://localhost:5668/sse \
  --model-refs EFORM,BAND_GAP_MFI \
  --concurrency 64
```

## Validation

Default deterministic validation:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags quick --timeout 90
```

This runs:

1. `mcp-docker/scripts/check_mat_mcp_stack.py`
2. `mcp-docker/scripts/check_mat_mcp_tool_calls.py --coverage-only`
3. `mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags quick`

A healthy stack currently shows:

- 16 healthy services
- no missing `list_tools` entries
- 105 configured tool entries and 89 unique tools with full fixture coverage
- 64 quick tool calls with 0 failures

GPU/ML checks:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags gpu,matgl --timeout 240
python3 mcp-docker/mat_mcp_stack.py verify --tags gpu,mace --timeout 240
```

Network and MP checks:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags network,mp --timeout 120
```

Run several groups in one command:

```bash
python3 mcp-docker/mat_mcp_stack.py verify \
  --tags quick \
  --tags gpu,matgl \
  --tags gpu,mace \
  --timeout 240
```

One-shot build, start, MACE warmup, and quick validation:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build all --profile production128 --tags quick --timeout 90
```

## Optional Failures

Some tests are expected to be optional because they depend on live network,
rate-limited APIs, or private keys.

Current known optional gaps:

- `materials-db-router-mcp/.env`: `MP_API_KEY` is empty.
- `reaction-thermo-mcp/.env`: `MP_API_KEY` is empty.
- `literature-evidence-mcp/.env`: `BING_API_KEY` and `TAVILY_API_KEY` are empty.
- arXiv and Semantic Scholar are intentionally not exposed in the default evidence
  tool config because they frequently return HTTP 429 in this environment. They
  remain callable inside `literature-evidence-mcp` only for ad-hoc debugging.
- Public OPTIMADE providers can return provider-specific server errors.

Use strict mode only when the relevant keys and network conditions are ready:

```bash
python3 mcp-docker/mat_mcp_stack.py verify \
  --tags network,mp \
  --timeout 120 \
  --strict-optional
```

## Direct Test Scripts

The wrapper calls these scripts internally, but they remain useful for focused
debugging:

```bash
python3 mcp-docker/scripts/check_mat_mcp_stack.py --timeout 60
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --coverage-only
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags quick --timeout 90
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags gpu,matgl --timeout 240
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags gpu,mace --timeout 240
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --list-tags
```

Persist per-tool outputs when debugging:

```bash
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py \
  --tags quick \
  --timeout 90 \
  --results-jsonl /tmp/mat_mcp_quick_tool_calls.jsonl
```

## Interpreting Results

`check_mat_mcp_stack.py` verifies service-level readiness:

- `/health` endpoint works
- MCP `list_tools` includes all configured tools
- several deterministic calls work

`check_mat_mcp_tool_calls.py` verifies function-level invocation:

- every configured tool has a fixture
- selected fixtures run through MCP SSE
- error-like text, tracebacks, wrapper errors, and JSON `{ok:false}` are
  detected
- optional tools do not fail the run unless `--strict-optional` is used

The quick fixtures validate call chains, input schemas, output shape, and gross
runtime health. They are not numerical-accuracy benchmarks. Accuracy/sanity
fixtures should use physically meaningful known structures and expected ranges.
