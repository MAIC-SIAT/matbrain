# Mat-MCP Tool Matrix

This directory contains the MCP services used by Mat-T1/Mat-R1 agentic RL.
The design goal is to expose research actions, not only Python packages.

## Actor / Main RL Tools

| Service | Port | Role |
| --- | ---: | --- |
| `crystallm-mcp` | 5669 | CIF generation |
| `matgl-mcp` | 5668 | MatGL/CHGNet property prediction and relaxation |
| `pymatgen-mcp` | 5672 | structure analysis, XRD, slabs, VASP I/O, magnetism |
| `smact-mcp` | 5673 | charge neutrality and chemical plausibility |
| `matminer-mcp` | 5674 | composition/structure descriptors |
| `pyxtal-mcp` | 5675 | symmetry-constrained crystal generation |
| `structure-verifier-mcp` | 5676 | reward-ready hard validation |
| `synthesis-kb-mcp` | 5678 | synthesis precedent, precursor, and route scoring |
| `uip-relax-mcp` | 5679 | MACE-MP/UIP relaxation and dense discovery reward |
| `reaction-thermo-mcp` | 5682 | reaction balancing, MP reaction energy, phase/open-system stability |
| `characterization-mcp` | 5683 | XRD simulation, pattern comparison, phase-purity scoring |
| `precursor-chemistry-mcp` | 5684 | PubChem/RDKit precursor, solvent, ligand, and safety checks |
| `defect-interface-mcp` | 5686 | vacancies, dopants, interstitials, interfaces, and diffusion paths |

Main tool config:

```text
../train/rl/tool_configs/structure_property_tools.yaml
```

## Evidence / Retrieval Tools

| Service | Port | Role |
| --- | ---: | --- |
| `materials-db-router-mcp` | 5677 | MP/OPTIMADE search, novelty, cross-database evidence |
| `literature-evidence-mcp` | 5680 | Bing/Tavily/arXiv/OpenAlex/Crossref/Semantic Scholar/PubChem evidence |
| `tool-registry-mcp` | 5685 | capability registry and task-conditioned tool selection |

Evidence tool config:

```text
../train/rl/tool_configs/evidence_tools.yaml
```

## Attribution Rule

For reviewer-facing experiments, separate these modes:

1. No-tool model.
2. Scripted pipeline using the same tools.
3. Mat-T1 with low-level actor tools.
4. Mat-T1 with verifier reward.
5. Evidence-only retrieval used after generation, not as the main property shortcut.

This keeps LLM planning, tool execution, and external retrieval attribution separable.

## Stack Checks

The unified entry point is:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 mcp-docker/mat_mcp_stack.py check-config
python3 mcp-docker/mat_mcp_stack.py --sudo --build up
python3 mcp-docker/mat_mcp_stack.py --sudo warmup-mace
python3 mcp-docker/mat_mcp_stack.py verify --tags quick --timeout 90
```

For a one-shot local bring-up and deterministic validation:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build all --tags quick --timeout 90
```

Use repeated `--tags` for broader checks:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags quick --tags gpu,matgl --tags gpu,mace --timeout 240
```

`mat-query-mcp` and `mattergen-mcp` are recorded in
`mcp-docker/mat_mcp_stack.yaml`, but are excluded by default because the current
RL tool configs intentionally do not expose them. Add `--include-extra` when
you explicitly want Docker operations to include those extra services.

After starting services, run:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 train/rl/check_mat_mcp_stack.py --timeout 60
```

For a lighter check that only verifies exposed tools and several no-network
calls:

```bash
python3 train/rl/smoke_test_mcp_tools.py --timeout 60
```

GPU-heavy tools (`matgl-mcp`, `uip-relax-mcp`, `crystallm-mcp`) and open-web/API
tools should be tested separately from deterministic no-network tools.
