# MatBrain MCP Services

This directory contains the Model Context Protocol (MCP) service code used by
the MatBrain / Mat-T1 / Mat-R1 agent system. Each subdirectory is a
self-contained Docker Compose project, and the current multi-service stack is
managed by `mat_mcp_stack.py`.

For day-to-day operations, start with:

- `MAT_MCP_STACK_OPS.md`: operational runbook
- `MAT_MCP_MANIFEST.md`: tool/service matrix and benchmark attribution notes
- `mat_mcp_stack.yaml`: machine-readable service inventory
- `mat_mcp_stack.py`: one-entry stack orchestration and validation CLI

## Current Default Stack

The default stack is the service set exposed by:

- `configs/structure_property_tools.yaml`
- `configs/evidence_tools.yaml`

| Service | Port | Purpose | Requires GPU |
| --- | --- | --- | --- |
| [`matgl-mcp/`](matgl-mcp/) | 5668 | M3GNet / MEGNet / TensorNet / CHGNet property prediction (relaxation, formation energy, internal energy, band gap, bulk modulus, MD) | Yes |
| [`crystallm-mcp/`](crystallm-mcp/) | 5669 | CrystaLLM conditional structure generation | Yes |
| [`pymatgen-mcp/`](pymatgen-mcp/) | 5672 | pymatgen structure analysis, XRD, slabs, VASP I/O, magnetism, and thermodynamic stability workflows | No |
| [`smact-mcp/`](smact-mcp/) | 5673 | oxidation-state, charge-neutrality, and Pauling plausibility checks | No |
| [`matminer-mcp/`](matminer-mcp/) | 5674 | composition and structure descriptors | No |
| [`pyxtal-mcp/`](pyxtal-mcp/) | 5675 | symmetry-constrained crystal generation | No |
| [`structure-verifier-mcp/`](structure-verifier-mcp/) | 5676 | reward-ready formula, structure, and synthesis-route validation | No |
| [`materials-db-router-mcp/`](materials-db-router-mcp/) | 5677 | MP/OPTIMADE lookup and novelty evidence | No |
| [`synthesis-kb-mcp/`](synthesis-kb-mcp/) | 5678 | offline synthesis precedent, precursor, and route scoring | No |
| [`uip-relax-mcp/`](uip-relax-mcp/) | 5679 | MACE-MP single point, relaxation, stability scoring, and discovery reward | Yes |
| [`literature-evidence-mcp/`](literature-evidence-mcp/) | 5680 | OpenAlex/Crossref/PubChem evidence search, with optional Bing/Tavily when API keys are configured | No |
| [`reaction-thermo-mcp/`](reaction-thermo-mcp/) | 5682 | reaction balancing and MP-based reaction/phase/open-system thermodynamics | No |
| [`characterization-mcp/`](characterization-mcp/) | 5683 | XRD simulation, pattern comparison, matching, and phase-purity scoring | No |
| [`precursor-chemistry-mcp/`](precursor-chemistry-mcp/) | 5684 | RDKit/PubChem precursor, solvent, ligand, and safety checks | No |
| [`tool-registry-mcp/`](tool-registry-mcp/) | 5685 | service/tool registry and task-conditioned tool selection | No |
| [`defect-interface-mcp/`](defect-interface-mcp/) | 5686 | vacancies, dopants, interstitials, interfaces, and diffusion paths | No |

Extra services are kept in the repository but are excluded from the default RL
tool configs:

| Service | Port | Purpose | Requires GPU |
| --- | --- | --- | --- |
| [`mat-query-mcp/`](mat-query-mcp/) | 5667 | direct Materials Project / OQMD lookup | No |
| [`mattergen-mcp/`](mattergen-mcp/) | 5670 | MatterGen diffusion-based structure generation baseline | Yes |

Use `--include-extra` with `mat_mcp_stack.py` only when you explicitly want
Docker operations to include those extra services.

---

## Unified Quick Start

`mat_mcp_stack.py` now supports stack sizing profiles:

- `--profile single`: one backend per service.
- `--profile smoke`: moderate fanout for validation.
- `--profile production32`: high concurrency.
- `--profile production128`: aggressive fanout for 128-core / 4-GPU hosts.

The public MCP URLs do not change between profiles. Only the internal backend pool changes.

Run a static configuration check:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 mcp-docker/mat_mcp_stack.py check-config
```

One-command bringup for the current 128-core / 4-GPU host:

```bash
bash mcp-docker/scripts/start_production128_stack.sh
```

Build and start the default stack in single-backend mode:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build up --profile single
```

Build and start the production stack:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build up --profile production128
```

Warm MACE-MP models for `uip-relax-mcp`:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo warmup-mace
```

Prewarm common MatGL property models:

```bash
python3 mcp-docker/scripts/prewarm_matgl_stack.py \
  --server-url http://localhost:5668/sse \
  --model-refs EFORM,BAND_GAP_MFI \
  --concurrency 64
```

Validate health, tool exposure, fixture coverage, and deterministic quick calls:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags quick --timeout 90
```

One-shot build, start, MACE warmup, and quick validation:

```bash
python3 mcp-docker/mat_mcp_stack.py --sudo --build all --profile production128 --tags quick --timeout 90
```

Preview the exact `docker compose` commands without changing state:

```bash
python3 mcp-docker/mat_mcp_stack.py --dry-run up --profile production128
```

GPU checks:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags gpu,matgl --timeout 240
python3 mcp-docker/mat_mcp_stack.py verify --tags gpu,mace --timeout 240
```

Network/API checks:

```bash
python3 mcp-docker/mat_mcp_stack.py verify --tags network,mp --timeout 120
```

Optional network/API tools are allowed to report optional failures unless
`--strict-optional` is used. arXiv and Semantic Scholar are not exposed in the
default evidence tool config because they are too rate-limit prone in this
environment. See `MAT_MCP_STACK_OPS.md` for details.

## External resources required at deploy time

The archive intentionally does NOT bundle public, redistributable data and
weights. Reviewers reproducing the system should fetch the following from the
sources below; all URLs were verified reachable at the time this archive was
prepared.

### Pre-trained model weights

| Where it goes | Where to get it | Size |
| --- | --- | --- |
| `crystallm-mcp/models/ckpt.pt` | [Zenodo 10642388 — `crystallm_v1_small.tar.gz`](https://zenodo.org/records/10642388/files/crystallm_v1_small.tar.gz) (the entrypoint script `crystallm-mcp/scripts/model_manager.sh` downloads it automatically on first container start) | ~285 MB tarball; ~297 MB extracted |
| `uip-relax-mcp/model-cache/mace/` | warmed by `uip-relax-mcp/download_mace_models.py` or `mat_mcp_stack.py --sudo warmup-mace` | `small` and `medium` MACE-MP models are enough for current fixtures |
| `mattergen-mcp/mattergen_ckpt/*/` | [HuggingFace `microsoft/mattergen`](https://huggingface.co/microsoft/mattergen) — 8 model variants (`mattergen_base`, `chemical_system`, `chemical_system_energy_above_hull`, `dft_band_gap`, `dft_mag_density`, `dft_mag_density_hhi_score`, `ml_bulk_modulus`, `space_group`). Also distributed via Git LFS in the [upstream repo](https://github.com/microsoft/mattergen). | ~3.4 GB total |
| `matgl-mcp/hf-cache/hub/` | predownload via `matgl-mcp/scripts/download_hf_models.py`; runtime loads local HF snapshots with `HF_HUB_OFFLINE=1` | size depends on selected models |
| CHGNet weights for `predict_internal_energy_MatGL` | Bundled in the official [`chgnet`](https://pypi.org/project/chgnet/) pip package (installed by `matgl-mcp/Dockerfile`); no separate download required | — |

### Vendored source

| Location | Upstream |
| --- | --- |
| `crystallm-mcp/tools/CrystaLLM/` | [github.com/lantunes/CrystaLLM](https://github.com/lantunes/CrystaLLM) (release archive included; the `resources/benchmarks/` directory is excluded — only needed for paper-replication of CrystaLLM's own evaluation, not for inference) |
| `mattergen-mcp/mattergen/` | [github.com/microsoft/mattergen](https://github.com/microsoft/mattergen) |

### Runtime data sources

| Resource | Source | Notes |
| --- | --- | --- |
| Materials Project API key | [docs.materialsproject.org](https://docs.materialsproject.org/) → "Get an API key" | Set as `MP_API_KEY` in services that need strict MP validation: `pymatgen-mcp/.env`, `materials-db-router-mcp/.env`, `reaction-thermo-mcp/.env`, and optionally `mat-query-mcp/.env` |
| Bing/Tavily API keys | provider dashboards | Set `BING_API_KEY` and `TAVILY_API_KEY` in `literature-evidence-mcp/.env` only when strict web evidence validation is needed |
| MP CIF and property archive (used by `mat-query-mcp` for offline lookup) | Reviewers can either (a) point `HOST_MP_CIF_ROOT` / `HOST_MP_PROPS_ROOT` in `mat-query-mcp/.env` to a local snapshot of the MP corpus, or (b) modify `mat-query-mcp/tools/mat_query_tools.py` to fall back to the live `MPRester` API for any CIF not present locally | The bundled snapshot of ~153k CIFs + JSONs (~1.5 GB) is not included; it can be regenerated from the MP API |
| OQMD | [oqmd.org](https://oqmd.org/) | `query_material_from_OQMD` scrapes the public web pages; no key needed |

---

## Per-Service Quick Start

Each MCP remains a stand-alone `docker compose` project:

```bash
cd mcp-docker/matgl-mcp   # or any service subdir
# edit .env to fill in your MP_API_KEY and host paths
docker compose up -d --build
docker compose logs -f
```

GPU services (`matgl-mcp`, `crystallm-mcp`, `uip-relax-mcp`, and optional
`mattergen-mcp`) require the host to have a working NVIDIA driver and
`nvidia-container-toolkit` registered with the Docker daemon
(`nvidia-ctk runtime configure --runtime=docker`). Tested on NVIDIA A100-SXM-64GB
with driver 580.159.03.

`pymatgen-mcp` calls `matgl-mcp` at the container DNS name `matgl-mcp:5668`,
so it must be brought up onto the same Docker network as `matgl-mcp` (see
`pymatgen-mcp/docker-compose.yml` — it joins `matgl-mcp-network` as an
`external` network).

---

## Verifying a deployment

Preferred verification uses the unified wrapper:

```bash
cd "$(git rev-parse --show-toplevel)"
python3 mcp-docker/mat_mcp_stack.py verify --tags quick --timeout 90
```

For focused debugging, run the lower-level scripts directly:

```bash
python3 mcp-docker/scripts/check_mat_mcp_stack.py --timeout 60
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --coverage-only
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags quick --timeout 90
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags gpu,matgl --timeout 240
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --tags gpu,mace --timeout 240
python3 mcp-docker/scripts/check_mat_mcp_tool_calls.py --list-tags
```

---

## Notes on energy basis for `analyze_thermodynamic_stability_pymatgen`

The `predict_internal_energy_MatGL` tool consumed by
`analyze_thermodynamic_stability_pymatgen` uses the official `chgnet` package
(`chgnet.model.CHGNet.load()` + `predict_structure`), whose per-atom energy
output is directly comparable to the Materials Project PBE total energies that
the phase diagram is constructed from. On three reference compounds —
`CsPbBr3` Pm-3m, `Li2ZrCl6` P-3m1, and `NaCl` Fm-3m — the resulting
`energy_above_hull` lands within CHGNet's published MAE (≈30–50 meV/atom) of
MP's reported values. When using hull-above for screening, allow for this
residual model error: a threshold of ~0.05 eV/atom on top of the nominal
0.025 eV/atom stability cutoff is a conservative choice that absorbs CHGNet's
intrinsic uncertainty.

---

## Repository layout & `.gitignore` policy

The repository **does not bundle** the large runtime artifacts referenced in
the "External resources" section above. Specifically, the following paths are
`.gitignore`d and will appear empty on a fresh clone:

| Path | Size | How to populate |
| --- | --- | --- |
| `mattergen-mcp/mattergen_ckpt/` | ~3.4 GB | Fetch from HuggingFace `microsoft/mattergen` (see External Resources table) |
| `crystallm-mcp/models/` (`ckpt.pt`) | ~297 MB | Auto-fetched by `crystallm-mcp/scripts/model_manager.sh` on first container start |
| `logs/`, `*/logs/`, `temp/`, `*/temp/` | varies | Created at runtime |
| `__pycache__/`, `*.pyc` | varies | Python build artifacts |
| `crystallm-mcp/tools/CrystaLLM/resources/benchmarks/` | ~hundreds of MB | Only needed for CrystaLLM's own paper-replication evaluation, not for inference |

`matgl-mcp/models/` (~8 MB total: M3GNet, MEGNet, TensorNet pretrained weights)
**is** tracked — these are small enough to live in git and are loaded directly
at runtime without an online download step.

`mat-query-mcp` reads MP CIFs/properties from a host-mounted directory
configured via `HOST_MP_CIF_ROOT` / `HOST_MP_PROPS_ROOT` in
`mat-query-mcp/.env`; that local corpus is not part of this repository.

### Pre-push verification (maintainer)

Before any `git push` to a public remote, confirm none of the runtime-only
artifacts have slipped into the index:

```bash
cd mcp-docker/
git ls-files | grep -E '(mattergen_ckpt/|crystallm-mcp/models/.+\.pt$|/logs/|MPDatasets/|/Props/)'
# (should print nothing)
```

If anything shows up, `git rm --cached <path>` it and update `.gitignore`.
