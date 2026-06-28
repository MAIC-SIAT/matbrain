#!/usr/bin/env python3
"""Invoke configured Mat-MCP tools with deterministic fixtures.

`check_mat_mcp_stack.py` verifies service health and tool exposure. This script
goes one level deeper: every tool listed in the main/evidence YAML configs must
have a fixture here, and selected fixtures are executed through MCP SSE.

Default mode intentionally runs only `quick` cases: offline, low-cost calls that
should be stable before training. Use `--tags all` or add tags such as `gpu`,
`network`, `mp`, `slow`, and `generation` for broader validation.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

try:
    import yaml
except ImportError as exc:  # pragma: no cover - user-facing environment check
    raise SystemExit(
        "Missing dependency for MCP tool-call check. Run this in the same "
        f"environment that has the MCP client installed. Original error: {exc}"
    ) from exc


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataset.kd.lib.mcp_runtime import call_tool as runtime_call_tool
from dataset.kd.lib.mcp_runtime import close_all_session_pools

MAIN_CONFIG = ROOT / "mcp-docker" / "configs" / "structure_property_tools.yaml"
EVIDENCE_CONFIG = ROOT / "mcp-docker" / "configs" / "evidence_tools.yaml"
LOCAL_NO_PROXY = "localhost,127.0.0.1,::1,your-mcp-host"
ALLOWED_UNCONFIGURED_CASES = {
    # Kept for ad-hoc diagnostics, but not exposed in the default evidence tool
    # config because these public endpoints are too rate-limit prone here.
    "arxiv_search",
    "semantic_scholar_search",
}

CRITICAL_GPU_PREFLIGHTS: dict[str, list[Path]] = {
    "generate_crystal_structures_crystallm": [
        ROOT / "mcp-docker" / "crystallm-mcp" / "models" / "ckpt.pt",
    ],
    "predict_formation_energy_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "models" / "M3GNet-MP-2018.6.1-Eform" / "model.json",
    ],
    "predict_multi_fidelity_band_gap_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "models" / "MEGNet-MP-2019.4.1-BandGap-mfi" / "model.json",
    ],
    "relax_crystal_structure_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "models" / "M3GNet-MP-2021.2.8-PES" / "model.json",
    ],
    "calculate_pes_observables_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "relax_structure_with_qet_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "calculate_qet_pes_observables_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "compare_qet_and_tensornet_pes_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "predict_bulk_modulus_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "predict_elastic_properties_MatGL": [
        ROOT / "mcp-docker" / "matgl-mcp" / "hf-cache" / "hub",
    ],
    "uip_single_point_energy": [
        ROOT / "mcp-docker" / "uip-relax-mcp" / "model-cache" / "mace",
    ],
    "uip_force_stability_score": [
        ROOT / "mcp-docker" / "uip-relax-mcp" / "model-cache" / "mace",
    ],
    "uip_relax_structure": [
        ROOT / "mcp-docker" / "uip-relax-mcp" / "model-cache" / "mace",
    ],
}

NACL_CIF = """data_NaCl
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.640000
_cell_length_b   5.640000
_cell_length_c   5.640000
_cell_angle_alpha   90.000000
_cell_angle_beta    90.000000
_cell_angle_gamma   90.000000
_symmetry_Int_Tables_number 1
_chemical_formula_structural NaCl
_chemical_formula_sum 'Na1 Cl1'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 Na Na1 1 0.000000 0.000000 0.000000 1
 Cl Cl1 1 0.500000 0.500000 0.500000 1
"""

CS_PB_BR3_CIF = """data_CsPbBr3
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   5.870000
_cell_length_b   5.870000
_cell_length_c   5.870000
_cell_angle_alpha   90.000000
_cell_angle_beta    90.000000
_cell_angle_gamma   90.000000
_symmetry_Int_Tables_number 1
_chemical_formula_structural CsPbBr3
_chemical_formula_sum 'Cs1 Pb1 Br3'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 Cs Cs1 1 0.000000 0.000000 0.000000 1
 Pb Pb1 1 0.500000 0.500000 0.500000 1
 Br Br1 1 0.500000 0.500000 0.000000 1
 Br Br2 1 0.500000 0.000000 0.500000 1
 Br Br3 1 0.000000 0.500000 0.500000 1
"""

CU_SLAB_CIF = """data_Cu_slab
_symmetry_space_group_name_H-M   'P 1'
_cell_length_a   2.560000
_cell_length_b   2.560000
_cell_length_c   18.000000
_cell_angle_alpha   90.000000
_cell_angle_beta    90.000000
_cell_angle_gamma   90.000000
_symmetry_Int_Tables_number 1
_chemical_formula_structural Cu2
_chemical_formula_sum 'Cu2'
loop_
 _symmetry_equiv_pos_site_id
 _symmetry_equiv_pos_as_xyz
  1  'x, y, z'
loop_
 _atom_site_type_symbol
 _atom_site_label
 _atom_site_symmetry_multiplicity
 _atom_site_fract_x
 _atom_site_fract_y
 _atom_site_fract_z
 _atom_site_occupancy
 Cu Cu1 1 0.000000 0.000000 0.420000 1
 Cu Cu2 1 0.500000 0.500000 0.500000 1
"""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server_url: str
    profile: str


@dataclass(frozen=True)
class Case:
    arguments: dict[str, Any]
    tags: set[str] = field(default_factory=lambda: {"quick"})
    timeout: float | None = None
    optional: bool = False
    allow_error: bool = False
    note: str = ""


def load_specs(paths: list[Path]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for path in paths:
        profile = "evidence" if "evidence" in path.name else "main"
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for item in data.get("tools", []):
            specs.append(ToolSpec(item["name"], item["server_url"], profile))
    return specs


def force_local_no_proxy() -> None:
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged = ",".join(part for part in [existing, LOCAL_NO_PROXY] if part).strip(",")
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def local_mcp_httpx_client_factory(
    headers: dict[str, Any] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
        "trust_env": False,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


def t(*tags: str) -> set[str]:
    return set(tags)


def cases() -> dict[str, Case]:
    route = {"precursors": ["BaCO3", "TiO2"], "temperature_c": 900, "atmosphere": "air"}
    xrd_obs = [27.4, 31.7, 45.5, 56.5, 66.2]
    return {
        # Generation / ML potentials
        "register_cif_artifact": Case({"cif_string": NACL_CIF, "label": "quickcheck_input"}, t("quick", "artifact")),
        "generate_crystal_structures_crystallm": Case(
            {"formula": "NaCl", "space_group": "Pm-3m", "num_samples": 1},
            t("generation", "gpu", "crystallm", "slow"),
            timeout=180,
            optional=True,
        ),
        "predict_formation_energy_MatGL": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False}, t("gpu", "matgl"), timeout=120
        ),
        "predict_internal_energy_MatGL": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False}, t("gpu", "matgl"), timeout=120
        ),
        "calculate_single_point_energy_MatGL": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False}, t("gpu", "matgl"), timeout=120
        ),
        "calculate_pes_observables_MatGL": Case(
            {"cif_string": NACL_CIF, "model_ref": "PES", "optimize_structure": False},
            t("gpu", "matgl"),
            timeout=120,
        ),
        "predict_multi_fidelity_band_gap_MatGL": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False}, t("gpu", "matgl"), timeout=120
        ),
        "classify_metallicity_from_band_gap": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False, "fidelity": "SCAN", "threshold_eV": 0.05},
            t("gpu", "matgl"),
            timeout=120,
        ),
        "screen_magnetic_moments_CHGNet": Case(
            {"cif_string": CS_PB_BR3_CIF, "optimize_structure": False, "moment_threshold_muB": 0.5},
            t("gpu", "matgl"),
            timeout=120,
        ),
        "predict_bulk_modulus_MatGL": Case(
            {"cif_string": NACL_CIF, "relax_structure": False}, t("gpu", "matgl", "slow"), timeout=180
        ),
        "predict_elastic_properties_MatGL": Case(
            {"cif_string": NACL_CIF, "relax_structure": False, "symmetry": False, "include_tensors": False},
            t("gpu", "matgl", "slow"),
            timeout=240,
        ),
        "relax_crystal_structure_MatGL": Case(
            {"cif_string": NACL_CIF, "fmax": 0.2}, t("gpu", "matgl", "slow"), timeout=180
        ),
        "relax_structure_with_qet_MatGL": Case(
            {"cif_string": NACL_CIF, "variant": "PBE", "fmax": 0.2},
            t("gpu", "matgl", "slow"),
            timeout=180,
        ),
        "calculate_qet_pes_observables_MatGL": Case(
            {"cif_string": NACL_CIF, "variant": "PBE", "optimize_structure": False},
            t("gpu", "matgl"),
            timeout=120,
        ),
        "compare_qet_and_tensornet_pes_MatGL": Case(
            {"cif_string": NACL_CIF, "qet_variant": "PBE", "optimize_structure": False},
            t("gpu", "matgl"),
            timeout=180,
        ),

        # PyMatGen structure/property tools
        "analyze_cif_structure_pymatgen": Case({"cif_string": NACL_CIF}),
        "compare_structures_pymatgen": Case({"cif_string_a": NACL_CIF, "cif_string_b": NACL_CIF}),
        "score_generated_cif_against_target_pymatgen": Case(
            {"generated_cif": NACL_CIF, "target_formula": "NaCl", "target_num_sites": 2}
        ),
        "check_chemical_formula_valence_pymatgen": Case({"formula": "Fe2O3"}),
        "check_structure_atomic_geometry_pymatgen": Case({"cif_string": NACL_CIF}),
        "analyze_thermodynamic_stability_pymatgen": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False},
            t("mp", "network", "gpu", "slow"),
            timeout=180,
            optional=True,
        ),
        "estimate_energy_above_hull_pymatgen": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False},
            t("mp", "network", "gpu", "slow"),
            timeout=180,
            optional=True,
        ),
        "analyze_thermodynamic_stability_with_phase_diagram_pymatgen": Case(
            {"cif_string": NACL_CIF, "optimize_structure": False},
            t("mp", "network", "gpu", "slow"),
            timeout=240,
            optional=True,
        ),
        "simulate_xrd_pattern_pymatgen": Case({"cif_string": NACL_CIF}),
        "substitute_elements_in_structure_pymatgen": Case(
            {"cif_string": NACL_CIF, "substitutions": {"Na": "K"}}
        ),
        "apply_vegards_law_refinement_pymatgen": Case(
            {
                "cif_string": NACL_CIF,
                "endmember_lattices": {
                    "NaCl": {"a": 5.64, "b": 5.64, "c": 5.64},
                    "KCl": {"a": 6.29, "b": 6.29, "c": 6.29},
                },
                "fractions": {"NaCl": 0.75, "KCl": 0.25},
            }
        ),
        "enumerate_fractional_occupancy_structure_pymatgen": Case({"cif_string": NACL_CIF, "max_structures": 1}),
        "build_bulk_supercell_or_slab_pymatgen": Case(
            {
                "mode": "bulk",
                "lattice_type": "cubic",
                "species": ["Na", "Cl"],
                "frac_coords": [[0, 0, 0], [0.5, 0.5, 0.5]],
                "lattice_parameters": {"a": 5.64},
            }
        ),
        "standardize_structure_pymatgen": Case({"cif_string": NACL_CIF}),
        "extract_symmetry_and_wyckoff_pymatgen": Case({"cif_string": NACL_CIF}),
        "estimate_oxidation_states_pymatgen": Case({"formula": "Fe2O3", "cif_string": NACL_CIF}),
        "compute_bond_valence_pymatgen": Case({"cif_string": NACL_CIF}),
        "analyze_coordination_environment_pymatgen": Case({"cif_string": NACL_CIF}),
        "perturb_lattice_or_positions_pymatgen": Case(
            {"cif_string": NACL_CIF, "lattice_scale": {"a": 1.01}, "site_shifts": {"1": [0.01, 0, 0]}}
        ),
        "convert_cif_to_vasp_inputs_pymatgen": Case({"cif_string": NACL_CIF, "task": "static", "kppa": 100}),
        "parse_vasp_outputs_pymatgen": Case(
            {"vasprun_xml": "", "outcar_text": ""},
            t("fixture_gap"),
            allow_error=True,
            optional=True,
            note="Needs a real vasprun.xml/OUTCAR fixture for a positive parser-path test.",
        ),
        "generate_surface_slabs_pymatgen": Case(
            {"cif_string": NACL_CIF, "miller_indices": [[1, 0, 0]], "max_slabs_per_miller": 1},
            t("quick", "surface"),
            timeout=90,
        ),
        "find_adsorption_sites_pymatgen": Case({"slab_cif": CU_SLAB_CIF}, t("quick", "surface"), timeout=90),
        "place_adsorbate_on_slab_pymatgen": Case(
            {"slab_cif": CU_SLAB_CIF, "adsorbate": "N", "site": [1.28, 1.28, 10.0]},
            t("quick", "surface"),
            timeout=90,
        ),
        "generate_nrr_intermediates_on_surface_pymatgen": Case(
            {"slab_cif": CU_SLAB_CIF, "site": [1.28, 1.28, 10.0]},
            t("quick", "surface"),
            timeout=120,
        ),
        "analyze_collinear_magnetism_pymatgen": Case({"cif_string": NACL_CIF}),
        "suggest_initial_magmoms_pymatgen": Case({"cif_string": NACL_CIF}),
        "enumerate_magnetic_orderings_pymatgen": Case(
            {"cif_string": NACL_CIF, "max_orderings": 1},
            t("slow", "magnetism"),
            timeout=120,
            optional=True,
        ),

        # SMACT
        "screen_smact_oxidation_states": Case({"formula": "Fe2O3", "max_combinations": 8}),
        "check_smact_charge_neutrality": Case({"formula": "Fe2O3", "oxidation_states": {"Fe": 3, "O": -2}}),
        "check_smact_pauling_test": Case({"formula": "Fe2O3", "oxidation_states": {"Fe": 3, "O": -2}}),
        "score_composition_chemical_validity_smact": Case({"formula": "Fe2O3"}),

        # Matminer
        "featurize_composition_matminer": Case({"formula": "Fe2O3"}),
        "featurize_structure_matminer": Case({"cif_string": NACL_CIF}),
        "compare_structure_features_matminer": Case({"cif_string_a": NACL_CIF, "cif_string_b": NACL_CIF}),
        "rank_candidates_by_descriptor_distance_matminer": Case(
            {"target_formula": "Fe2O3", "candidate_formulas": ["FeO", "Al2O3", "LiFePO4"], "top_k": 2}
        ),

        # PyXtal
        "generate_crystal_pyxtal": Case(
            {"formula": "CsPbBr3", "space_group": 221, "attempts": 2, "max_returned": 1},
            t("quick", "generation"),
            timeout=90,
            optional=True,
        ),
        "check_wyckoff_feasibility_pyxtal": Case({"formula": "CsPbBr3", "space_group": 221}),
        "list_wyckoff_positions_pyxtal": Case({"space_group": 221}),
        "convert_pyxtal_to_pymatgen_summary": Case(
            {"formula": "CsPbBr3", "space_group": 221}, t("quick", "generation"), timeout=90, optional=True
        ),

        # Verifier
        "validate_formula_verifier": Case({"formula": "Fe2O3"}),
        "validate_structure_verifier": Case({"cif_string": NACL_CIF}),
        "score_candidate_structure_verifier": Case({"cif_string": NACL_CIF, "target_formula": "NaCl"}),
        "validate_synthesis_route_verifier": Case(
            {"target_formula": "BaTiO3", "precursor_formulas": ["BaCO3", "TiO2"], "temperature_c": 900, "atmosphere": "air"}
        ),

        # Materials DB router
        "search_materials_by_formula_db_router": Case({"formula": "BaTiO3", "top_k": 2}, t("mp", "network"), optional=True),
        "search_materials_by_chemsys_db_router": Case({"chemsys": "Ba-Ti-O", "top_k": 2}, t("mp", "network"), optional=True),
        "fetch_mp_structure_db_router": Case({"material_id": "mp-2998"}, t("mp", "network"), optional=True),
        "check_novelty_against_mp_db_router": Case({"cif_string": NACL_CIF, "top_k": 2}, t("mp", "network"), optional=True),
        "optimade_structure_search_db_router": Case(
            {"filter": 'chemical_formula_reduced="NaCl"', "provider": "cod", "page_limit": 1},
            t("network", "optimade"),
            optional=True,
        ),
        "compare_materials_across_databases_db_router": Case(
            {"formula": "NaCl", "providers": ["cod"], "top_k": 1}, t("mp", "network", "optimade"), optional=True
        ),

        # Synthesis KB
        "search_synthesis_recipes_kb": Case({"target_formula": "BaTiO3", "top_k": 2}),
        "recommend_precursors_kb": Case({"target_formula": "BaTiO3", "top_k": 4}),
        "recommend_synthesis_conditions_kb": Case({"target_formula": "BaTiO3"}),
        "rank_synthesis_routes_kb": Case({"target_formula": "BaTiO3", "candidate_routes": [route], "top_k": 1}),
        "balance_precursor_reaction_kb": Case(
            {"target_formula": "BaTiO3", "precursor_formulas": ["BaCO3", "TiO2"], "byproducts": ["CO2"]}
        ),
        "check_precursor_compatibility_kb": Case(
            {"target_formula": "BaTiO3", "precursor_formulas": ["BaCO3", "TiO2"], "atmosphere": "air"}
        ),

        # UIP / MACE
        "uip_single_point_energy": Case(
            {"cif_string": NACL_CIF, "model_name": "small", "device": "cpu", "dtype": "float32"},
            t("gpu", "mace"),
            timeout=180,
        ),
        "uip_relax_structure": Case(
            {"cif_string": NACL_CIF, "model_name": "small", "device": "cpu", "dtype": "float32", "max_steps": 2, "fmax": 0.5},
            t("gpu", "mace", "slow"),
            timeout=240,
        ),
        "uip_force_stability_score": Case(
            {"cif_string": NACL_CIF, "model_name": "small", "device": "cpu", "dtype": "float32"},
            t("gpu", "mace"),
            timeout=180,
        ),
        "calculate_discovery_reward": Case(
            {"cif_string": NACL_CIF, "target_formula": "NaCl", "run_uip_relax": False}
        ),
        "compare_uip_relaxations": Case(
            {"cif_string": NACL_CIF, "model_names": ["small"], "device": "cpu", "dtype": "float32", "max_steps": 1, "fmax": 0.8},
            t("gpu", "mace", "slow"),
            timeout=240,
        ),

        # Literature/evidence
        "tavily_search": Case({"query": "BaTiO3 solid state synthesis", "top_k": 1}, t("network", "api_key"), optional=True),
        "bing_search": Case({"query": "BaTiO3 solid state synthesis", "top_k": 1}, t("network", "api_key"), optional=True),
        "semantic_scholar_search": Case({"query": "BaTiO3 solid state synthesis", "top_k": 1}, t("network"), optional=True),
        "openalex_search": Case({"query": "BaTiO3 solid state synthesis", "top_k": 1}, t("network"), optional=True),
        "crossref_search": Case({"query": "BaTiO3 solid state synthesis", "top_k": 1}, t("network"), optional=True),
        "arxiv_search": Case({"query": "materials synthesis text mining", "top_k": 1}, t("network"), optional=True),
        "pubchem_search": Case({"query": "ethanol", "top_k": 1}, t("network"), optional=True),
        "evidence_search_all": Case(
            {"query": "BaTiO3 solid state synthesis", "top_k_per_source": 1, "include_web": False},
            t("network"),
            optional=True,
        ),

        # Reaction thermo
        "balance_reaction_thermo": Case(
            {"reactant_formulas": ["BaCO3", "TiO2"], "product_formulas": ["BaTiO3", "CO2"]}
        ),
        "calculate_reaction_energy_mp": Case(
            {"reactant_formulas": ["BaCO3", "TiO2"], "product_formulas": ["BaTiO3", "CO2"]},
            t("mp", "network"),
            optional=True,
        ),
        "rank_precursor_reactions_thermo": Case(
            {"target_formula": "BaTiO3", "candidate_precursor_sets": [["BaCO3", "TiO2"]], "byproducts": ["CO2"], "top_k": 1}
        ),
        "analyze_phase_stability_mp": Case({"formula": "BaTiO3"}, t("mp", "network"), optional=True),
        "analyze_open_system_stability_mp": Case(
            {"formula": "BaTiO3", "open_element": "O", "chemical_potential_eV": -1.0},
            t("mp", "network"),
            optional=True,
        ),

        # Characterization
        "simulate_xrd_pattern_characterization": Case({"cif_string": NACL_CIF}),
        "extract_xrd_peak_table_characterization": Case({"cif_string": NACL_CIF, "top_k": 3}),
        "compare_xrd_patterns_characterization": Case({"cif_string_a": NACL_CIF, "cif_string_b": NACL_CIF}),
        "match_xrd_to_candidate_structures_characterization": Case(
            {"reference_cif": NACL_CIF, "candidate_cifs": [NACL_CIF], "candidate_labels": ["NaCl"], "top_k": 1}
        ),
        "score_phase_purity_from_xrd_characterization": Case(
            {"target_cif": NACL_CIF, "observed_two_theta": xrd_obs, "observed_intensity": [1, 1, 1, 1, 1]}
        ),

        # Precursor chemistry
        "resolve_precursor_pubchem": Case({"name_or_formula": "ethanol", "top_k": 1}, t("network", "pubchem"), optional=True),
        "get_pubchem_compound_properties": Case({"identifier": "ethanol"}, t("network", "pubchem"), optional=True),
        "get_pubchem_safety_summary": Case({"name_or_cid": "ethanol"}, t("network", "pubchem"), optional=True),
        "canonicalize_smiles_rdkit": Case({"smiles": "CCO"}),
        "check_precursor_hazard_flags": Case({"precursor_names": ["ethanol"]}, t("network", "pubchem"), optional=True),
        "check_solvent_ligand_compatibility": Case({"solvent": "water", "ligand": "ethanol"}, t("network", "pubchem"), optional=True),

        # Tool registry
        "list_mat_mcp_services": Case({"profile": "all"}),
        "list_tools_by_domain": Case({"domain": "synthesis", "profile": "all"}),
        "select_tools_for_task": Case({"task": "synthesis_planning", "profile": "main", "max_tools": 8}),
        "get_tool_policy_profile": Case({}),

        # Defects/interfaces
        "generate_vacancy_defects": Case({"cif_string": NACL_CIF, "max_structures": 1}),
        "generate_substitutional_dopants": Case(
            {"cif_string": NACL_CIF, "dopant_element": "K", "host_species": "Na", "max_structures": 1}
        ),
        "generate_interstitial_defects": Case(
            {
                "cif_string": NACL_CIF,
                "interstitial_element": "Li",
                "candidate_frac_coords": [[0.25, 0.25, 0.25]],
                "min_distance_angstrom": 1.0,
                "max_structures": 1,
            }
        ),
        "calculate_lattice_mismatch": Case({"substrate_cif": NACL_CIF, "film_cif": NACL_CIF}),
        "generate_heterostructure_interface": Case({"substrate_cif": NACL_CIF, "film_cif": NACL_CIF}),
        "generate_diffusion_path": Case(
            {
                "cif_string": NACL_CIF,
                "migrating_element": "Li",
                "start_frac_coords": [0.25, 0.25, 0.25],
                "end_frac_coords": [0.75, 0.75, 0.75],
                "n_images": 3,
                "include_cifs": False,
            }
        ),
    }


def selected(case: Case, selected_tags: set[str], excluded_tags: set[str]) -> bool:
    if selected_tags == {"all"}:
        included = True
    else:
        included = bool(case.tags & selected_tags)
    return included and not bool(case.tags & excluded_tags)


async def call_tool(server_url: str, tool_name: str, arguments: dict[str, Any], timeout: float) -> str:
    result = await runtime_call_tool(server_url, tool_name, arguments, timeout)
    if not result.ok:
        raise RuntimeError(result.error or result.content or f"tool call failed: {tool_name}")
    return result.content or ""


def parse_json_payload(text: str) -> Any | None:
    stripped = text.strip()
    fenced_match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", stripped, flags=re.S)
    if fenced_match:
        try:
            return json.loads(fenced_match.group(1).strip())
        except Exception:
            return None
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            return json.loads(stripped)
        except Exception:
            return None
    return None


def preflight_rows_for_tool(name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in CRITICAL_GPU_PREFLIGHTS.get(name, []):
        exists = path.exists()
        readable = exists and os.access(path, os.R_OK)
        row = {
            "tool": name,
            "preflight_path": str(path),
            "exists": exists,
            "readable": readable,
            "ok": bool(exists and readable),
        }
        if exists:
            row["mode"] = oct(path.stat().st_mode & 0o777)
            row["kind"] = "file" if path.is_file() else "dir"
        rows.append(row)
    return rows


def semantic_validation(name: str, text: str) -> tuple[bool, str]:
    parsed = parse_json_payload(text)
    if name == "generate_crystal_structures_crystallm":
        if "# CrystaLLM Generated Crystal Structures" not in text or "## Structure" not in text:
            return False, "missing crystallm structure block"
        return True, "semantic_ok"

    if name == "register_cif_artifact":
        if not isinstance(parsed, dict):
            return False, "artifact payload is not json"
        required = ("ok", "cif_ref", "canonical_ref", "artifact_type")
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "artifact registration missing fields"
        return True, "semantic_ok"

    if name == "predict_formation_energy_MatGL":
        if "形成能预测结果" not in text or "eV/atom" not in text:
            return False, "missing formation energy payload"
        return True, "semantic_ok"

    if name == "predict_internal_energy_MatGL":
        if not isinstance(parsed, dict):
            return False, "internal-energy payload is not json"
        required = ("ok", "formula", "total_internal_energy_eV", "energy_per_atom", "num_atoms")
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "internal-energy payload missing fields"
        return True, "semantic_ok"

    if name == "predict_multi_fidelity_band_gap_MatGL":
        required = ("多保真度带隙预测结果", "PBE", "HSE", "SCAN")
        if not all(token in text for token in required):
            return False, "missing band-gap table"
        return True, "semantic_ok"

    if name == "classify_metallicity_from_band_gap":
        if not isinstance(parsed, dict):
            return False, "metallicity payload is not json"
        required = ("ok", "formula", "selected_fidelity", "band_gap_eV", "threshold_eV", "is_metal", "classification")
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "metallicity payload missing fields"
        return True, "semantic_ok"

    if name == "screen_magnetic_moments_CHGNet":
        if not isinstance(parsed, dict):
            return False, "magnetic-screen payload is not json"
        required = (
            "ok",
            "formula",
            "moment_threshold_muB",
            "has_nonzero_local_moments",
            "num_magnetic_sites",
            "magnetic_species",
            "total_magnetic_moment_muB",
            "screening_interpretation",
        )
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "magnetic-screen payload missing fields"
        return True, "semantic_ok"

    if name == "analyze_thermodynamic_stability_pymatgen":
        required = ("热力学稳定性分析结果", "Energy Above Hull", "热力学稳定性")
        if not all(token in text for token in required):
            return False, "missing thermo-stability summary"
        return True, "semantic_ok"

    if name == "estimate_energy_above_hull_pymatgen":
        if not isinstance(parsed, dict):
            return False, "ehull payload is not json"
        required = (
            "ok",
            "formula",
            "formation_energy_per_atom_eV",
            "energy_above_hull_eV_per_atom",
            "energy_threshold_eV_per_atom",
            "is_stable",
            "decomposition",
        )
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "ehull payload missing fields"
        return True, "semantic_ok"

    if name == "relax_crystal_structure_MatGL":
        if "结构优化结果" not in text or "优化状态" not in text:
            return False, "missing relaxation summary"
        return True, "semantic_ok"

    if name == "uip_single_point_energy":
        if not isinstance(parsed, dict):
            return False, "uip payload is not json"
        required = ("ok", "energy_per_atom_eV", "max_force_eV_A", "formula")
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "uip single-point missing fields"
        return True, "semantic_ok"

    if name == "uip_force_stability_score":
        if not isinstance(parsed, dict):
            return False, "uip force payload is not json"
        if parsed.get("ok") is not True or "single_point" not in parsed or "stability" not in parsed:
            return False, "uip force score missing fields"
        return True, "semantic_ok"

    if name == "uip_relax_structure":
        if not isinstance(parsed, dict):
            return False, "uip relax payload is not json"
        required = ("ok", "converged", "initial", "final", "relaxed_cif", "stability")
        if not all(key in parsed for key in required) or parsed.get("ok") is not True:
            return False, "uip relax missing fields"
        if "data_" not in str(parsed.get("relaxed_cif", "")):
            return False, "uip relax missing relaxed cif"
        return True, "semantic_ok"

    return True, "semantic_skip"


def classify_result(name: str, text: str, case: Case) -> tuple[bool, str]:
    stripped = text.strip()
    if not stripped:
        return False, "empty MCP response"
    error_like = (
        stripped.startswith("Error:")
        or stripped.startswith("Error：")
        or "PermissionError:" in stripped
        or "RuntimeError:" in stripped
        or "Failed during " in stripped
        or "Error calling tool" in stripped
        or "Traceback (" in stripped
    )
    parsed = parse_json_payload(stripped)
    json_ok_false = isinstance(parsed, dict) and parsed.get("ok") is False
    if error_like or json_ok_false:
        if case.allow_error:
            return True, "allowed_error"
        if case.optional:
            return True, "optional_failure"
        return False, "tool returned error"
    semantic_ok, semantic_status = semantic_validation(name, stripped)
    if not semantic_ok:
        if case.allow_error:
            return True, f"allowed_{semantic_status}"
        if case.optional:
            return True, f"optional_{semantic_status}"
        return False, semantic_status
    return True, semantic_status


async def run_case(name: str, server_url: str, case: Case, default_timeout: float) -> dict[str, Any]:
    timeout = float(case.timeout or default_timeout)
    started = time.perf_counter()
    row: dict[str, Any] = {
        "tool": name,
        "server_url": server_url,
        "tags": sorted(case.tags),
        "optional": case.optional,
        "allow_error": case.allow_error,
        "arguments": case.arguments,
    }
    try:
        text = await call_tool(server_url, name, case.arguments, timeout)
        elapsed = time.perf_counter() - started
        ok, status = classify_result(name, text, case)
        row.update(
            {
                "ok": ok,
                "status": status,
                "elapsed_s": round(elapsed, 3),
                "result_excerpt": text[:1200],
                "result_size": len(text),
            }
        )
    except Exception as exc:
        elapsed = time.perf_counter() - started
        if case.optional:
            ok = True
            status = "optional_exception"
        elif case.allow_error:
            ok = True
            status = "allowed_exception"
        else:
            ok = False
            status = "exception"
        row.update({"ok": ok, "status": status, "elapsed_s": round(elapsed, 3), "error": repr(exc)})
    if case.note:
        row["note"] = case.note
    return row


def parse_tags(value: str) -> set[str]:
    tags = {item.strip() for item in value.split(",") if item.strip()}
    return tags or {"quick"}


def parse_tools(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


async def main() -> None:
    force_local_no_proxy()
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, default=MAIN_CONFIG)
    parser.add_argument("--evidence-config", type=Path, default=EVIDENCE_CONFIG)
    parser.add_argument("--tags", default="quick", help="Comma-separated tags to run, or all. Default: quick")
    parser.add_argument("--exclude-tags", default="", help="Comma-separated tags to skip.")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--tool", default="", help="Comma-separated tool names to run.")
    parser.add_argument("--list-tags", action="store_true", help="Print known fixture tags and exit.")
    parser.add_argument("--coverage-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--strict-optional", action="store_true", help="Count optional/API-key/network failures as failures.")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--results-jsonl", type=Path, default=None)
    args = parser.parse_args()

    all_cases = cases()
    if args.list_tags:
        known_tags = sorted({tag for case in all_cases.values() for tag in case.tags})
        print(json.dumps({"tags": known_tags}, ensure_ascii=False), flush=True)
        return

    specs = load_specs([args.main_config, args.evidence_config])
    server_by_name: dict[str, str] = {}
    profiles_by_name: dict[str, set[str]] = {}
    for spec in specs:
        server_by_name.setdefault(spec.name, spec.server_url)
        profiles_by_name.setdefault(spec.name, set()).add(spec.profile)

    configured = set(server_by_name)
    missing = sorted(configured - set(all_cases))
    extra = sorted((set(all_cases) - configured) - ALLOWED_UNCONFIGURED_CASES)
    coverage = {
        "configured_unique_tools": len(configured),
        "fixture_cases": len(all_cases),
        "missing_cases": missing,
        "extra_cases_not_in_configs": extra,
    }
    print(json.dumps({"coverage": coverage}, ensure_ascii=False), flush=True)
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing fixture cases for configured tools: {missing}")
    if args.coverage_only:
        return

    selected_tags = parse_tags(args.tags)
    excluded_tags = parse_tags(args.exclude_tags) if args.exclude_tags else set()
    requested_tools = parse_tools(args.tool)
    selected_items = [
        (name, server_by_name[name], all_cases[name])
        for name in sorted(configured)
        if name in all_cases and selected(all_cases[name], selected_tags, excluded_tags)
        and (not requested_tools or name in requested_tools)
    ]
    unknown_requested = sorted(requested_tools - configured)
    if unknown_requested:
        raise SystemExit(f"Requested tools are not in configs: {unknown_requested}")
    print(
        json.dumps(
            {
                "selected": {
                    "tags": sorted(selected_tags),
                    "exclude_tags": sorted(excluded_tags),
                    "tool_count": len(selected_items),
                }
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    preflight_failures: list[dict[str, Any]] = []
    seen_preflight: set[tuple[str, str]] = set()
    for name, _, case in selected_items:
        for row in preflight_rows_for_tool(name):
            key = (row["tool"], row["preflight_path"])
            if key in seen_preflight:
                continue
            seen_preflight.add(key)
            row["status"] = "preflight_ok" if row["ok"] else "preflight_failed"
            print(json.dumps({"preflight": row}, ensure_ascii=False), flush=True)
            if not row["ok"]:
                preflight_failures.append(row)
    if preflight_failures:
        failed = [f'{row["tool"]}:{row["preflight_path"]}' for row in preflight_failures]
        raise SystemExit(f"Mat-MCP preflight failed for {failed}")

    output_handle = None
    if args.results_jsonl:
        args.results_jsonl.parent.mkdir(parents=True, exist_ok=True)
        output_handle = args.results_jsonl.open("w", encoding="utf-8")

    failures: list[dict[str, Any]] = []
    try:
        for name, server_url, case in selected_items:
            row = await run_case(name, server_url, case, args.timeout)
            if args.strict_optional and row["status"].startswith("optional"):
                row["ok"] = False
            row["profiles"] = sorted(profiles_by_name.get(name, []))
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if output_handle:
                output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                output_handle.flush()
            if not row["ok"]:
                failures.append(row)
                if args.fail_fast:
                    break
    finally:
        if output_handle:
            output_handle.close()
        await close_all_session_pools()

    summary = {
        "tested": len(selected_items),
        "failures": len(failures),
        "failed_tools": [row["tool"] for row in failures],
    }
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)
    if failures:
        raise SystemExit(f"Mat-MCP tool-call check failed for {len(failures)} tools")
    print("Mat-MCP tool-call check passed.", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
