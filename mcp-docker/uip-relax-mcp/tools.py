import json
import os
import gc
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import torch
from ase.filters import FrechetCellFilter
from ase.io import read, write
from ase.optimize import FIRE
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from core import llm_tool


DEFAULT_DEVICE = os.getenv("UIP_DEVICE", "cuda")
DEFAULT_DTYPE = os.getenv("UIP_DTYPE", "float32")
DEFAULT_MACE_MODEL = os.getenv("MACE_MP_MODEL", "medium")
DEFAULT_MAX_STEPS = int(os.getenv("UIP_MAX_RELAX_STEPS", "120"))
DEFAULT_FMAX = float(os.getenv("UIP_FMAX", "0.05"))
DEFAULT_CUDA_TRIM_AFTER_TOOL = os.getenv("UIP_CUDA_TRIM_AFTER_TOOL", "1").strip().lower() not in {"0", "false", "no"}


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _load_structure(cif_string: str) -> Structure:
    return Structure.from_str(cif_string, fmt="cif")


def _atoms_from_cif(cif_string: str):
    structure = _load_structure(cif_string)
    return AseAtomsAdaptor.get_atoms(structure)


def _structure_to_cif(structure: Structure) -> str:
    return structure.to(fmt="cif")


def _atoms_to_cif(atoms) -> str:
    structure = AseAtomsAdaptor.get_structure(atoms)
    return _structure_to_cif(structure)


def _finite(value: Any) -> Any:
    try:
        val = float(value)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return value


def _inference_mode():
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


def _normalize_uip_runtime(
    model_family: str,
    model_name: str,
    device: str,
    dtype: str,
) -> tuple[str, str, str, str]:
    family = str(model_family or "mace_mp").strip().lower()
    if family in {"mace", "mace-mp", "mace_mp", "mace-mp-0", "mace_mp_0"}:
        family = "mace_mp"
    else:
        family = "mace_mp"

    name = str(model_name or DEFAULT_MACE_MODEL).strip().lower()
    if name not in {"small", "medium"}:
        name = DEFAULT_MACE_MODEL
    if name not in {"small", "medium"}:
        name = "medium"

    normalized_device = str(device or DEFAULT_DEVICE).strip().lower()
    if normalized_device in {"", "cpu"}:
        normalized_device = DEFAULT_DEVICE

    normalized_dtype = str(dtype or DEFAULT_DTYPE).strip().lower() or DEFAULT_DTYPE
    return family, name, normalized_device, normalized_dtype


def _detach_calculator(atoms) -> None:
    if atoms is not None and getattr(atoms, "calc", None) is not None:
        atoms.calc = None


def _trim_cuda_cache() -> None:
    if DEFAULT_CUDA_TRIM_AFTER_TOOL and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _finalize_tool_cleanup(*atoms_objects) -> None:
    for atoms in atoms_objects:
        _detach_calculator(atoms)
    gc.collect()
    _trim_cuda_cache()


@lru_cache(maxsize=8)
def _get_mace_calculator(model: str, device: str, dtype: str):
    from mace.calculators import mace_mp

    return mace_mp(
        model=model,
        device=device,
        default_dtype=dtype,
    )


def _get_calculator(model_family: str, model_name: str, device: str, dtype: str):
    family = model_family.lower()
    if family in {"mace", "mace-mp", "mace_mp"}:
        return _get_mace_calculator(model_name or DEFAULT_MACE_MODEL, device, dtype)
    raise ValueError(f"Unsupported UIP model_family={model_family}. Supported: mace_mp")


def _single_point(atoms, calculator) -> Dict[str, Any]:
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    forces = np.array(atoms.get_forces(), dtype=float)
    stress = None
    try:
        stress = [_finite(x) for x in atoms.get_stress()]
    except Exception:
        stress = None
    force_norms = np.linalg.norm(forces, axis=1) if len(forces) else np.array([0.0])
    return {
        "energy_eV": energy,
        "energy_per_atom_eV": float(energy / max(len(atoms), 1)),
        "max_force_eV_A": float(np.max(force_norms)),
        "mean_force_eV_A": float(np.mean(force_norms)),
        "stress": stress,
        "volume_A3": float(atoms.get_volume()),
        "volume_per_atom_A3": float(atoms.get_volume() / max(len(atoms), 1)),
    }


def _force_stability_score(max_force: float, volume_per_atom: float, initial_volume_per_atom: float | None = None) -> Dict[str, Any]:
    force_score = float(max(0.0, min(1.0, 1.0 - max(0.0, max_force - 0.05) / 0.95)))
    volume_score = 1.0
    warnings: List[str] = []
    if volume_per_atom < 3.0 or volume_per_atom > 200.0:
        volume_score = 0.0
        warnings.append("relaxed volume per atom outside broad physical range")
    collapse_score = 1.0
    if initial_volume_per_atom:
        ratio = volume_per_atom / max(initial_volume_per_atom, 1e-8)
        if ratio < 0.55 or ratio > 1.80:
            collapse_score = 0.0
            warnings.append(f"large volume change during relaxation: ratio={ratio:.3f}")
        elif ratio < 0.75 or ratio > 1.35:
            collapse_score = 0.5
            warnings.append(f"moderate volume change during relaxation: ratio={ratio:.3f}")
    score = 0.55 * force_score + 0.25 * volume_score + 0.20 * collapse_score
    return {
        "score": float(score),
        "force_score": force_score,
        "volume_score": float(volume_score),
        "collapse_score": float(collapse_score),
        "warnings": warnings,
    }


@llm_tool(
    name="uip_single_point_energy",
    description="Compute universal interatomic potential single-point energy, forces, and stress for a CIF structure.",
)
async def uip_single_point_energy(
    cif_string: str,
    model_family: str = "mace_mp",
    model_name: str = "medium",
    device: str = "",
    dtype: str = "",
) -> str:
    model_family, model_name, device, dtype = _normalize_uip_runtime(model_family, model_name, device, dtype)
    atoms = _atoms_from_cif(cif_string)
    try:
        calculator = _get_calculator(model_family, model_name or DEFAULT_MACE_MODEL, device, dtype)
        result = _single_point(atoms, calculator)
        payload = {
            "ok": True,
            "model_family": model_family,
            "model_name": model_name or DEFAULT_MACE_MODEL,
            "device": device,
            "dtype": dtype,
            "formula": _load_structure(cif_string).composition.reduced_formula,
            "num_atoms": len(atoms),
            **result,
            "notes": [
                "UIP raw energies are surrogate rewards and should not be mixed directly with MP corrected hull energies."
            ],
        }
        return _json_text(payload)
    finally:
        _finalize_tool_cleanup(atoms)


@llm_tool(
    name="uip_relax_structure",
    description="Relax a CIF structure using a universal interatomic potential and return relaxed CIF plus reward-ready stability metrics.",
)
async def uip_relax_structure(
    cif_string: str,
    model_family: str = "mace_mp",
    model_name: str = "medium",
    device: str = "",
    dtype: str = "",
    fmax: float = DEFAULT_FMAX,
    max_steps: int = DEFAULT_MAX_STEPS,
    relax_cell: bool = True,
) -> str:
    model_family, model_name, device, dtype = _normalize_uip_runtime(model_family, model_name, device, dtype)
    max_steps = min(max(1, int(max_steps)), DEFAULT_MAX_STEPS)
    atoms = _atoms_from_cif(cif_string)
    initial_atoms = None
    opt_target = None
    optimizer = None
    try:
        initial_atoms = atoms.copy()
        initial = _single_point(
            initial_atoms,
            _get_calculator(model_family, model_name or DEFAULT_MACE_MODEL, device, dtype),
        )
        calculator = _get_calculator(model_family, model_name or DEFAULT_MACE_MODEL, device, dtype)
        atoms.calc = calculator
        opt_target = FrechetCellFilter(atoms) if relax_cell else atoms
        optimizer = FIRE(opt_target, logfile=None)
        optimizer.run(fmax=float(fmax), steps=max_steps)
        final = _single_point(atoms, calculator)
        stability = _force_stability_score(
            final["max_force_eV_A"],
            final["volume_per_atom_A3"],
            initial["volume_per_atom_A3"],
        )
        payload = {
            "ok": True,
            "model_family": model_family,
            "model_name": model_name or DEFAULT_MACE_MODEL,
            "device": device,
            "dtype": dtype,
            "fmax_target": float(fmax),
            "max_steps": max_steps,
            "relax_cell": bool(relax_cell),
            "converged": bool(final["max_force_eV_A"] <= float(fmax)),
            "initial": initial,
            "final": final,
            "delta_energy_per_atom_eV": float(final["energy_per_atom_eV"] - initial["energy_per_atom_eV"]),
            "volume_change_ratio": float(final["volume_per_atom_A3"] / max(initial["volume_per_atom_A3"], 1e-8)),
            "relaxed_cif": _atoms_to_cif(atoms),
            "stability": stability,
            "reward_components": {
                "uip_relaxation_stability": stability["score"],
                "uip_force_score": stability["force_score"],
                "uip_volume_score": stability["volume_score"],
                "uip_energy_drop_per_atom": float(initial["energy_per_atom_eV"] - final["energy_per_atom_eV"]),
            },
            "notes": [
                "UIP raw energies are surrogate rewards and should not be mixed directly with MP corrected hull energies."
            ],
        }
        return _json_text(payload)
    finally:
        del optimizer
        del opt_target
        _finalize_tool_cleanup(atoms, initial_atoms)


@llm_tool(
    name="uip_force_stability_score",
    description="Score force and volume stability from a UIP single point without relaxing the structure.",
)
async def uip_force_stability_score(
    cif_string: str,
    model_family: str = "mace_mp",
    model_name: str = "medium",
    device: str = "",
    dtype: str = "",
) -> str:
    model_family, model_name, device, dtype = _normalize_uip_runtime(model_family, model_name, device, dtype)
    atoms = _atoms_from_cif(cif_string)
    try:
        calculator = _get_calculator(model_family, model_name or DEFAULT_MACE_MODEL, device, dtype)
        result = _single_point(atoms, calculator)
        stability = _force_stability_score(result["max_force_eV_A"], result["volume_per_atom_A3"])
        payload = {
            "ok": True,
            "model_family": model_family,
            "model_name": model_name or DEFAULT_MACE_MODEL,
            "device": device,
            "single_point": result,
            "stability": stability,
            "reward_components": {
                "uip_force_stability": stability["score"],
                "uip_force_score": stability["force_score"],
                "uip_volume_score": stability["volume_score"],
            },
        }
        return _json_text(payload)
    finally:
        _finalize_tool_cleanup(atoms)


@llm_tool(
    name="calculate_discovery_reward",
    description="Compute a dense discovery reward from formula/geometry validity plus optional UIP relaxation stability.",
)
async def calculate_discovery_reward(
    cif_string: str,
    target_formula: str = "",
    run_uip_relax: bool = True,
    model_family: str = "mace_mp",
    model_name: str = "small",
    device: str = "",
    dtype: str = "",
    fmax: float = 0.08,
    max_steps: int = 80,
) -> str:
    from pymatgen.analysis.bond_valence import BVAnalyzer
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

    structure = _load_structure(cif_string)
    errors: List[str] = []
    warnings: List[str] = []
    min_dist = 999.0
    if len(structure) > 1:
        matrix = np.array(structure.distance_matrix, dtype=float)
        np.fill_diagonal(matrix, np.inf)
        min_dist = float(np.min(matrix))
    volume_per_atom = float(structure.volume / max(len(structure), 1))
    geometry_score = 1.0
    if min_dist < 0.7:
        geometry_score -= 0.45
        errors.append("minimum interatomic distance below 0.7 A")
    if volume_per_atom < 3.0 or volume_per_atom > 200.0:
        geometry_score -= 0.35
        errors.append("volume per atom outside broad physical range")
    geometry_score = float(max(0.0, geometry_score))

    formula_score = 1.0
    if target_formula:
        from pymatgen.core import Composition

        target = Composition(target_formula).fractional_composition
        actual = structure.composition.fractional_composition
        elements = sorted({el.symbol for el in target.elements} | {el.symbol for el in actual.elements})
        l1 = sum(abs(float(target.get_atomic_fraction(el)) - float(actual.get_atomic_fraction(el))) for el in elements)
        formula_score = float(max(0.0, 1.0 - l1))
        if l1 > 1e-6:
            warnings.append(f"composition differs from target; fractional L1={l1:.4f}")

    symmetry_score = 0.8
    symmetry = {"available": False}
    try:
        sga = SpacegroupAnalyzer(structure, symprec=0.1)
        symmetry = {
            "available": True,
            "space_group_number": int(sga.get_space_group_number()),
            "space_group_symbol": sga.get_space_group_symbol(),
        }
        symmetry_score = 1.0
    except Exception as exc:
        warnings.append(f"symmetry analysis failed: {exc}")

    bond_valence_score = 0.7
    try:
        BVAnalyzer().get_oxi_state_decorated_structure(structure)
        bond_valence_score = 1.0
    except Exception as exc:
        warnings.append(f"bond-valence analysis failed: {exc}")

    uip_score = 0.0
    uip_payload = None
    if run_uip_relax:
        text = await uip_relax_structure(
            cif_string,
            model_family=model_family,
            model_name=model_name,
            device=device,
            dtype=dtype,
            fmax=fmax,
            max_steps=max_steps,
            relax_cell=True,
        )
        uip_payload = json.loads(text)
        uip_score = float(uip_payload.get("stability", {}).get("score", 0.0))
    else:
        uip_score = 0.5

    reward = (
        0.25 * formula_score
        + 0.25 * geometry_score
        + 0.10 * symmetry_score
        + 0.10 * bond_valence_score
        + 0.30 * uip_score
    )
    payload = {
        "ok": True,
        "formula": structure.composition.reduced_formula,
        "target_formula": target_formula,
        "valid": bool(reward >= 0.70 and not errors),
        "discovery_reward": float(reward),
        "errors": errors,
        "warnings": warnings,
        "subscores": {
            "formula_score": formula_score,
            "geometry_score": geometry_score,
            "symmetry_score": symmetry_score,
            "bond_valence_score": bond_valence_score,
            "uip_score": uip_score,
        },
        "structure_metrics": {
            "min_distance_A": min_dist,
            "volume_per_atom_A3": volume_per_atom,
            "symmetry": symmetry,
        },
        "uip_result": uip_payload,
        "reward_components": {
            "discovery_reward": float(reward),
            "formula_reward": formula_score,
            "geometry_reward": geometry_score,
            "symmetry_reward": symmetry_score,
            "bond_valence_reward": bond_valence_score,
            "uip_reward": uip_score,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="compare_uip_relaxations",
    description="Compare relaxation stability from one or more UIP model settings and return disagreement metrics.",
)
async def compare_uip_relaxations(
    cif_string: str,
    model_names: List[str] = None,
    device: str = "",
    dtype: str = "",
    fmax: float = 0.08,
    max_steps: int = 80,
) -> str:
    model_names = model_names or ["small", "medium"]
    rows = []
    for model_name in model_names:
        text = await uip_relax_structure(
            cif_string,
            model_family="mace_mp",
            model_name=model_name,
            device=device,
            dtype=dtype,
            fmax=fmax,
            max_steps=max_steps,
            relax_cell=True,
        )
        data = json.loads(text)
        rows.append({
            "model_name": model_name,
            "ok": data.get("ok", False),
            "energy_per_atom_eV": data.get("final", {}).get("energy_per_atom_eV"),
            "max_force_eV_A": data.get("final", {}).get("max_force_eV_A"),
            "volume_per_atom_A3": data.get("final", {}).get("volume_per_atom_A3"),
            "stability_score": data.get("stability", {}).get("score"),
        })
    energies = [row["energy_per_atom_eV"] for row in rows if row["energy_per_atom_eV"] is not None]
    scores = [row["stability_score"] for row in rows if row["stability_score"] is not None]
    payload = {
        "ok": True,
        "results": rows,
        "disagreement": {
            "energy_per_atom_std_eV": float(np.std(energies)) if energies else None,
            "stability_score_std": float(np.std(scores)) if scores else None,
        },
        "reward_components": {
            "multi_uip_agreement": float(max(0.0, 1.0 - np.std(scores))) if scores else 0.0,
            "mean_uip_stability": float(np.mean(scores)) if scores else 0.0,
        },
    }
    return _json_text(payload)
