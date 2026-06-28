import itertools
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.analysis.phase_diagram import GrandPotentialPhaseDiagram, PhaseDiagram
from pymatgen.analysis.reaction_calculator import Reaction
from pymatgen.core import Composition

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

from core import llm_tool


MP_API_KEY = os.getenv("MP_API_KEY", "")


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _comp(formula: str) -> Composition:
    return Composition(str(formula).strip())


def _reaction(reactants: List[str], products: List[str]) -> Reaction:
    return Reaction([_comp(item) for item in reactants], [_comp(item) for item in products])


def _mp_ready() -> bool:
    return bool(MP_API_KEY and MPRester is not None)


def _get_mp_entries(chemsys: List[str]):
    if not _mp_ready():
        raise RuntimeError("MP_API_KEY is not set or mp-api is unavailable")
    with MPRester(MP_API_KEY) as mpr:
        return mpr.get_entries_in_chemsys(
            sorted(set(chemsys)),
            additional_criteria={"thermo_types": ["GGA_GGA+U"]},
        )


def _entry_energy_map(entries) -> Dict[str, float]:
    best: Dict[str, float] = {}
    for entry in entries:
        formula = entry.composition.reduced_formula
        epa = float(entry.energy_per_atom)
        if formula not in best or epa < best[formula]:
            best[formula] = epa
    return best


def _reaction_energy_from_map(reaction: Reaction, energies: Dict[str, float]) -> Tuple[float | None, List[str]]:
    missing: List[str] = []
    total = 0.0
    for comp, coeff in zip(reaction.all_comp, reaction.coeffs):
        formula = comp.reduced_formula
        if formula not in energies:
            missing.append(formula)
            continue
        total += float(coeff) * energies[formula] * comp.num_atoms
    if missing:
        return None, sorted(set(missing))
    product_atoms = sum(abs(float(coeff)) * comp.num_atoms for comp, coeff in zip(reaction.all_comp, reaction.coeffs) if coeff < 0)
    return float(total / max(product_atoms, 1e-8)), []


@llm_tool(
    name="balance_reaction_thermo",
    description="Balance a reaction from precursor formulas to target and optional byproducts using pymatgen Reaction.",
)
async def balance_reaction_thermo(
    reactant_formulas: List[str],
    product_formulas: List[str],
) -> str:
    reaction = _reaction(reactant_formulas, product_formulas)
    payload = {
        "ok": True,
        "reactants": [_comp(item).reduced_formula for item in reactant_formulas],
        "products": [_comp(item).reduced_formula for item in product_formulas],
        "balanced": True,
        "reaction": str(reaction),
        "coefficients": [float(x) for x in reaction.coeffs],
        "normalized_repr": reaction.normalized_repr,
    }
    return _json_text(payload)


@llm_tool(
    name="calculate_reaction_energy_mp",
    description="Calculate approximate reaction energy per product atom using MP GGA/GGA+U entries for all compounds in a balanced reaction.",
)
async def calculate_reaction_energy_mp(
    reactant_formulas: List[str],
    product_formulas: List[str],
) -> str:
    reaction = _reaction(reactant_formulas, product_formulas)
    elements = sorted({el.symbol for formula in reactant_formulas + product_formulas for el in _comp(formula).elements})
    payload: Dict[str, Any] = {
        "ok": True,
        "reaction": str(reaction),
        "chemical_system": "-".join(elements),
        "reaction_energy_per_product_atom_eV": None,
        "missing_formulas": [],
        "errors": [],
        "notes": ["Uses lowest MP entry energy per reduced formula; use as route ranking evidence, not final DFT truth."],
    }
    try:
        entries = _get_mp_entries(elements)
        energies = _entry_energy_map(entries)
        energy, missing = _reaction_energy_from_map(reaction, energies)
        payload["reaction_energy_per_product_atom_eV"] = energy
        payload["missing_formulas"] = missing
        payload["entry_energy_per_atom_eV"] = {key: energies[key] for key in sorted(energies) if key in {c.reduced_formula for c in reaction.all_comp}}
        payload["reward_components"] = {
            "reaction_energy_score": float(1.0 / (1.0 + np.exp(float(energy or 0.0)))) if energy is not None else 0.0,
            "mp_energy_coverage": 1.0 if not missing else 0.0,
        }
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="rank_precursor_reactions_thermo",
    description="Enumerate common gas byproducts, balance candidate precursor routes, and rank by MP reaction energy when available.",
)
async def rank_precursor_reactions_thermo(
    target_formula: str,
    candidate_precursor_sets: List[List[str]],
    byproducts: List[str] = None,
    top_k: int = 10,
) -> str:
    byproducts = byproducts or ["CO2", "H2O", "O2", "N2", "NO2"]
    rows = []
    for idx, precursors in enumerate(candidate_precursor_sets):
        best_row: Dict[str, Any] | None = None
        for n_byproducts in range(0, min(3, len(byproducts)) + 1):
            for subset in itertools.combinations(byproducts, n_byproducts):
                products = [target_formula] + list(subset)
                try:
                    reaction = _reaction(precursors, products)
                except Exception:
                    continue
                row = {
                    "candidate_index": idx,
                    "precursors": [_comp(item).reduced_formula for item in precursors],
                    "products": [_comp(item).reduced_formula for item in products],
                    "reaction": str(reaction),
                    "reaction_energy_per_product_atom_eV": None,
                    "missing_formulas": [],
                    "score": 0.5,
                }
                try:
                    elements = sorted({el.symbol for formula in precursors + products for el in _comp(formula).elements})
                    entries = _get_mp_entries(elements)
                    energy, missing = _reaction_energy_from_map(reaction, _entry_energy_map(entries))
                    row["reaction_energy_per_product_atom_eV"] = energy
                    row["missing_formulas"] = missing
                    row["score"] = float(1.0 / (1.0 + np.exp(float(energy or 0.0)))) if energy is not None else 0.2
                except Exception as exc:
                    row["mp_error"] = str(exc)
                if best_row is None or row["score"] > best_row["score"]:
                    best_row = row
        if best_row is not None:
            rows.append(best_row)
    rows.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "ok": True,
        "target_formula": _comp(target_formula).reduced_formula,
        "ranked_routes": rows[: max(1, int(top_k))],
        "reward_components": {
            "best_reaction_route_score": rows[0]["score"] if rows else 0.0,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="analyze_phase_stability_mp",
    description="Compute MP phase diagram stability for a formula when MP entries are available.",
)
async def analyze_phase_stability_mp(formula: str) -> str:
    comp = _comp(formula)
    elements = [el.symbol for el in comp.elements]
    payload: Dict[str, Any] = {
        "ok": True,
        "formula": comp.reduced_formula,
        "chemical_system": "-".join(elements),
        "stable_entries": [],
        "target_matches": [],
        "lowest_energy_above_hull_eV_atom": None,
        "errors": [],
    }
    try:
        entries = _get_mp_entries(elements)
        pd = PhaseDiagram(entries)
        stable = []
        matches = []
        for entry in entries:
            e_hull = float(pd.get_e_above_hull(entry))
            row = {
                "formula": entry.composition.reduced_formula,
                "entry_id": getattr(entry, "entry_id", None),
                "energy_above_hull_eV_atom": e_hull,
                "energy_per_atom_eV": float(entry.energy_per_atom),
            }
            if e_hull <= 1e-8:
                stable.append(row)
            if entry.composition.reduced_formula == comp.reduced_formula:
                matches.append(row)
        matches.sort(key=lambda item: item["energy_above_hull_eV_atom"])
        payload["stable_entries"] = stable[:64]
        payload["target_matches"] = matches[:16]
        payload["lowest_energy_above_hull_eV_atom"] = matches[0]["energy_above_hull_eV_atom"] if matches else None
        payload["reward_components"] = {
            "mp_known_stability": float(max(0.0, 1.0 - (matches[0]["energy_above_hull_eV_atom"] / 0.2))) if matches else 0.0,
            "mp_formula_known": 1.0 if matches else 0.0,
        }
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="analyze_open_system_stability_mp",
    description="Analyze grand-potential phase stability under an open element chemical potential using MP entries.",
)
async def analyze_open_system_stability_mp(
    formula: str,
    open_element: str = "O",
    chemical_potential_eV: float = -1.0,
) -> str:
    comp = _comp(formula)
    elements = sorted({el.symbol for el in comp.elements} | {open_element})
    payload: Dict[str, Any] = {
        "ok": True,
        "formula": comp.reduced_formula,
        "open_element": open_element,
        "chemical_potential_eV": float(chemical_potential_eV),
        "stable_entries": [],
        "target_matches": [],
        "errors": [],
    }
    try:
        entries = _get_mp_entries(elements)
        open_el = Composition(open_element).elements[0]
        gpd = GrandPotentialPhaseDiagram(entries, {open_el: float(chemical_potential_eV)})
        rows = []
        matches = []
        for entry in entries:
            try:
                transformed = gpd.pd_entry(entry)
                e_hull = float(gpd.get_e_above_hull(transformed))
            except Exception:
                continue
            row = {
                "formula": entry.composition.reduced_formula,
                "entry_id": getattr(entry, "entry_id", None),
                "grand_potential_e_above_hull": e_hull,
            }
            if e_hull <= 1e-8:
                rows.append(row)
            if entry.composition.reduced_formula == comp.reduced_formula:
                matches.append(row)
        matches.sort(key=lambda item: item["grand_potential_e_above_hull"])
        payload["stable_entries"] = rows[:64]
        payload["target_matches"] = matches[:16]
        payload["reward_components"] = {
            "open_system_stability": float(max(0.0, 1.0 - (matches[0]["grand_potential_e_above_hull"] / 0.2))) if matches else 0.0,
        }
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)
