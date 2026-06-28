import itertools
import json
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.core import Composition, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from smact import Element as SmactElement
    from smact.screening import pauling_test
except Exception:
    SmactElement = None
    pauling_test = None

from core import llm_tool


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _load_structure(cif_string: str) -> Structure:
    return Structure.from_str(cif_string, fmt="cif")


def _finite(value: Any) -> Any:
    try:
        val = float(value)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return value


def _score_from_penalties(base: float, penalties: List[float], bonuses: List[float] = None) -> float:
    value = base - sum(penalties) + sum(bonuses or [])
    return float(max(0.0, min(1.0, value)))


def _min_distance(structure: Structure) -> Tuple[float, Tuple[int, int]]:
    if len(structure) < 2:
        return 999.0, (-1, -1)
    matrix = np.array(structure.distance_matrix, dtype=float)
    np.fill_diagonal(matrix, np.inf)
    flat_index = int(np.argmin(matrix))
    i, j = np.unravel_index(flat_index, matrix.shape)
    return float(matrix[i, j]), (int(i), int(j))


def _smact_formula_score(comp: Composition, max_combinations: int = 4096) -> Dict[str, Any]:
    if SmactElement is None:
        return {
            "available": False,
            "score": 0.0,
            "neutral_assignment_count": 0,
            "pauling_pass_count": 0,
            "examples": [],
        }
    symbols = [el.symbol for el in comp.elements]
    amounts = [float(comp[el]) for el in comp.elements]
    elements = [SmactElement(symbol) for symbol in symbols]
    oxidation_lists = [sorted(set(int(x) for x in element.oxidation_states)) for element in elements]
    enegs = [float(element.pauling_eneg or 0.0) for element in elements]

    neutral_count = 0
    pauling_count = 0
    checked = 0
    examples = []
    for ox_states in itertools.product(*oxidation_lists):
        checked += 1
        charge = sum(amount * int(oxi) for amount, oxi in zip(amounts, ox_states))
        if abs(charge) >= 1e-8:
            if checked >= max_combinations:
                break
            continue
        neutral_count += 1
        passed = bool(pauling_test and pauling_test(ox_states, enegs, symbols=symbols))
        if passed:
            pauling_count += 1
        if len(examples) < 12:
            examples.append({"oxidation_states": dict(zip(symbols, ox_states)), "pauling_passed": passed})
        if checked >= max_combinations:
            break

    score = 0.0
    if neutral_count:
        score += 0.6
    if pauling_count:
        score += 0.4
    return {
        "available": True,
        "score": float(score),
        "checked_combinations": checked,
        "neutral_assignment_count": neutral_count,
        "pauling_pass_count": pauling_count,
        "examples": examples,
    }


@llm_tool(
    name="validate_formula_verifier",
    description="Validate a chemical formula with charge-balance and SMACT Pauling hard constraints; returns reward-ready JSON.",
)
async def validate_formula_verifier(formula: str) -> str:
    comp = Composition(formula)
    errors: List[str] = []
    warnings: List[str] = []
    penalties: List[float] = []

    if not comp.elements:
        errors.append("empty composition")
        penalties.append(1.0)
    if comp.num_atoms <= 0:
        errors.append("non-positive atom count")
        penalties.append(1.0)
    if len(comp.elements) > 8:
        warnings.append("large chemical system")
        penalties.append(0.05)

    oxi_guesses = []
    try:
        oxi_guesses = comp.oxi_state_guesses(max_sites=64)
    except Exception as exc:
        warnings.append(f"pymatgen oxidation-state guessing failed: {exc}")

    smact = _smact_formula_score(comp)
    if not oxi_guesses and smact.get("neutral_assignment_count", 0) == 0:
        errors.append("no charge-neutral oxidation-state assignment found")
        penalties.append(0.45)
    if smact.get("available") and smact.get("pauling_pass_count", 0) == 0:
        warnings.append("no SMACT Pauling-valid assignment found")
        penalties.append(0.20)

    score = _score_from_penalties(1.0, penalties)
    payload = {
        "ok": True,
        "formula": comp.reduced_formula,
        "anonymous_formula": comp.anonymized_formula,
        "elements": [el.symbol for el in comp.elements],
        "num_atoms_reduced": float(comp.num_atoms),
        "valid": bool(score >= 0.65 and not errors),
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "oxidation_state_guesses": oxi_guesses[:12],
        "smact": smact,
        "reward_components": {
            "formula_validity": score,
            "charge_balance_penalty": -0.45 if "no charge-neutral oxidation-state assignment found" in errors else 0.0,
            "pauling_penalty": -0.20 if smact.get("available") and smact.get("pauling_pass_count", 0) == 0 else 0.0,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="validate_structure_verifier",
    description="Validate CIF geometry, composition, symmetry, density, and minimum interatomic distance; returns reward-ready JSON.",
)
async def validate_structure_verifier(
    cif_string: str,
    min_distance_angstrom: float = 0.7,
    min_volume_per_atom: float = 3.0,
    max_volume_per_atom: float = 200.0,
    symprec: float = 0.1,
) -> str:
    structure = _load_structure(cif_string)
    errors: List[str] = []
    warnings: List[str] = []
    penalties: List[float] = []
    bonuses: List[float] = []

    min_dist, pair = _min_distance(structure)
    volume_per_atom = float(structure.volume / max(len(structure), 1))
    density = float(structure.density)
    if len(structure) == 0:
        errors.append("structure has zero sites")
        penalties.append(1.0)
    if min_dist < float(min_distance_angstrom):
        errors.append(f"minimum interatomic distance {min_dist:.3f} A is below threshold")
        penalties.append(min(0.5, (float(min_distance_angstrom) - min_dist) / max(float(min_distance_angstrom), 1e-6)))
    if volume_per_atom < float(min_volume_per_atom):
        errors.append(f"volume per atom {volume_per_atom:.3f} A^3 is too small")
        penalties.append(0.35)
    if volume_per_atom > float(max_volume_per_atom):
        errors.append(f"volume per atom {volume_per_atom:.3f} A^3 is too large")
        penalties.append(0.35)
    if density <= 0:
        errors.append("non-positive density")
        penalties.append(0.5)

    sga_payload: Dict[str, Any] = {"available": False}
    try:
        sga = SpacegroupAnalyzer(structure, symprec=float(symprec))
        sga_payload = {
            "available": True,
            "space_group_number": int(sga.get_space_group_number()),
            "space_group_symbol": sga.get_space_group_symbol(),
            "crystal_system": sga.get_crystal_system(),
        }
        bonuses.append(0.05)
    except Exception as exc:
        warnings.append(f"space-group analysis failed: {exc}")
        penalties.append(0.05)

    bv_payload: Dict[str, Any] = {"available": False}
    try:
        analyzer = BVAnalyzer()
        oxi_structure = analyzer.get_oxi_state_decorated_structure(structure)
        bv_payload = {
            "available": True,
            "site_valences": [_finite(site.specie.oxi_state) for site in oxi_structure],
        }
        bonuses.append(0.05)
    except Exception as exc:
        warnings.append(f"bond-valence oxidation assignment failed: {exc}")

    score = _score_from_penalties(1.0, penalties, bonuses)
    payload = {
        "ok": True,
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "valid": bool(score >= 0.70 and not errors),
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "geometry": {
            "min_distance_angstrom": min_dist,
            "min_distance_site_pair": pair,
            "volume": float(structure.volume),
            "volume_per_atom": volume_per_atom,
            "density_g_cm3": density,
            "lattice_abc": [_finite(x) for x in structure.lattice.abc],
            "lattice_angles": [_finite(x) for x in structure.lattice.angles],
        },
        "symmetry": sga_payload,
        "bond_valence": bv_payload,
        "reward_components": {
            "structure_validity": score,
            "distance_penalty": -1.0 if min_dist < float(min_distance_angstrom) else 0.0,
            "volume_penalty": -0.35 if volume_per_atom < float(min_volume_per_atom) or volume_per_atom > float(max_volume_per_atom) else 0.0,
            "symmetry_bonus": 0.05 if sga_payload.get("available") else 0.0,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="score_candidate_structure_verifier",
    description="Score a candidate CIF against target formula and optional space group using hard validity constraints.",
)
async def score_candidate_structure_verifier(
    cif_string: str,
    target_formula: str = "",
    expected_space_group: int = 0,
    min_distance_angstrom: float = 0.7,
    symprec: float = 0.1,
) -> str:
    structure = _load_structure(cif_string)
    formula_score = json.loads(await validate_formula_verifier(structure.composition.reduced_formula))
    structure_score = json.loads(await validate_structure_verifier(cif_string, min_distance_angstrom, 3.0, 200.0, symprec))

    errors: List[str] = []
    warnings: List[str] = []
    rewards = {
        "formula_validity": float(formula_score.get("score", 0.0)),
        "structure_validity": float(structure_score.get("score", 0.0)),
        "target_formula_match": 0.0,
        "space_group_match": 0.0,
    }

    if target_formula:
        target_comp = Composition(target_formula).fractional_composition
        actual_comp = structure.composition.fractional_composition
        elements = sorted({el.symbol for el in target_comp.elements} | {el.symbol for el in actual_comp.elements})
        comp_l1 = sum(abs(float(target_comp.get_atomic_fraction(el)) - float(actual_comp.get_atomic_fraction(el))) for el in elements)
        rewards["target_formula_match"] = float(max(0.0, 1.0 - comp_l1))
        if comp_l1 > 1e-6:
            errors.append(f"composition differs from target formula; fractional L1={comp_l1:.4f}")
    else:
        comp_l1 = None

    observed_sg = None
    try:
        observed_sg = int(SpacegroupAnalyzer(structure, symprec=float(symprec)).get_space_group_number())
    except Exception as exc:
        warnings.append(f"space-group check failed: {exc}")
    if int(expected_space_group or 0) > 0:
        rewards["space_group_match"] = 1.0 if observed_sg == int(expected_space_group) else 0.0
        if observed_sg != int(expected_space_group):
            warnings.append(f"observed space group {observed_sg} != expected {expected_space_group}")

    weights = {
        "formula_validity": 0.25,
        "structure_validity": 0.45,
        "target_formula_match": 0.20 if target_formula else 0.0,
        "space_group_match": 0.10 if int(expected_space_group or 0) > 0 else 0.0,
    }
    weight_sum = sum(weights.values()) or 1.0
    total_score = sum(rewards[key] * weight for key, weight in weights.items()) / weight_sum
    payload = {
        "ok": True,
        "formula": structure.composition.reduced_formula,
        "target_formula": target_formula,
        "valid": bool(total_score >= 0.70 and not errors and structure_score.get("valid", False)),
        "score": float(total_score),
        "errors": errors + structure_score.get("errors", []),
        "warnings": warnings + formula_score.get("warnings", []) + structure_score.get("warnings", []),
        "observed_space_group": observed_sg,
        "composition_fractional_l1": comp_l1,
        "subscores": rewards,
        "weights": weights,
        "reward_components": {
            "candidate_structure_score": float(total_score),
            **{f"{key}_reward": value for key, value in rewards.items()},
        },
    }
    return _json_text(payload)


@llm_tool(
    name="validate_synthesis_route_verifier",
    description="Hard-check a synthesis route for precursor coverage, balanced elements, temperature range, and atmosphere consistency.",
)
async def validate_synthesis_route_verifier(
    target_formula: str,
    precursor_formulas: List[str],
    temperature_c: float = 800.0,
    atmosphere: str = "air",
) -> str:
    target = Composition(target_formula)
    precursors = [Composition(item) for item in precursor_formulas]
    target_elements = {el.symbol for el in target.elements}
    precursor_elements = {el.symbol for comp in precursors for el in comp.elements}
    nonvolatile = {el for el in target_elements if el not in {"O", "H", "C", "N"}}
    covered = nonvolatile <= precursor_elements

    errors: List[str] = []
    warnings: List[str] = []
    penalties: List[float] = []
    if not covered:
        missing = sorted(nonvolatile - precursor_elements)
        errors.append(f"missing target nonvolatile elements in precursors: {missing}")
        penalties.append(0.40)
    if float(temperature_c) < 100 or float(temperature_c) > 1800:
        warnings.append("temperature outside common inorganic synthesis range")
        penalties.append(0.15)
    if "oxide" in target.reduced_formula.lower() and "h2" in atmosphere.lower():
        warnings.append("reducing atmosphere may conflict with oxide synthesis target")
        penalties.append(0.10)
    if len(precursors) == 0:
        errors.append("no precursors provided")
        penalties.append(1.0)

    score = _score_from_penalties(1.0, penalties)
    payload = {
        "ok": True,
        "target_formula": target.reduced_formula,
        "precursor_formulas": [comp.reduced_formula for comp in precursors],
        "target_elements": sorted(target_elements),
        "precursor_elements": sorted(precursor_elements),
        "nonvolatile_target_elements_covered": covered,
        "temperature_c": float(temperature_c),
        "atmosphere": atmosphere,
        "valid": bool(score >= 0.70 and not errors),
        "score": score,
        "errors": errors,
        "warnings": warnings,
        "reward_components": {
            "synthesis_route_validity": score,
            "element_coverage": 1.0 if covered else 0.0,
            "temperature_plausibility": 0.0 if float(temperature_c) < 100 or float(temperature_c) > 1800 else 1.0,
        },
    }
    return _json_text(payload)
