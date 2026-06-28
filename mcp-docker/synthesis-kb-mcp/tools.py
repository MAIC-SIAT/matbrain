import csv
import json
import os
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np
from pymatgen.analysis.reaction_calculator import BalancedReaction, Reaction
from pymatgen.core import Composition

from core import llm_tool


KB_PATH = os.getenv("SYNTHESIS_KB_PATH", "/data/synthesis_recipes.jsonl")

COMMON_PRECURSORS = {
    "Li": ["Li2CO3", "LiOH", "LiNO3"],
    "Na": ["Na2CO3", "NaNO3", "NaOH"],
    "K": ["K2CO3", "KNO3", "KOH"],
    "Mg": ["MgO", "MgCO3", "Mg(NO3)2"],
    "Ca": ["CaCO3", "CaO", "Ca(NO3)2"],
    "Sr": ["SrCO3", "SrO", "Sr(NO3)2"],
    "Ba": ["BaCO3", "BaO", "Ba(NO3)2"],
    "Al": ["Al2O3", "Al(NO3)3", "Al(OH)3"],
    "Ti": ["TiO2", "TiCl4", "Ti(OC3H7)4"],
    "V": ["V2O5", "NH4VO3"],
    "Cr": ["Cr2O3", "Cr(NO3)3"],
    "Mn": ["MnO2", "Mn2O3", "MnCO3", "Mn(NO3)2"],
    "Fe": ["Fe2O3", "Fe3O4", "Fe(NO3)3", "FeC2O4"],
    "Co": ["Co3O4", "CoO", "Co(NO3)2"],
    "Ni": ["NiO", "Ni(NO3)2"],
    "Cu": ["CuO", "Cu(NO3)2", "Cu2O"],
    "Zn": ["ZnO", "Zn(NO3)2"],
    "Zr": ["ZrO2", "ZrOCl2"],
    "Nb": ["Nb2O5"],
    "Mo": ["MoO3", "(NH4)6Mo7O24"],
    "W": ["WO3", "(NH4)10W12O41"],
    "La": ["La2O3", "La(NO3)3"],
    "Ce": ["CeO2", "Ce(NO3)3"],
    "Pr": ["Pr6O11", "Pr(NO3)3"],
    "Nd": ["Nd2O3", "Nd(NO3)3"],
    "Sm": ["Sm2O3", "Sm(NO3)3"],
    "Gd": ["Gd2O3", "Gd(NO3)3"],
    "Y": ["Y2O3", "Y(NO3)3"],
    "O": ["O2"],
    "S": ["S", "H2S", "CS2"],
    "N": ["N2", "NH3"],
    "P": ["NH4H2PO4", "(NH4)2HPO4", "P2O5"],
    "F": ["NH4F", "LiF", "NaF"],
    "Cl": ["NH4Cl", "NaCl", "KCl"],
}


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _safe_comp(formula: str) -> Composition:
    return Composition(str(formula).strip())


def _load_kb() -> List[Dict[str, Any]]:
    if not os.path.exists(KB_PATH):
        return []
    rows: List[Dict[str, Any]] = []
    if KB_PATH.endswith(".jsonl"):
        with open(KB_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    elif KB_PATH.endswith(".json"):
        with open(KB_PATH, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        rows = data if isinstance(data, list) else data.get("recipes", [])
    elif KB_PATH.endswith(".csv"):
        with open(KB_PATH, "r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    return rows


def _recipe_target(recipe: Dict[str, Any]) -> str:
    return str(recipe.get("target_formula") or recipe.get("target") or recipe.get("product") or "")


def _recipe_precursors(recipe: Dict[str, Any]) -> List[str]:
    raw = recipe.get("precursors") or recipe.get("precursor_formulas") or recipe.get("reactants") or []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                raw = parsed
            else:
                raw = re.split(r"[,;+]", raw)
        except Exception:
            raw = re.split(r"[,;+]", raw)
    return [str(item).strip() for item in raw if str(item).strip()]


def _recipe_temperature(recipe: Dict[str, Any]) -> float | None:
    for key in ["temperature_c", "calcination_temperature_c", "sintering_temperature_c", "temperature"]:
        value = recipe.get(key)
        if value in (None, ""):
            continue
        match = re.search(r"[-+]?\d+(\.\d+)?", str(value))
        if match:
            return float(match.group(0))
    return None


def _composition_similarity(a: Composition, b: Composition) -> float:
    elems = sorted({el.symbol for el in a.elements} | {el.symbol for el in b.elements})
    l1 = sum(abs(float(a.fractional_composition.get_atomic_fraction(el)) - float(b.fractional_composition.get_atomic_fraction(el))) for el in elems)
    return float(max(0.0, 1.0 - l1 / 2.0))


def _target_elements(formula: str) -> List[str]:
    return [el.symbol for el in _safe_comp(formula).elements]


def _common_precursor_candidates(target_formula: str) -> Dict[str, List[str]]:
    result: Dict[str, List[str]] = {}
    for symbol in _target_elements(target_formula):
        if symbol in {"O", "H", "C"}:
            continue
        result[symbol] = COMMON_PRECURSORS.get(symbol, [symbol])
    return result


def _route_score(target_formula: str, precursors: List[str], temperature_c: float | None = None, atmosphere: str = "air") -> Dict[str, Any]:
    target = _safe_comp(target_formula)
    precursor_comps = [_safe_comp(item) for item in precursors]
    target_elements = {el.symbol for el in target.elements}
    precursor_elements = {el.symbol for comp in precursor_comps for el in comp.elements}
    nonvolatile = {el for el in target_elements if el not in {"O", "H", "C", "N"}}
    coverage = len(nonvolatile & precursor_elements) / max(len(nonvolatile), 1)
    oxygen_target = "O" in target_elements
    oxygen_sources = sum(1 for comp in precursor_comps if "O" in {el.symbol for el in comp.elements})
    oxygen_source_score = 1.0 if not oxygen_target or oxygen_sources > 0 else 0.0
    temp_score = 1.0
    warnings: List[str] = []
    if temperature_c is not None:
        if float(temperature_c) < 100 or float(temperature_c) > 1800:
            temp_score = 0.4
            warnings.append("temperature outside broad inorganic synthesis range")
        elif 500 <= float(temperature_c) <= 1300:
            temp_score = 1.0
        else:
            temp_score = 0.8
    precursor_count_score = 1.0 if 1 <= len(precursors) <= 6 else 0.6
    atmosphere_score = 1.0
    if oxygen_target and "h2" in atmosphere.lower():
        atmosphere_score = 0.5
        warnings.append("strongly reducing atmosphere may be inconsistent with oxide target")
    score = 0.45 * coverage + 0.20 * oxygen_source_score + 0.15 * temp_score + 0.10 * precursor_count_score + 0.10 * atmosphere_score
    return {
        "score": float(score),
        "element_coverage": float(coverage),
        "oxygen_source_score": float(oxygen_source_score),
        "temperature_score": float(temp_score),
        "precursor_count_score": float(precursor_count_score),
        "atmosphere_score": float(atmosphere_score),
        "warnings": warnings,
    }


@llm_tool(
    name="search_synthesis_recipes_kb",
    description="Search local text-mined synthesis recipe KB by target formula or similar composition.",
)
async def search_synthesis_recipes_kb(target_formula: str, top_k: int = 10, min_similarity: float = 0.2) -> str:
    target = _safe_comp(target_formula)
    kb = _load_kb()
    rows = []
    for recipe in kb:
        formula = _recipe_target(recipe)
        if not formula:
            continue
        try:
            sim = _composition_similarity(target, _safe_comp(formula))
        except Exception:
            continue
        if sim >= float(min_similarity):
            rows.append({
                "similarity": sim,
                "target_formula": formula,
                "precursors": _recipe_precursors(recipe),
                "temperature_c": _recipe_temperature(recipe),
                "atmosphere": recipe.get("atmosphere") or recipe.get("environment"),
                "method": recipe.get("method") or recipe.get("synthesis_type"),
                "doi": recipe.get("doi"),
                "source": recipe.get("source"),
            })
    rows.sort(key=lambda item: item["similarity"], reverse=True)
    payload = {
        "ok": True,
        "kb_path": KB_PATH,
        "kb_available": bool(kb),
        "target_formula": target.reduced_formula,
        "results": rows[: max(1, int(top_k))],
        "notes": [] if kb else ["No local synthesis KB found. Set SYNTHESIS_KB_PATH to a JSONL/JSON/CSV recipe file."],
    }
    return _json_text(payload)


@llm_tool(
    name="recommend_precursors_kb",
    description="Recommend precursor sets from local synthesis precedents plus common inorganic precursor heuristics.",
)
async def recommend_precursors_kb(target_formula: str, top_k: int = 8, method: str = "solid-state") -> str:
    target = _safe_comp(target_formula)
    kb = _load_kb()
    precursor_counter: Dict[str, Counter] = defaultdict(Counter)
    recipe_hits = []
    for recipe in kb:
        formula = _recipe_target(recipe)
        if not formula:
            continue
        try:
            sim = _composition_similarity(target, _safe_comp(formula))
        except Exception:
            continue
        if sim < 0.25:
            continue
        recipe_hits.append({"formula": formula, "similarity": sim, "precursors": _recipe_precursors(recipe)})
        for precursor in _recipe_precursors(recipe):
            try:
                comp = _safe_comp(precursor)
            except Exception:
                continue
            for el in comp.elements:
                if el.symbol in _target_elements(target_formula):
                    precursor_counter[el.symbol][comp.reduced_formula] += sim

    candidates = _common_precursor_candidates(target_formula)
    for symbol, counter in precursor_counter.items():
        ranked = [item for item, _ in counter.most_common(8)]
        candidates[symbol] = list(dict.fromkeys(ranked + candidates.get(symbol, [])))[:8]

    route = []
    for symbol in _target_elements(target_formula):
        if symbol in {"O", "H", "C"}:
            continue
        options = candidates.get(symbol, [symbol])
        route.append(options[0])
    route_score = _route_score(target_formula, route, None, "air")
    payload = {
        "ok": True,
        "target_formula": target.reduced_formula,
        "method": method,
        "kb_available": bool(kb),
        "candidate_precursors_by_element": candidates,
        "recommended_precursor_set": route,
        "recommended_set_score": route_score,
        "similar_recipe_count": len(recipe_hits),
        "similar_recipe_examples": sorted(recipe_hits, key=lambda item: item["similarity"], reverse=True)[: max(1, int(top_k))],
    }
    return _json_text(payload)


@llm_tool(
    name="recommend_synthesis_conditions_kb",
    description="Recommend temperature range and atmosphere from local precedents and formula heuristics.",
)
async def recommend_synthesis_conditions_kb(target_formula: str, method: str = "solid-state") -> str:
    target = _safe_comp(target_formula)
    kb = _load_kb()
    temps = []
    atmospheres = []
    for recipe in kb:
        formula = _recipe_target(recipe)
        if not formula:
            continue
        try:
            sim = _composition_similarity(target, _safe_comp(formula))
        except Exception:
            continue
        if sim < 0.35:
            continue
        temp = _recipe_temperature(recipe)
        if temp is not None:
            temps.append((sim, temp))
        atmosphere = recipe.get("atmosphere") or recipe.get("environment")
        if atmosphere:
            atmospheres.append((sim, str(atmosphere)))

    if temps:
        temp_values = np.array([temp for _, temp in temps], dtype=float)
        recommended_range = [float(np.percentile(temp_values, 25)), float(np.percentile(temp_values, 75))]
        recommended_temperature = float(np.median(temp_values))
    elif method.lower() in {"solid-state", "ceramic", "calcination"}:
        recommended_range = [700.0, 1100.0]
        recommended_temperature = 900.0
    else:
        recommended_range = [150.0, 500.0]
        recommended_temperature = 300.0

    atmosphere = "air"
    if atmospheres:
        counter = Counter()
        for sim, item in atmospheres:
            counter[item.lower()] += sim
        atmosphere = counter.most_common(1)[0][0]
    elif "O" not in {el.symbol for el in target.elements}:
        atmosphere = "Ar or N2"

    payload = {
        "ok": True,
        "target_formula": target.reduced_formula,
        "method": method,
        "kb_available": bool(kb),
        "recommended_temperature_c": recommended_temperature,
        "recommended_temperature_range_c": recommended_range,
        "recommended_atmosphere": atmosphere,
        "precedent_temperature_count": len(temps),
        "precedent_atmosphere_count": len(atmospheres),
    }
    return _json_text(payload)


@llm_tool(
    name="rank_synthesis_routes_kb",
    description="Rank candidate synthesis routes using element coverage, precursor plausibility, temperature, and atmosphere heuristics.",
)
async def rank_synthesis_routes_kb(target_formula: str, candidate_routes: List[Dict[str, Any]], top_k: int = 10) -> str:
    rows = []
    for idx, route in enumerate(candidate_routes):
        precursors = route.get("precursors") or route.get("precursor_formulas") or []
        if isinstance(precursors, str):
            precursors = re.split(r"[,;+]", precursors)
        temperature = route.get("temperature_c")
        atmosphere = route.get("atmosphere") or "air"
        score = _route_score(target_formula, [str(item).strip() for item in precursors if str(item).strip()], temperature, atmosphere)
        rows.append({
            "index": idx,
            "precursors": precursors,
            "temperature_c": temperature,
            "atmosphere": atmosphere,
            **score,
        })
    rows.sort(key=lambda item: item["score"], reverse=True)
    payload = {
        "ok": True,
        "target_formula": _safe_comp(target_formula).reduced_formula,
        "ranked_routes": rows[: max(1, int(top_k))],
        "reward_components": {
            "best_route_score": rows[0]["score"] if rows else 0.0,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="balance_precursor_reaction_kb",
    description="Attempt to balance precursor formulas into a target formula plus common gas byproducts.",
)
async def balance_precursor_reaction_kb(target_formula: str, precursor_formulas: List[str], byproducts: List[str] = None) -> str:
    byproducts = byproducts or ["CO2", "H2O", "O2", "N2", "NO2"]
    reactants = [_safe_comp(item) for item in precursor_formulas]
    products = [_safe_comp(target_formula)] + [_safe_comp(item) for item in byproducts]
    payload: Dict[str, Any] = {
        "ok": True,
        "target_formula": _safe_comp(target_formula).reduced_formula,
        "precursor_formulas": [comp.reduced_formula for comp in reactants],
        "byproducts_considered": byproducts,
        "balanced": False,
        "reaction": None,
        "errors": [],
    }
    try:
        reaction = Reaction(reactants, products)
        payload["balanced"] = True
        payload["reaction"] = str(reaction)
        payload["coefficients"] = [float(x) for x in reaction.coeffs]
    except Exception as exc:
        payload["errors"].append(str(exc))
        try:
            coeffs = {comp: 1.0 for comp in reactants}
            coeffs[_safe_comp(target_formula)] = -1.0
            balanced = BalancedReaction(coeffs)
            payload["balanced"] = True
            payload["reaction"] = str(balanced)
        except Exception:
            pass
    return _json_text(payload)


@llm_tool(
    name="check_precursor_compatibility_kb",
    description="Check precursor set compatibility for target element coverage, duplicate roles, and obvious route risks.",
)
async def check_precursor_compatibility_kb(target_formula: str, precursor_formulas: List[str], atmosphere: str = "air") -> str:
    score = _route_score(target_formula, precursor_formulas, None, atmosphere)
    target_elements = set(_target_elements(target_formula))
    precursor_elements = {el.symbol for formula in precursor_formulas for el in _safe_comp(formula).elements}
    payload = {
        "ok": True,
        "target_formula": _safe_comp(target_formula).reduced_formula,
        "precursor_formulas": [_safe_comp(item).reduced_formula for item in precursor_formulas],
        "target_elements": sorted(target_elements),
        "precursor_elements": sorted(precursor_elements),
        "missing_elements": sorted((target_elements - {"O", "H", "C", "N"}) - precursor_elements),
        "extra_elements": sorted(precursor_elements - target_elements - {"O", "H", "C", "N"}),
        "score": score["score"],
        "warnings": score["warnings"],
        "reward_components": {
            "precursor_compatibility": score["score"],
            "element_coverage": score["element_coverage"],
        },
    }
    return _json_text(payload)
