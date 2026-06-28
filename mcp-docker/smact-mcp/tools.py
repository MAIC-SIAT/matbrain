import itertools
import json
from typing import Any, Dict, List

from pymatgen.core import Composition
from smact import Element
from smact.screening import pauling_test

from core import llm_tool


def _json_block(payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _composition_elements(formula: str) -> List[str]:
    comp = Composition(formula)
    return [el.symbol for el in comp.elements]


def _smact_element(symbol: str) -> Element:
    return Element(symbol)


def _pauling_enegs(elements: List[Element]) -> List[float]:
    return [float(element.pauling_eneg or 0.0) for element in elements]


@llm_tool(
    name="screen_smact_oxidation_states",
    description="Enumerate plausible SMACT oxidation-state combinations for a formula without database lookup.",
)
async def screen_smact_oxidation_states(formula: str, max_combinations: int = 64) -> str:
    max_combinations = max(1, min(int(max_combinations), 512))
    comp = Composition(formula)
    amounts = [int(v) if float(v).is_integer() else float(v) for v in comp.get_el_amt_dict().values()]
    symbols = _composition_elements(formula)
    oxidation_lists = []
    for symbol in symbols:
        oxi_states = sorted(set(int(x) for x in _smact_element(symbol).oxidation_states))
        oxidation_lists.append(oxi_states)

    charge_neutral = []
    checked = 0
    for ox_states in itertools.product(*oxidation_lists):
        checked += 1
        charge = sum(float(amount) * int(oxi) for amount, oxi in zip(amounts, ox_states))
        if abs(charge) < 1e-8:
            charge_neutral.append(dict(zip(symbols, ox_states)))
            if len(charge_neutral) >= max_combinations:
                break

    payload = {
        "formula": comp.reduced_formula,
        "elements": symbols,
        "stoichiometry": dict(zip(symbols, amounts)),
        "oxidation_state_options": dict(zip(symbols, oxidation_lists)),
        "checked_combinations": checked,
        "charge_neutral_count_returned": len(charge_neutral),
        "charge_neutral_combinations": charge_neutral,
    }
    return "# SMACT Oxidation-State Screen\n\n" + _json_block(payload)


@llm_tool(
    name="check_smact_charge_neutrality",
    description="Check whether a specified oxidation-state assignment is charge neutral for a formula.",
)
async def check_smact_charge_neutrality(formula: str, oxidation_states: Dict[str, int]) -> str:
    comp = Composition(formula)
    amounts = comp.get_el_amt_dict()
    missing = sorted(set(amounts) - set(oxidation_states))
    charge = sum(float(amounts[el]) * int(oxidation_states.get(el, 0)) for el in amounts)
    payload = {
        "formula": comp.reduced_formula,
        "oxidation_states": oxidation_states,
        "missing_elements": missing,
        "total_charge": charge,
        "is_charge_neutral": bool(not missing and abs(charge) < 1e-8),
    }
    return "# SMACT Charge Neutrality\n\n" + _json_block(payload)


@llm_tool(
    name="check_smact_pauling_test",
    description="Run SMACT Pauling electronegativity screening for a formula and optional oxidation-state assignment.",
)
async def check_smact_pauling_test(formula: str, oxidation_states: Dict[str, int] = None) -> str:
    comp = Composition(formula)
    symbols = _composition_elements(formula)
    elements = [_smact_element(symbol) for symbol in symbols]
    enegs = _pauling_enegs(elements)
    if oxidation_states:
        ox_states = [int(oxidation_states[symbol]) for symbol in symbols]
        pauling_ok = bool(pauling_test(ox_states, enegs, symbols=symbols))
        tested = [dict(zip(symbols, ox_states))]
    else:
        tested = []
        pauling_ok = False
        oxidation_lists = [sorted(set(int(x) for x in element.oxidation_states)) for element in elements]
        for ox_states in itertools.product(*oxidation_lists):
            charge = sum(float(comp[el.symbol]) * int(oxi) for el, oxi in zip(comp.elements, ox_states))
            if abs(charge) >= 1e-8:
                continue
            tested.append(dict(zip(symbols, ox_states)))
            if pauling_test(ox_states, enegs, symbols=symbols):
                pauling_ok = True
                break

    payload = {
        "formula": comp.reduced_formula,
        "pauling_test_passed": pauling_ok,
        "tested_assignments": tested[:32],
        "num_tested_assignments": len(tested),
    }
    return "# SMACT Pauling Test\n\n" + _json_block(payload)


@llm_tool(
    name="score_composition_chemical_validity_smact",
    description="Score formula plausibility using charge-neutral oxidation states and SMACT Pauling screening.",
)
async def score_composition_chemical_validity_smact(formula: str) -> str:
    comp = Composition(formula)
    symbols = _composition_elements(formula)
    elements = [_smact_element(symbol) for symbol in symbols]
    enegs = _pauling_enegs(elements)
    oxidation_lists = [sorted(set(int(x) for x in element.oxidation_states)) for element in elements]
    neutral_count = 0
    pauling_count = 0
    examples = []
    for ox_states in itertools.product(*oxidation_lists):
        charge = sum(float(comp[el.symbol]) * int(oxi) for el, oxi in zip(comp.elements, ox_states))
        if abs(charge) >= 1e-8:
            continue
        neutral_count += 1
        passed = bool(pauling_test(ox_states, enegs, symbols=symbols))
        if passed:
            pauling_count += 1
        if len(examples) < 16:
            examples.append({"oxidation_states": dict(zip(symbols, ox_states)), "pauling_passed": passed})
    score = 0.0
    if neutral_count:
        score += 0.6
    if pauling_count:
        score += 0.4
    payload = {
        "formula": comp.reduced_formula,
        "neutral_assignment_count": neutral_count,
        "pauling_pass_count": pauling_count,
        "chemical_validity_score": score,
        "examples": examples,
    }
    return "# SMACT Chemical Validity Score\n\n" + _json_block(payload)
