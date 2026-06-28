import json
from typing import Any, Dict, List

from pymatgen.io.cif import CifWriter
from pyxtal import pyxtal
from pyxtal.symmetry import Group

from core import llm_tool


_UNICODE_SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
    }
)


def _json_block(payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _formula_to_species_counts(formula: str) -> tuple[list[str], list[int]]:
    from pymatgen.core import Composition

    normalized_formula = str(formula).translate(_UNICODE_SUBSCRIPT_TRANSLATION)
    comp = Composition(normalized_formula)
    reduced = comp.reduced_composition.get_el_amt_dict()
    species = []
    counts = []
    for el, amt in reduced.items():
        species.append(el)
        counts.append(int(round(float(amt))))
    return species, counts


@llm_tool(
    name="generate_crystal_pyxtal",
    description="Generate random symmetry-constrained crystal structures with PyXtal for a formula and space group.",
)
async def generate_crystal_pyxtal(
    formula: str,
    space_group: int,
    dimension: int = 3,
    attempts: int = 3,
    max_returned: int = 2,
) -> str:
    species, counts = _formula_to_species_counts(formula)
    attempts = max(1, min(int(attempts), 20))
    max_returned = max(1, min(int(max_returned), 8))
    records = []
    result = "# PyXtal Crystal Generation\n\n"
    for attempt in range(attempts):
        xtal = pyxtal()
        try:
            xtal.from_random(int(dimension), int(space_group), species, counts)
            if not xtal.valid:
                continue
            structure = xtal.to_pymatgen()
            records.append({
                "attempt": attempt,
                "formula": structure.composition.reduced_formula,
                "num_sites": len(structure),
                "space_group": int(space_group),
                "density": float(structure.density),
            })
            result += f"\n\n## Candidate {len(records)}\n\n```cif\n"
            result += str(CifWriter(structure)).strip()
            result += "\n```"
            if len(records) >= max_returned:
                break
        except Exception as exc:
            records.append({"attempt": attempt, "error": str(exc)})
    payload = {
        "target_formula": formula,
        "species": species,
        "counts": counts,
        "space_group": int(space_group),
        "num_successful_candidates": sum(1 for r in records if "error" not in r),
        "records": records,
    }
    return "# PyXtal Crystal Generation\n\n" + _json_block(payload) + result


@llm_tool(
    name="check_wyckoff_feasibility_pyxtal",
    description="Check whether a formula/site-count target is likely compatible with a PyXtal space group.",
)
async def check_wyckoff_feasibility_pyxtal(formula: str, space_group: int) -> str:
    species, counts = _formula_to_species_counts(formula)
    group = Group(int(space_group))
    multiplicities = []
    try:
        for wyckoff in group:
            try:
                multiplicities.append(int(wyckoff.multiplicity))
            except Exception:
                pass
    except Exception:
        try:
            multiplicities = [int(wp.multiplicity) for wp in group.Wyckoff_positions]
        except Exception:
            multiplicities = []
    possible = {}
    for species_name, count in zip(species, counts):
        dp = [False] * (count + 1)
        dp[0] = True
        for total in range(count + 1):
            if not dp[total]:
                continue
            for mult in multiplicities:
                nxt = total + mult
                if nxt <= count:
                    dp[nxt] = True
        possible[species_name] = bool(dp[count]) if multiplicities else None
    payload = {
        "formula": formula,
        "species": species,
        "counts": counts,
        "space_group": int(space_group),
        "wyckoff_multiplicities": sorted(set(multiplicities)),
        "per_species_count_feasible": possible,
        "all_counts_feasible": all(v is True for v in possible.values()) if multiplicities else None,
    }
    return "# PyXtal Wyckoff Feasibility\n\n" + _json_block(payload)


@llm_tool(
    name="list_wyckoff_positions_pyxtal",
    description="List Wyckoff position multiplicities and labels for a space group using PyXtal.",
)
async def list_wyckoff_positions_pyxtal(space_group: int) -> str:
    group = Group(int(space_group))
    positions = []
    try:
        iterator = list(group)
    except Exception:
        iterator = getattr(group, "Wyckoff_positions", [])
    for wp in iterator:
        positions.append({
            "letter": getattr(wp, "letter", None),
            "multiplicity": getattr(wp, "multiplicity", None),
            "site_symmetry": str(getattr(wp, "site_symmetry", "")),
        })
    payload = {
        "space_group": int(space_group),
        "num_positions": len(positions),
        "positions": positions,
    }
    return "# PyXtal Wyckoff Positions\n\n" + _json_block(payload)


@llm_tool(
    name="convert_pyxtal_to_pymatgen_summary",
    description="Generate one PyXtal candidate and return a compact pymatgen-compatible summary plus CIF.",
)
async def convert_pyxtal_to_pymatgen_summary(formula: str, space_group: int) -> str:
    species, counts = _formula_to_species_counts(formula)
    xtal = pyxtal()
    xtal.from_random(3, int(space_group), species, counts)
    if not xtal.valid:
        return "Error: PyXtal failed to generate a valid structure"
    structure = xtal.to_pymatgen()
    payload = {
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "lattice": {
            "a": structure.lattice.a,
            "b": structure.lattice.b,
            "c": structure.lattice.c,
            "alpha": structure.lattice.alpha,
            "beta": structure.lattice.beta,
            "gamma": structure.lattice.gamma,
            "volume": structure.lattice.volume,
        },
        "density": float(structure.density),
    }
    result = "# PyXtal to pymatgen Summary\n\n" + _json_block(payload)
    result += "\n\n## CIF\n\n```cif\n" + str(CifWriter(structure)).strip() + "\n```"
    return result
