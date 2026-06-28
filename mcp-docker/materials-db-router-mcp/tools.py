import json
import os
from typing import Any, Dict, List

import httpx
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifWriter

try:
    from mp_api.client import MPRester
except Exception:
    MPRester = None

from core import llm_tool


MP_API_KEY = os.getenv("MP_API_KEY", "")
OPTIMADE_TIMEOUT = float(os.getenv("OPTIMADE_TIMEOUT", "30"))
DEFAULT_OPTIMADE_ENDPOINTS = {
    "cod": "https://www.crystallography.net/cod/optimade/v1/structures",
    "materials_project": "https://optimade.materialsproject.org/v1/structures",
    "jarvis": "https://jarvis.nist.gov/optimade/jarvisdft/v1/structures",
}


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _doc_to_dict(doc: Any) -> Dict[str, Any]:
    if hasattr(doc, "model_dump"):
        data = doc.model_dump()
    elif hasattr(doc, "dict"):
        data = doc.dict()
    else:
        data = dict(doc)
    wanted = [
        "material_id",
        "formula_pretty",
        "formula_anonymous",
        "chemsys",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "is_stable",
        "theoretical",
        "database_IDs",
    ]
    return {key: data.get(key) for key in wanted if key in data}


def _mp_available() -> bool:
    return bool(MP_API_KEY and MPRester is not None)


def _mp_search(**kwargs) -> List[Dict[str, Any]]:
    if not _mp_available():
        raise RuntimeError("Materials Project API unavailable; set MP_API_KEY and install mp-api")
    fields = [
        "material_id",
        "formula_pretty",
        "formula_anonymous",
        "chemsys",
        "symmetry",
        "energy_above_hull",
        "formation_energy_per_atom",
        "band_gap",
        "is_stable",
        "theoretical",
        "database_IDs",
    ]
    with MPRester(MP_API_KEY) as mpr:
        docs = mpr.materials.summary.search(fields=fields, **kwargs)
    return [_doc_to_dict(doc) for doc in docs]


@llm_tool(
    name="search_materials_by_formula_db_router",
    description="Search Materials Project by formula and return compact evidence fields; retrieval tool for novelty/evidence tasks.",
)
async def search_materials_by_formula_db_router(formula: str, top_k: int = 10) -> str:
    payload: Dict[str, Any] = {
        "ok": True,
        "query": {"formula": formula, "top_k": int(top_k)},
        "sources": ["materials_project"],
        "results": [],
        "errors": [],
    }
    try:
        docs = _mp_search(formula=[Composition(formula).reduced_formula])
        payload["results"] = docs[: max(1, int(top_k))]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="search_materials_by_chemsys_db_router",
    description="Search Materials Project by chemical system and stable/near-stable filters.",
)
async def search_materials_by_chemsys_db_router(
    chemsys: str,
    top_k: int = 20,
    stable_only: bool = False,
    max_energy_above_hull: float = 0.10,
) -> str:
    elements = "-".join(sorted(part.strip() for part in chemsys.replace(",", "-").split("-") if part.strip()))
    payload: Dict[str, Any] = {
        "ok": True,
        "query": {
            "chemsys": elements,
            "top_k": int(top_k),
            "stable_only": bool(stable_only),
            "max_energy_above_hull": float(max_energy_above_hull),
        },
        "sources": ["materials_project"],
        "results": [],
        "errors": [],
    }
    try:
        kwargs: Dict[str, Any] = {"chemsys": elements}
        if stable_only:
            kwargs["is_stable"] = True
        docs = _mp_search(**kwargs)
        docs = [
            item for item in docs
            if item.get("energy_above_hull") is None or float(item.get("energy_above_hull")) <= float(max_energy_above_hull)
        ]
        docs.sort(key=lambda item: float(item.get("energy_above_hull") or 999.0))
        payload["results"] = docs[: max(1, int(top_k))]
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="fetch_mp_structure_db_router",
    description="Fetch a Materials Project structure by material_id and return CIF plus compact metadata.",
)
async def fetch_mp_structure_db_router(material_id: str, conventional_unit_cell: bool = False) -> str:
    payload: Dict[str, Any] = {
        "ok": True,
        "material_id": material_id,
        "source": "materials_project",
        "metadata": {},
        "cif": None,
        "errors": [],
    }
    try:
        if not _mp_available():
            raise RuntimeError("Materials Project API unavailable; set MP_API_KEY and install mp-api")
        with MPRester(MP_API_KEY) as mpr:
            structure = mpr.get_structure_by_material_id(material_id, conventional_unit_cell=bool(conventional_unit_cell))
            docs = mpr.materials.summary.search(material_ids=[material_id], fields=["material_id", "formula_pretty", "symmetry", "energy_above_hull", "is_stable"])
        payload["metadata"] = _doc_to_dict(docs[0]) if docs else {}
        payload["cif"] = str(CifWriter(structure))
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="check_novelty_against_mp_db_router",
    description="Check whether a candidate CIF matches same-formula Materials Project structures using StructureMatcher.",
)
async def check_novelty_against_mp_db_router(
    cif_string: str,
    top_k: int = 20,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> str:
    structure = Structure.from_str(cif_string, fmt="cif")
    formula = structure.composition.reduced_formula
    matcher = StructureMatcher(ltol=float(ltol), stol=float(stol), angle_tol=float(angle_tol))
    payload: Dict[str, Any] = {
        "ok": True,
        "formula": formula,
        "novel": None,
        "matched_materials": [],
        "checked_count": 0,
        "errors": [],
        "reward_components": {"novelty": 0.0, "known_structure_match_penalty": 0.0},
    }
    try:
        if not _mp_available():
            raise RuntimeError("Materials Project API unavailable; set MP_API_KEY and install mp-api")
        docs = _mp_search(formula=[formula])[: max(1, int(top_k))]
        with MPRester(MP_API_KEY) as mpr:
            for doc in docs:
                mpid = str(doc.get("material_id"))
                try:
                    ref = mpr.get_structure_by_material_id(mpid)
                    payload["checked_count"] += 1
                    if matcher.fit(structure, ref):
                        payload["matched_materials"].append(doc)
                except Exception as exc:
                    payload["errors"].append(f"{mpid}: {exc}")
        payload["novel"] = len(payload["matched_materials"]) == 0
        payload["reward_components"] = {
            "novelty": 1.0 if payload["novel"] else 0.0,
            "known_structure_match_penalty": 0.0 if payload["novel"] else -1.0,
        }
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="optimade_structure_search_db_router",
    description="Query an OPTIMADE structures endpoint with a raw filter; useful for COD/JARVIS/MP cross-database evidence.",
)
async def optimade_structure_search_db_router(
    filter: str,
    provider: str = "cod",
    endpoint_url: str = "",
    page_limit: int = 5,
) -> str:
    endpoint = endpoint_url or DEFAULT_OPTIMADE_ENDPOINTS.get(provider, provider)
    payload: Dict[str, Any] = {
        "ok": True,
        "provider": provider,
        "endpoint_url": endpoint,
        "filter": filter,
        "results": [],
        "errors": [],
    }
    try:
        async with httpx.AsyncClient(timeout=OPTIMADE_TIMEOUT, trust_env=True) as client:
            response = await client.get(endpoint, params={"filter": filter, "page_limit": int(page_limit)})
            response.raise_for_status()
            data = response.json()
        rows = []
        for item in data.get("data", [])[: max(1, int(page_limit))]:
            attrs = item.get("attributes", {})
            rows.append({
                "id": item.get("id"),
                "chemical_formula_reduced": attrs.get("chemical_formula_reduced"),
                "chemical_formula_anonymous": attrs.get("chemical_formula_anonymous"),
                "elements": attrs.get("elements"),
                "nelements": attrs.get("nelements"),
                "nsites": attrs.get("nsites"),
            })
        payload["results"] = rows
        payload["meta"] = data.get("meta", {})
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="compare_materials_across_databases_db_router",
    description="Cross-check a formula in MP and configured OPTIMADE providers, returning source coverage for evidence tasks.",
)
async def compare_materials_across_databases_db_router(formula: str, providers: List[str] = None, top_k: int = 5) -> str:
    comp = Composition(formula)
    reduced = comp.reduced_formula
    providers = providers or ["cod", "jarvis"]
    payload: Dict[str, Any] = {
        "ok": True,
        "formula": reduced,
        "sources": {},
        "errors": [],
    }
    mp_text = await search_materials_by_formula_db_router(reduced, top_k)
    payload["sources"]["materials_project"] = json.loads(mp_text)
    for provider in providers:
        optimade_filter = f'chemical_formula_reduced="{reduced}"'
        text = await optimade_structure_search_db_router(optimade_filter, provider, "", top_k)
        payload["sources"][provider] = json.loads(text)
    return _json_text(payload)
