import json
import re
from typing import Any, Dict, List
from urllib.parse import quote

import httpx

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except Exception:
    Chem = None
    Descriptors = None
    rdMolDescriptors = None

from core import llm_tool


HTTP_TIMEOUT = 30.0
HAZARD_KEYWORDS = [
    "toxic",
    "fatal",
    "flammable",
    "corrosive",
    "explosive",
    "oxidizer",
    "carcinogen",
    "mutagen",
    "reproductive",
    "acute toxicity",
]


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


async def _get_json(url: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT, trust_env=True) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()


async def _pubchem_cids(name_or_formula: str, top_k: int = 10) -> List[int]:
    data = await _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name_or_formula)}/cids/JSON")
    return data.get("IdentifierList", {}).get("CID", [])[: max(1, int(top_k))]


async def _pubchem_properties(cid: int) -> Dict[str, Any]:
    props = "MolecularFormula,MolecularWeight,IUPACName,CanonicalSMILES,IsomericSMILES,InChI,InChIKey,XLogP,TPSA,Charge,Complexity"
    data = await _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/{props}/JSON")
    return data.get("PropertyTable", {}).get("Properties", [{}])[0]


async def _pubchem_synonyms(cid: int, top_k: int = 20) -> List[str]:
    data = await _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON")
    info = data.get("InformationList", {}).get("Information", [{}])[0]
    return info.get("Synonym", [])[: max(1, int(top_k))]


async def _pubchem_description(cid: int) -> str:
    try:
        data = await _get_json(f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/compound/{cid}/JSON")
    except Exception:
        return ""
    texts = []
    def walk(section):
        for info in section.get("Information", []):
            value = info.get("Value", {})
            if "StringWithMarkup" in value:
                for item in value["StringWithMarkup"]:
                    text = item.get("String")
                    if text:
                        texts.append(text)
        for sub in section.get("Section", []):
            heading = str(sub.get("TOCHeading", "")).lower()
            if any(key in heading for key in ["safety", "hazard", "ghs", "toxic", "flamm"]):
                walk(sub)
    for section in data.get("Record", {}).get("Section", []):
        walk(section)
    return " ".join(texts[:8])[:3000]


def _rdkit_payload(smiles: str) -> Dict[str, Any]:
    if Chem is None:
        return {"available": False, "error": "RDKit is not installed"}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"available": True, "valid": False, "error": "invalid SMILES"}
    canonical = Chem.MolToSmiles(mol, canonical=True)
    return {
        "available": True,
        "valid": True,
        "canonical_smiles": canonical,
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "exact_mol_weight": float(Descriptors.ExactMolWt(mol)),
        "num_h_donors": int(Descriptors.NumHDonors(mol)),
        "num_h_acceptors": int(Descriptors.NumHAcceptors(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "logp": float(Descriptors.MolLogP(mol)),
        "rotatable_bonds": int(Descriptors.NumRotatableBonds(mol)),
        "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
    }


def _hazard_flags(text: str) -> List[str]:
    lower = str(text or "").lower()
    return [keyword for keyword in HAZARD_KEYWORDS if keyword in lower]


@llm_tool(
    name="resolve_precursor_pubchem",
    description="Resolve precursor name/formula to PubChem CIDs, synonyms, and basic compound properties.",
)
async def resolve_precursor_pubchem(name_or_formula: str, top_k: int = 5) -> str:
    payload: Dict[str, Any] = {"ok": True, "query": name_or_formula, "results": [], "errors": []}
    try:
        cids = await _pubchem_cids(name_or_formula, top_k)
        for cid in cids:
            props = await _pubchem_properties(cid)
            synonyms = await _pubchem_synonyms(cid, 8)
            payload["results"].append({"cid": cid, "properties": props, "synonyms": synonyms})
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="get_pubchem_compound_properties",
    description="Fetch PubChem molecular properties by CID or name.",
)
async def get_pubchem_compound_properties(identifier: str, identifier_type: str = "name") -> str:
    payload: Dict[str, Any] = {"ok": True, "identifier": identifier, "identifier_type": identifier_type, "properties": {}, "errors": []}
    try:
        cid = int(identifier) if identifier_type == "cid" else (await _pubchem_cids(identifier, 1))[0]
        props = await _pubchem_properties(cid)
        payload["cid"] = cid
        payload["properties"] = props
        smiles = props.get("CanonicalSMILES")
        if smiles:
            payload["rdkit"] = _rdkit_payload(smiles)
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="get_pubchem_safety_summary",
    description="Fetch PubChem safety/hazard text and simple hazard keyword flags for a precursor.",
)
async def get_pubchem_safety_summary(name_or_cid: str, identifier_type: str = "name") -> str:
    payload: Dict[str, Any] = {"ok": True, "identifier": name_or_cid, "safety_text": "", "hazard_flags": [], "errors": []}
    try:
        cid = int(name_or_cid) if identifier_type == "cid" else (await _pubchem_cids(name_or_cid, 1))[0]
        text = await _pubchem_description(cid)
        payload["cid"] = cid
        payload["safety_text"] = text
        payload["hazard_flags"] = _hazard_flags(text)
        payload["reward_components"] = {
            "hazard_penalty": -min(1.0, 0.15 * len(payload["hazard_flags"])),
            "safety_info_available": 1.0 if text else 0.0,
        }
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)


@llm_tool(
    name="canonicalize_smiles_rdkit",
    description="Canonicalize a SMILES string and compute RDKit descriptors when RDKit is available.",
)
async def canonicalize_smiles_rdkit(smiles: str) -> str:
    payload = {"ok": True, "input_smiles": smiles, "rdkit": _rdkit_payload(smiles)}
    return _json_text(payload)


@llm_tool(
    name="check_precursor_hazard_flags",
    description="Resolve each precursor in a set through PubChem and return hazard flags for route-level filtering.",
)
async def check_precursor_hazard_flags(precursor_names: List[str]) -> str:
    rows = []
    for name in precursor_names:
        try:
            cid = (await _pubchem_cids(name, 1))[0]
            text = await _pubchem_description(cid)
            rows.append({"name": name, "cid": cid, "hazard_flags": _hazard_flags(text), "safety_text_excerpt": text[:500]})
        except Exception as exc:
            rows.append({"name": name, "error": str(exc), "hazard_flags": []})
    total_flags = sum(len(row.get("hazard_flags", [])) for row in rows)
    payload = {
        "ok": True,
        "precursors": rows,
        "total_hazard_flags": total_flags,
        "reward_components": {
            "route_safety_score": float(max(0.0, 1.0 - 0.1 * total_flags)),
            "hazard_penalty": float(-0.1 * total_flags),
        },
    }
    return _json_text(payload)


@llm_tool(
    name="check_solvent_ligand_compatibility",
    description="Heuristic compatibility check for solvent/ligand molecules using PubChem/RDKit properties.",
)
async def check_solvent_ligand_compatibility(solvent: str, ligand: str = "", max_mw: float = 600.0) -> str:
    payload: Dict[str, Any] = {"ok": True, "solvent": solvent, "ligand": ligand, "warnings": [], "score": 1.0, "details": {}, "errors": []}
    try:
        solvent_props = (await _pubchem_properties((await _pubchem_cids(solvent, 1))[0]))
        payload["details"]["solvent"] = solvent_props
        score = 1.0
        if float(solvent_props.get("MolecularWeight", 0) or 0) > max_mw:
            payload["warnings"].append("solvent molecular weight is unusually high")
            score -= 0.2
        if ligand:
            ligand_props = (await _pubchem_properties((await _pubchem_cids(ligand, 1))[0]))
            payload["details"]["ligand"] = ligand_props
            if float(ligand_props.get("MolecularWeight", 0) or 0) > max_mw:
                payload["warnings"].append("ligand molecular weight is high")
                score -= 0.1
        payload["score"] = float(max(0.0, score))
        payload["reward_components"] = {"solvent_ligand_compatibility": payload["score"]}
    except Exception as exc:
        payload["ok"] = False
        payload["errors"].append(str(exc))
    return _json_text(payload)
