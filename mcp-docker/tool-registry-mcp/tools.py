import json
import os
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml

from core import llm_tool


MAIN_CONFIG = os.getenv("MAIN_TOOL_CONFIG", "/configs/structure_property_tools.yaml")
EVIDENCE_CONFIG = os.getenv("EVIDENCE_TOOL_CONFIG", "/configs/evidence_tools.yaml")

DOMAIN_KEYWORDS = {
    "generation": ["generate_crystal", "pyxtal", "crystallm"],
    "structure": ["structure", "cif", "symmetry", "wyckoff", "coordination", "bond", "standardize", "perturb"],
    "chemistry": ["smact", "valence", "oxidation", "formula", "precursor", "pubchem", "rdkit", "solvent", "ligand"],
    "property": ["predict", "matgl", "uip", "mace", "energy", "band_gap", "bulk_modulus", "relax"],
    "synthesis": ["synthesis", "reaction", "precursor", "thermo", "route", "open_system"],
    "characterization": ["xrd", "characterization", "phase_purity", "peak"],
    "database": ["db_router", "optimade", "materials", "novelty", "mp_"],
    "literature": ["search", "openalex", "crossref", "bing", "tavily", "evidence"],
    "verifier": ["verifier", "validate", "score_candidate", "reward"],
    "surface": ["slab", "surface", "adsorption", "adsorbate", "nrr"],
    "magnetism": ["magnet", "magmom"],
}

TASK_PRESETS = {
    "structure_design": ["generation", "structure", "chemistry", "property", "verifier"],
    "property_prediction": ["structure", "property", "verifier"],
    "synthesis_planning": ["chemistry", "synthesis", "database", "characterization", "verifier"],
    "rebuttal_evidence": ["literature", "database", "characterization", "synthesis"],
    "catalysis_surface": ["surface", "structure", "property", "chemistry"],
    "magnetism": ["magnetism", "structure", "property"],
}

ARTIFACT_STORE_DIR = Path(os.getenv("ARTIFACT_STORE_DIR", "/app/artifacts"))
ARTIFACT_PREFIX = "cif_"


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _canonical_cif_ref(cif_string: str) -> str:
    digest = hashlib.sha256(cif_string.encode("utf-8")).hexdigest()
    return f"{ARTIFACT_PREFIX}{digest[:12]}"


def _artifact_path(cif_ref: str) -> Path:
    if not cif_ref.startswith(ARTIFACT_PREFIX):
        raise ValueError(f"unsupported cif_ref: {cif_ref}")
    digest = cif_ref[len(ARTIFACT_PREFIX) :].strip()
    if len(digest) != 12:
        raise ValueError(f"invalid cif_ref digest: {cif_ref}")
    path = ARTIFACT_STORE_DIR / "cif"
    path.mkdir(parents=True, exist_ok=True)
    matches = sorted(path.glob(f"{digest}*.cif"))
    if matches:
        return matches[0]
    return path / f"{digest}.cif"


def _formula_hint(cif_string: str) -> str | None:
    for pattern in [
        r"_chemical_formula_structural\s+([^\n\r]+)",
        r"_chemical_formula_sum\s+([^\n\r]+)",
        r"data_([A-Za-z0-9_.()+\-]+)",
    ]:
        import re

        match = re.search(pattern, cif_string)
        if match:
            return match.group(1).strip().strip("'").strip('"')
    return None


def _load_yaml(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return list(data.get("tools", []))


def _load_all_tools() -> List[Dict[str, Any]]:
    rows = []
    for profile, path in [("main", MAIN_CONFIG), ("evidence", EVIDENCE_CONFIG)]:
        try:
            for item in _load_yaml(path):
                row = dict(item)
                row["profile"] = profile
                row["domains"] = _infer_domains(row["name"])
                rows.append(row)
        except Exception as exc:
            rows.append({"name": "__config_error__", "server_url": "", "profile": profile, "domains": [], "error": f"{path}: {exc}"})
    return rows


def _infer_domains(tool_name: str) -> List[str]:
    name = tool_name.lower()
    domains = []
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(keyword in name for keyword in keywords):
            domains.append(domain)
    return domains or ["general"]


@llm_tool(
    name="list_mat_mcp_services",
    description="List configured Mat-MCP services, profiles, and tool counts from YAML configs.",
)
async def list_mat_mcp_services(profile: str = "all") -> str:
    tools = [row for row in _load_all_tools() if profile == "all" or row.get("profile") == profile]
    by_server: Dict[str, Dict[str, Any]] = {}
    for row in tools:
        url = row.get("server_url", "")
        by_server.setdefault(url, {"server_url": url, "profiles": set(), "tool_count": 0, "tools": []})
        by_server[url]["profiles"].add(row.get("profile"))
        by_server[url]["tool_count"] += 1
        by_server[url]["tools"].append(row.get("name"))
    services = []
    for item in by_server.values():
        item["profiles"] = sorted(item["profiles"])
        services.append(item)
    services.sort(key=lambda item: item["server_url"])
    return _json_text({"ok": True, "profile": profile, "service_count": len(services), "services": services})


@llm_tool(
    name="list_tools_by_domain",
    description="List Mat-MCP tools matching a domain such as structure, synthesis, property, database, literature, verifier.",
)
async def list_tools_by_domain(domain: str, profile: str = "all") -> str:
    domain = domain.lower()
    rows = [
        row for row in _load_all_tools()
        if (profile == "all" or row.get("profile") == profile) and domain in row.get("domains", [])
    ]
    return _json_text({"ok": True, "domain": domain, "profile": profile, "tool_count": len(rows), "tools": rows})


@llm_tool(
    name="select_tools_for_task",
    description="Select a compact tool subset for a task preset or free-text task description.",
)
async def select_tools_for_task(task: str, profile: str = "main", max_tools: int = 24) -> str:
    task_key = task.lower().replace(" ", "_")
    domains = TASK_PRESETS.get(task_key)
    if domains is None:
        task_text = task.lower()
        domains = [domain for domain, keywords in DOMAIN_KEYWORDS.items() if domain in task_text or any(k in task_text for k in keywords)]
    domains = domains or ["structure", "property", "verifier"]
    candidates = [
        row for row in _load_all_tools()
        if (profile == "all" or row.get("profile") == profile) and any(domain in row.get("domains", []) for domain in domains)
    ]
    # Prefer lower-level validators and deterministic tools before slow/external tools.
    priority_terms = ["validate", "verifier", "smact", "pymatgen", "matminer", "xrd", "reaction", "rdkit", "uip"]
    def score(row):
        name = row.get("name", "").lower()
        return -sum(1 for term in priority_terms if term in name), name
    selected = sorted(candidates, key=score)[: max(1, int(max_tools))]
    return _json_text({"ok": True, "task": task, "profile": profile, "domains": domains, "tool_count": len(selected), "tools": selected})


@llm_tool(
    name="get_tool_policy_profile",
    description="Return recommended tool exposure policy for RL, evidence, and verifier-only use.",
)
async def get_tool_policy_profile() -> str:
    return _json_text({
        "ok": True,
        "profiles": {
            "main_rl": {
                "config": MAIN_CONFIG,
                "rule": "Prefer deterministic/offline actor tools; avoid open web retrieval during rollout.",
            },
            "evidence": {
                "config": EVIDENCE_CONFIG,
                "rule": "Use retrieval after generation for rebuttal, precedent, and post-hoc validation.",
            },
            "verifier_reward": {
                "rule": "Use verifier outputs in reward code; do not let policy optimize by directly editing reward components.",
            },
        },
        "task_presets": TASK_PRESETS,
        "domains": sorted(DOMAIN_KEYWORDS),
    })


@llm_tool(
    name="register_cif_artifact",
    description="Register a raw CIF string as a reusable artifact handle and return a short cif_ref for later tool calls.",
)
async def register_cif_artifact(cif_string: str, label: str = "input_cif") -> str:
    cif_content = cif_string.strip()
    full_digest = hashlib.sha256(cif_content.encode("utf-8")).hexdigest()
    cif_ref = _canonical_cif_ref(cif_content)
    path = ARTIFACT_STORE_DIR / "cif" / f"{full_digest}.cif"
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(cif_content, encoding="utf-8")
    payload = {
        "ok": True,
        "cif_ref": cif_ref,
        "canonical_ref": f"cif:sha256:{full_digest}",
        "label": label,
        "chars": len(cif_content),
        "formula_hint": _formula_hint(cif_content),
        "artifact_type": "cif",
    }
    return _json_text(payload)
