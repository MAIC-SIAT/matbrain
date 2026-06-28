#!/usr/bin/env python3
"""Fast smoke test for the RL MCP tool pool.

This intentionally avoids slow generation/relaxation calls. It verifies that
the active MCP servers expose the expected tool names and that quick low-level
tools work through SSE.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any

try:
    import yaml
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError as exc:  # pragma: no cover - user-facing environment check
    raise SystemExit(
        "Missing dependency. Run this in the same environment that has the MCP "
        f"client installed. Original error: {exc}"
    ) from exc


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server_url: str


async def list_tools(server_url: str, timeout: float) -> set[str]:
    async with sse_client(server_url, timeout=timeout, sse_read_timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            return {tool.name for tool in result.tools}


async def call_tool(
    server_url: str, tool_name: str, arguments: dict[str, Any], timeout: float
) -> str:
    async with sse_client(server_url, timeout=timeout, sse_read_timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments), timeout=timeout
            )
            return result.content[0].text


def load_tool_specs(path: str) -> list[ToolSpec]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return [ToolSpec(item["name"], item["server_url"]) for item in data["tools"]]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="train/rl/tool_configs/structure_property_tools.yaml",
        help="RL tool config YAML.",
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    specs = load_tool_specs(args.config)
    by_server: dict[str, set[str]] = {}
    for spec in specs:
        by_server.setdefault(spec.server_url, set()).add(spec.name)

    print("Checking MCP tool lists...")
    for server_url, expected in sorted(by_server.items()):
        observed = await list_tools(server_url, args.timeout)
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        print(
            json.dumps(
                {
                    "server_url": server_url,
                    "expected_count": len(expected),
                    "observed_count": len(observed),
                    "missing": missing,
                    "extra_not_in_rl_pool": extra,
                },
                ensure_ascii=False,
            )
        )
        if missing:
            raise SystemExit(f"Missing expected tools from {server_url}: {missing}")

    pymatgen_url = "http://localhost:5672/pymatgen/sse"
    print("Calling quick PyMatGen validation tool...")
    text = await call_tool(
        pymatgen_url,
        "check_chemical_formula_valence_pymatgen",
        {"formula": "Fe2O3"},
        args.timeout,
    )
    print(text[:1000])

    smact_url = "http://localhost:5673/sse"
    print("Calling quick SMACT validation tool...")
    text = await call_tool(
        smact_url,
        "score_composition_chemical_validity_smact",
        {"formula": "Fe2O3"},
        args.timeout,
    )
    print(text[:1000])

    matminer_url = "http://localhost:5674/sse"
    print("Calling quick matminer composition featurizer...")
    text = await call_tool(
        matminer_url,
        "featurize_composition_matminer",
        {"formula": "Fe2O3"},
        args.timeout,
    )
    print(text[:1000])

    pyxtal_url = "http://localhost:5675/sse"
    print("Calling quick PyXtal Wyckoff feasibility check...")
    text = await call_tool(
        pyxtal_url,
        "check_wyckoff_feasibility_pyxtal",
        {"formula": "Fe2O3", "space_group": 167},
        args.timeout,
    )
    print(text[:1000])

    verifier_url = "http://localhost:5676/sse"
    print("Calling quick structure verifier formula check...")
    text = await call_tool(
        verifier_url,
        "validate_formula_verifier",
        {"formula": "Fe2O3"},
        args.timeout,
    )
    print(text[:1000])

    synthesis_url = "http://localhost:5678/sse"
    print("Calling quick synthesis precursor recommender...")
    text = await call_tool(
        synthesis_url,
        "recommend_precursors_kb",
        {"target_formula": "BaTiO3"},
        args.timeout,
    )
    print(text[:1000])

    reaction_url = "http://localhost:5682/sse"
    print("Calling quick reaction balancing tool...")
    text = await call_tool(
        reaction_url,
        "balance_reaction_thermo",
        {"reactant_formulas": ["BaCO3", "TiO2"], "product_formulas": ["BaTiO3", "CO2"]},
        args.timeout,
    )
    print(text[:1000])

    precursor_url = "http://localhost:5684/sse"
    print("Calling quick RDKit SMILES canonicalizer...")
    text = await call_tool(
        precursor_url,
        "canonicalize_smiles_rdkit",
        {"smiles": "CCO"},
        args.timeout,
    )
    print(text[:1000])

    registry_url = "http://localhost:5685/sse"
    print("Calling quick tool registry selector...")
    text = await call_tool(
        registry_url,
        "select_tools_for_task",
        {"task": "structure_design", "profile": "main", "max_tools": 8},
        args.timeout,
    )
    print(text[:1000])

    print("MCP smoke test passed.")


if __name__ == "__main__":
    asyncio.run(main())
