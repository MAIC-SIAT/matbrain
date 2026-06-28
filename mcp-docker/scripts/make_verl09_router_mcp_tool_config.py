#!/usr/bin/env python3
"""Generate a compact router-tool YAML for Mat-MCP on verl 0.9."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml

from train.rl.make_verl09_native_mcp_tool_config import (
    fetch_server_tools,
    load_selected_tools,
    normalize_parameters_schema,
)
from train.rl.mat_mcp_router_tool import _parameter_summary


PORT_DOMAINS = {
    "5668": "property",
    "5669": "structure",
    "5672": "structure",
    "5673": "stability",
    "5674": "property",
    "5675": "structure",
    "5676": "structure",
    "5677": "evidence",
    "5678": "synthesis",
    "5679": "property",
    "5680": "evidence",
    "5682": "stability",
    "5683": "characterization",
    "5684": "synthesis",
    "5685": "registry",
    "5686": "defect_interface",
}


NAME_DOMAIN_HINTS = {
    "synthesis": "synthesis",
    "precursor": "synthesis",
    "xrd": "characterization",
    "characterization": "characterization",
    "stability": "stability",
    "reaction": "stability",
    "interface": "defect_interface",
    "defect": "defect_interface",
    "dopant": "defect_interface",
    "vacancy": "defect_interface",
    "pubchem": "synthesis",
    "search": "evidence",
    "evidence": "evidence",
    "tool": "registry",
}


def infer_domain(name: str, server_url: str) -> str:
    lowered = name.lower()
    for hint, domain in NAME_DOMAIN_HINTS.items():
        if hint in lowered:
            return domain
    for port, domain in PORT_DOMAINS.items():
        if f":{port}" in server_url:
            return domain
    return "other"


async def build_catalog(selected: dict[str, str], timeout: float) -> dict[str, Any]:
    by_server: dict[str, set[str]] = {}
    for name, server_url in selected.items():
        by_server.setdefault(server_url, set()).add(name)

    tools_out: list[dict[str, Any]] = []
    for server_url, names in sorted(by_server.items()):
        tools = await fetch_server_tools(server_url, timeout)
        for tool in tools:
            name = str(tool.name)
            if name not in names:
                continue
            input_schema = getattr(tool, "inputSchema", None)
            if input_schema is None and hasattr(tool, "model_dump"):
                input_schema = tool.model_dump().get("inputSchema")
            parameters_schema = normalize_parameters_schema(input_schema)
            tools_out.append(
                {
                    "name": name,
                    "server_url": server_url,
                    "domain": infer_domain(name, server_url),
                    "description": str(getattr(tool, "description", "") or f"Mat-MCP tool {name}"),
                    "parameters": _parameter_summary(parameters_schema),
                    "required": parameters_schema.get("required", []),
                    "parameters_schema": parameters_schema,
                }
            )

    found = {item["name"] for item in tools_out}
    missing = sorted(set(selected) - found)
    if missing:
        raise SystemExit(f"Missing selected tools from MCP list_tools: {missing}")
    return {"version": 1, "tools": sorted(tools_out, key=lambda item: item["name"])}


def router_tool_schema(name: str, description: str, properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "class_name": "train.rl.mat_mcp_router_tool.MatMCPRouterTool",
        "config": {
            "type": "native",
        },
        "tool_schema": {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
    }


def build_router_yaml(catalog_path: Path, timeout: float, max_response_chars: int) -> dict[str, Any]:
    base_config = {
        "type": "native",
        "catalog_path": str(catalog_path),
        "timeout": timeout,
        "max_response_chars": max_response_chars,
    }
    tools = [
        router_tool_schema(
            "mat_mcp_list_tool_domains",
            "List compact Mat-MCP tool domains and examples. Use this before selecting tools.",
            {},
            [],
        ),
        router_tool_schema(
            "mat_mcp_select_tools",
            "Select a small set of relevant Mat-MCP backend tools for a user task.",
            {
                "task": {"type": "string", "description": "The current user task or subproblem."},
                "domain": {"type": "string", "description": "Optional domain filter such as structure, property, synthesis, stability, characterization, defect_interface, evidence."},
                "max_tools": {"type": "integer", "description": "Maximum number of backend tools to return."},
            },
            ["task"],
        ),
        router_tool_schema(
            "mat_mcp_describe_tools",
            "Return argument schemas for selected Mat-MCP backend tools.",
            {
                "tool_names": {"type": "array", "description": "Names of selected backend tools to describe."},
            },
            ["tool_names"],
        ),
        router_tool_schema(
            "mat_mcp_call_tool",
            "Execute one selected Mat-MCP backend tool by name with JSON arguments.",
            {
                "tool_name": {"type": "string", "description": "Exact backend tool name selected from the catalog."},
                "arguments": {"type": "object", "description": "JSON arguments for the backend tool."},
            },
            ["tool_name", "arguments"],
        ),
    ]
    for item in tools:
        item["config"].update(base_config)
    return {"tools": tools}


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--evidence-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--catalog-output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-response-chars", type=int, default=2000)
    args = parser.parse_args()

    input_paths = [args.main_config]
    if args.evidence_config:
        input_paths.append(args.evidence_config)
    selected = load_selected_tools(input_paths)
    catalog = await build_catalog(selected, timeout=args.timeout)

    args.catalog_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.catalog_output.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output.write_text(
        yaml.safe_dump(build_router_yaml(args.catalog_output, args.timeout, args.max_response_chars), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "catalog": str(args.catalog_output), "backend_tools": len(catalog["tools"]), "router_tools": 4}, ensure_ascii=False))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
