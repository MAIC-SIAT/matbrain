#!/usr/bin/env python3
"""Generate verl 0.9 native-tool YAML for Mat-MCP SSE tools."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

import yaml
from mcp import ClientSession
from mcp.client.sse import sse_client


def load_selected_tools(paths: list[Path]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for path in paths:
        if path is None:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("tools", []):
            selected[str(item["name"])] = str(item["server_url"])
    return selected


def normalize_property_schema(prop: Any) -> dict[str, Any]:
    if not isinstance(prop, dict):
        return {"type": "string"}
    out: dict[str, Any] = {}
    typ = prop.get("type")
    if typ is None:
        if "enum" in prop:
            typ = "string"
        elif "anyOf" in prop and isinstance(prop["anyOf"], list):
            for option in prop["anyOf"]:
                if isinstance(option, dict) and option.get("type") != "null":
                    typ = option.get("type")
                    break
        else:
            typ = "string"
    out["type"] = typ
    if "description" in prop and prop["description"] is not None:
        out["description"] = str(prop["description"])
    if "enum" in prop and isinstance(prop["enum"], list):
        out["enum"] = prop["enum"]
    return out


def normalize_parameters_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}, "required": []}
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        properties = {}
    required = schema.get("required")
    if not isinstance(required, list):
        required = []
    return {
        "type": "object",
        "properties": {str(key): normalize_property_schema(value) for key, value in properties.items()},
        "required": [str(item) for item in required],
    }


async def fetch_server_tools(server_url: str, timeout: float) -> list[Any]:
    async with sse_client(server_url, timeout=timeout, sse_read_timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            result = await session.list_tools()
            return list(result.tools)


async def build_tools(selected: dict[str, str], timeout: float, class_name: str) -> list[dict[str, Any]]:
    by_server: dict[str, set[str]] = {}
    for name, server_url in selected.items():
        by_server.setdefault(server_url, set()).add(name)

    output: list[dict[str, Any]] = []
    for server_url, names in sorted(by_server.items()):
        tools = await fetch_server_tools(server_url, timeout)
        for tool in tools:
            name = str(tool.name)
            if name not in names:
                continue
            input_schema = getattr(tool, "inputSchema", None)
            if input_schema is None and hasattr(tool, "model_dump"):
                input_schema = tool.model_dump().get("inputSchema")
            output.append(
                {
                    "class_name": class_name,
                    "config": {
                        "type": "native",
                        "server_url": server_url,
                        "timeout": timeout,
                    },
                    "tool_schema": {
                        "type": "function",
                        "function": {
                            "name": name,
                            "description": str(getattr(tool, "description", "") or f"Mat-MCP tool {name}"),
                            "parameters": normalize_parameters_schema(input_schema),
                        },
                    },
                }
            )
    found = {item["tool_schema"]["function"]["name"] for item in output}
    missing = sorted(set(selected) - found)
    if missing:
        raise SystemExit(f"Missing selected tools from MCP list_tools: {missing}")
    return output


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--evidence-config", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--class-name", default="train.rl.mat_mcp_native_tool.MatMCPTool")
    args = parser.parse_args()

    input_paths = [args.main_config]
    if args.evidence_config:
        input_paths.append(args.evidence_config)
    selected = load_selected_tools(input_paths)
    tools = await build_tools(selected, timeout=args.timeout, class_name=args.class_name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump({"tools": tools}, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "tools": len(tools)}, ensure_ascii=False))


def main() -> None:
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
