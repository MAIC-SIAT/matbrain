#!/usr/bin/env python3
"""Smoke-test verl 0.9 native Mat-MCP tool configuration."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from verl.tools.tool_registry import initialize_tools_from_config


async def call_tool(tool: Any, arguments: dict[str, Any]) -> None:
    instance_id, create_response = await tool.create()
    try:
        if create_response and not create_response.is_empty():
            print(json.dumps({"create_response": create_response.model_dump()}, ensure_ascii=False))
        response, reward, metrics = await tool.execute(instance_id, arguments)
        print(
            json.dumps(
                {
                    "tool": tool.name,
                    "reward": reward,
                    "metrics": metrics,
                    "response_excerpt": (response.text or "")[:1000],
                },
                ensure_ascii=False,
            )
        )
        if metrics.get("status") == "error" or (response.text or "").lstrip().startswith(("Error:", "Error：")):
            raise SystemExit(f"Tool call failed for {tool.name}: {(response.text or '')[:500]}")
    finally:
        await tool.release(instance_id)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool-config", type=Path, required=True)
    parser.add_argument("--call-tool", default=None)
    parser.add_argument("--arguments", default="{}")
    args = parser.parse_args()

    tools = initialize_tools_from_config(str(args.tool_config))
    by_name = {tool.name: tool for tool in tools}
    print(json.dumps({"tool_count": len(tools), "first_tools": list(by_name)[:10]}, ensure_ascii=False))

    if args.call_tool:
        if args.call_tool not in by_name:
            raise SystemExit(f"Unknown tool {args.call_tool!r}; available examples: {list(by_name)[:20]}")
        parsed_args = json.loads(args.arguments)
        if not isinstance(parsed_args, dict):
            raise SystemExit("--arguments must decode to a JSON object")
        asyncio.run(call_tool(by_name[args.call_tool], parsed_args))


if __name__ == "__main__":
    main()
