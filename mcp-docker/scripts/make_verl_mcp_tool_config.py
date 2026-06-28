#!/usr/bin/env python3
"""Build verl/SGLang multi-turn MCP tool configs from MatBrain tool YAMLs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

import yaml


def load_tools(paths: list[Path]) -> tuple[dict[str, str], list[str]]:
    servers: dict[str, str] = {}
    tool_names: list[str] = []
    for path in paths:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for item in data.get("tools", []):
            name = str(item["name"])
            url = str(item["server_url"])
            parsed = urlparse(url)
            server_name = f"mat_mcp_{parsed.hostname}_{parsed.port}_{parsed.path.strip('/').replace('/', '_')}"
            servers.setdefault(server_name, url)
            if name not in tool_names:
                tool_names.append(name)
    return servers, tool_names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", type=Path, required=True)
    parser.add_argument("--evidence-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("train/rl/tool_configs/verl_mcp"))
    parser.add_argument("--name", default="mat_mcp_structure_property")
    parser.add_argument(
        "--runtime-root",
        default=None,
        help="Optional repo root path as seen by the training runtime, e.g. /workspace/matbrain.",
    )
    parser.add_argument("--rate-limit", type=int, default=120)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    input_paths = [args.main_config]
    if args.evidence_config:
        input_paths.append(args.evidence_config)

    servers, tool_names = load_tools(input_paths)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    servers_path = args.output_dir / f"{args.name}_servers.json"
    tool_config_path = args.output_dir / f"{args.name}_tool_config.yaml"

    servers_payload = {
        "mcpServers": {
            server_name: {"url": url}
            for server_name, url in sorted(servers.items())
        }
    }
    servers_path.write_text(json.dumps(servers_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.runtime_root:
        repo_root = Path.cwd().resolve()
        server_config_for_runtime = Path(args.runtime_root) / servers_path.resolve().relative_to(repo_root)
    else:
        server_config_for_runtime = servers_path.resolve()

    tool_payload = {
        "tools": [
            {
                "class_name": "verl.tools.mcp_base_tool.MCPBaseTool",
                "config": {
                    "rate_limit": args.rate_limit,
                    "timeout": args.timeout,
                    "type": "mcp",
                },
                "mcp": {
                    "mcp_servers_config_path": str(server_config_for_runtime),
                    "tool_selected_list": tool_names,
                },
            }
        ]
    }
    tool_config_path.write_text(yaml.safe_dump(tool_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(json.dumps({
        "servers_config": str(servers_path),
        "tool_config": str(tool_config_path),
        "servers": len(servers),
        "tools": len(tool_names),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
