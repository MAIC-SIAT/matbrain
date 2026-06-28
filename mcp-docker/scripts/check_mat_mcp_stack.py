#!/usr/bin/env python3
"""Check Mat-MCP service health and tool exposure.

Default mode avoids slow GPU calls and open-web/API-key calls. It verifies:
1. HTTP /health endpoints for all services present in tool YAMLs.
2. MCP list_tools matches configured tool names.
3. A few deterministic no-network tool calls.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    import httpx
    import yaml
    from mcp import ClientSession
    from mcp.client.sse import sse_client
except ImportError as exc:
    raise SystemExit(f"Missing dependency for MCP stack check: {exc}") from exc


ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs"
MAIN_CONFIG = CONFIG_ROOT / "structure_property_tools.yaml"
EVIDENCE_CONFIG = CONFIG_ROOT / "evidence_tools.yaml"


@dataclass(frozen=True)
class ToolSpec:
    name: str
    server_url: str
    profile: str


def load_specs(paths: list[str]) -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for path in paths:
        profile = "evidence" if "evidence" in path else "main"
        with open(path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        for item in data.get("tools", []):
            specs.append(ToolSpec(item["name"], item["server_url"], profile))
    return specs


def health_url(server_url: str) -> str:
    parsed = urlparse(server_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/sse"):
        path = path[: -len("/sse")]
    return f"{parsed.scheme}://{parsed.netloc}{path}/health"


async def check_health(urls: list[str], timeout: float) -> list[dict[str, Any]]:
    rows = []
    async with httpx.AsyncClient(timeout=timeout, trust_env=True) as client:
        for url in sorted(set(urls)):
            hurl = health_url(url)
            try:
                response = await client.get(hurl)
                rows.append({"server_url": url, "health_url": hurl, "ok": response.status_code == 200, "status": response.status_code, "body": response.text[:500]})
            except Exception as exc:
                rows.append({"server_url": url, "health_url": hurl, "ok": False, "error": str(exc)})
    return rows


async def list_tools(server_url: str, timeout: float) -> set[str]:
    async with sse_client(server_url, timeout=timeout, sse_read_timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
            return {tool.name for tool in result.tools}


async def call_tool(server_url: str, tool_name: str, arguments: dict[str, Any], timeout: float) -> str:
    async with sse_client(server_url, timeout=timeout, sse_read_timeout=timeout) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(session.call_tool(tool_name, arguments), timeout=timeout)
            return result.content[0].text


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-config", default=str(MAIN_CONFIG))
    parser.add_argument("--evidence-config", default=str(EVIDENCE_CONFIG))
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--skip-health", action="store_true")
    parser.add_argument("--skip-list-tools", action="store_true")
    parser.add_argument("--skip-quick-calls", action="store_true")
    args = parser.parse_args()

    specs = load_specs([args.main_config, args.evidence_config])
    by_server: dict[str, set[str]] = {}
    server_by_tool: dict[str, str] = {}
    for spec in specs:
        by_server.setdefault(spec.server_url, set()).add(spec.name)
        server_by_tool.setdefault(spec.name, spec.server_url)

    if not args.skip_health:
        print("Checking /health endpoints...")
        health_rows = await check_health(list(by_server), args.timeout)
        for row in health_rows:
            print(json.dumps(row, ensure_ascii=False))
        failed = [row for row in health_rows if not row["ok"]]
        if failed:
            raise SystemExit(f"Health check failed for {len(failed)} services")

    if not args.skip_list_tools:
        print("Checking MCP list_tools against YAML configs...")
        for server_url, expected in sorted(by_server.items()):
            observed = await list_tools(server_url, args.timeout)
            missing = sorted(expected - observed)
            extra = sorted(observed - expected)
            row = {
                "server_url": server_url,
                "expected_count": len(expected),
                "observed_count": len(observed),
                "missing": missing,
                "extra_not_in_configs": extra,
            }
            print(json.dumps(row, ensure_ascii=False))
            if missing:
                raise SystemExit(f"Missing expected tools from {server_url}: {missing}")

    if not args.skip_quick_calls:
        print("Running deterministic quick tool calls...")
        calls = [
            ("validate_formula_verifier", {"formula": "Fe2O3"}),
            ("balance_reaction_thermo", {"reactant_formulas": ["BaCO3", "TiO2"], "product_formulas": ["BaTiO3", "CO2"]}),
            ("canonicalize_smiles_rdkit", {"smiles": "CCO"}),
            ("select_tools_for_task", {"task": "synthesis_planning", "profile": "main", "max_tools": 8}),
        ]
        for name, arguments in calls:
            server_url = server_by_tool.get(name)
            if not server_url:
                raise SystemExit(f"Quick-call tool is not present in supplied configs: {name}")
            text = await call_tool(server_url, name, arguments, args.timeout)
            print(json.dumps({"server_url": server_url, "tool": name, "result_excerpt": text[:700]}, ensure_ascii=False))

    print("Mat-MCP stack check passed.")


if __name__ == "__main__":
    asyncio.run(main())
