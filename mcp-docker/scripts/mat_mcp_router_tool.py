"""Router-style verl native tools for Mat-MCP.

This module avoids injecting every Mat-MCP tool schema into the initial
tool-agent prompt.  The agent first lists/selects/describes tools from a compact
catalog, then executes the chosen backend tool through a generic executor.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


ROUTER_TOOL_NAMES = {
    "mat_mcp_list_tool_domains",
    "mat_mcp_select_tools",
    "mat_mcp_describe_tools",
    "mat_mcp_call_tool",
}


DOMAIN_KEYWORDS = {
    "structure": {"structure", "cif", "symmetry", "wyckoff", "slab", "surface", "geometry", "coordination"},
    "property": {"property", "band", "gap", "energy", "formation", "bulk", "modulus", "density", "magnetic"},
    "stability": {"stability", "stable", "hull", "phase", "thermodynamic", "reaction"},
    "synthesis": {"synthesis", "precursor", "route", "reaction", "solvent", "hazard", "condition", "sinter"},
    "characterization": {"xrd", "peak", "pattern", "characterization", "phase", "purity"},
    "defect_interface": {"defect", "dopant", "vacancy", "interstitial", "interface", "heterostructure", "diffusion"},
    "evidence": {"literature", "paper", "doi", "evidence", "search", "citation", "pubchem"},
    "registry": {"tool", "registry", "domain", "policy"},
}


def _json_text(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False)


def _tokens(text: str) -> set[str]:
    return {tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if len(tok) >= 2}


def _parameter_summary(parameters: dict[str, Any]) -> list[dict[str, Any]]:
    properties = parameters.get("properties") if isinstance(parameters, dict) else {}
    required = set(parameters.get("required") or []) if isinstance(parameters, dict) else set()
    if not isinstance(properties, dict):
        return []
    out = []
    for name, spec in properties.items():
        if not isinstance(spec, dict):
            spec = {"type": "string"}
        out.append(
            {
                "name": name,
                "type": spec.get("type", "string"),
                "required": name in required,
                "description": spec.get("description", ""),
            }
        )
    return out


class MatMCPRouterTool(BaseTool):
    """Small router/executor tool layer over the full Mat-MCP catalog."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.catalog_path = Path(str(config["catalog_path"]))
        self.timeout = float(config.get("timeout", 120))
        self.max_response_chars = int(config.get("max_response_chars", 6000))
        self.catalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        self.tools: dict[str, dict[str, Any]] = {item["name"]: item for item in self.catalog.get("tools", [])}

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            if self.name == "mat_mcp_list_tool_domains":
                text = self._list_tool_domains()
            elif self.name == "mat_mcp_select_tools":
                text = self._select_tools(parameters or {})
            elif self.name == "mat_mcp_describe_tools":
                text = self._describe_tools(parameters or {})
            elif self.name == "mat_mcp_call_tool":
                text = await self._call_tool(parameters or {})
            else:
                text = _json_text({"error": f"Unknown Mat-MCP router tool: {self.name}"})
            return ToolResponse(text=text), 0.0, {"status": "ok", "router_tool": self.name}
        except Exception as exc:
            return (
                ToolResponse(text=_json_text({"error": str(exc), "router_tool": self.name})),
                0.0,
                {"status": "error", "router_tool": self.name, "api_request_error": str(exc)},
            )

    async def release(self, instance_id: str, **kwargs) -> None:
        return None

    def _list_tool_domains(self) -> str:
        domains: dict[str, list[str]] = {}
        for item in self.tools.values():
            domains.setdefault(item.get("domain", "other"), []).append(item["name"])
        payload = {
            "usage": [
                "Call mat_mcp_select_tools with the user task before executing domain tools.",
                "Call mat_mcp_describe_tools for selected tool schemas.",
                "Call mat_mcp_call_tool with tool_name and arguments to execute a selected tool.",
            ],
            "domains": [
                {"domain": domain, "count": len(names), "examples": names[:8]}
                for domain, names in sorted(domains.items())
            ],
        }
        return _json_text(payload)

    def _select_tools(self, parameters: dict[str, Any]) -> str:
        task = str(parameters.get("task") or parameters.get("query") or "")
        max_tools = int(parameters.get("max_tools") or 8)
        requested_domain = str(parameters.get("domain") or "").strip().lower()
        task_tokens = _tokens(task)

        scored = []
        for item in self.tools.values():
            domain = item.get("domain", "other")
            if requested_domain and domain != requested_domain:
                continue
            haystack = " ".join(
                [
                    item["name"],
                    item.get("description", ""),
                    domain,
                    " ".join(str(p.get("name", "")) for p in item.get("parameters", [])),
                ]
            )
            item_tokens = _tokens(haystack)
            score = len(task_tokens & item_tokens)
            score += len(task_tokens & DOMAIN_KEYWORDS.get(domain, set())) * 2
            if score > 0:
                scored.append((score, item["name"], item))

        if not scored:
            scored = [(0, item["name"], item) for item in self.tools.values() if not requested_domain or item.get("domain") == requested_domain]

        selected = [item for _, _, item in sorted(scored, key=lambda x: (-x[0], x[1]))[:max_tools]]
        payload = {
            "task": task,
            "selected_tools": [
                {
                    "name": item["name"],
                    "domain": item.get("domain", "other"),
                    "description": item.get("description", ""),
                    "parameters": item.get("parameters", []),
                }
                for item in selected
            ],
            "next_step": "Call mat_mcp_describe_tools for any selected tools you may use, then mat_mcp_call_tool.",
        }
        return _json_text(payload)

    def _describe_tools(self, parameters: dict[str, Any]) -> str:
        names = parameters.get("tool_names") or parameters.get("tools") or []
        if isinstance(names, str):
            names = [name.strip() for name in names.split(",") if name.strip()]
        if not isinstance(names, list):
            names = []
        descriptions = []
        missing = []
        for name in names[:12]:
            item = self.tools.get(str(name))
            if not item:
                missing.append(str(name))
                continue
            descriptions.append(
                {
                    "name": item["name"],
                    "domain": item.get("domain", "other"),
                    "description": item.get("description", ""),
                    "server_url": item.get("server_url", ""),
                    "parameters": item.get("parameters", []),
                    "required": item.get("required", []),
                }
            )
        return _json_text({"tools": descriptions, "missing": missing})

    async def _call_tool(self, parameters: dict[str, Any]) -> str:
        tool_name = str(parameters.get("tool_name") or "").strip()
        arguments = parameters.get("arguments") or {}
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a JSON object")
        item = self.tools.get(tool_name)
        if not item:
            raise ValueError(f"Unknown Mat-MCP tool: {tool_name}")

        async with sse_client(item["server_url"], timeout=self.timeout, sse_read_timeout=self.timeout) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments)

        text_parts = []
        for part in result.content or []:
            if getattr(part, "type", None) == "text":
                text_parts.append(str(getattr(part, "text", "")))
            else:
                try:
                    text_parts.append(json.dumps(part.model_dump(), ensure_ascii=False))
                except Exception:
                    text_parts.append(str(part))
        text = "\n".join(part for part in text_parts if part)
        if len(text) > self.max_response_chars:
            text = text[: self.max_response_chars] + "\n...[truncated]"
        return _json_text({"tool_name": tool_name, "arguments": arguments, "response": text})
