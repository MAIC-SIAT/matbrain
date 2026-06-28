"""Multi-server MCP client over SSE.

Workflow:
- On startup, `discover()` is called once per server. For each server we open a
  short-lived SSE session, fetch list_tools(), and cache (server_name, tool_name,
  description, inputSchema) in `registry`.
- At dispatch time, `call(tool_name, args)` looks up which server owns the tool,
  opens a fresh SSE session, invokes call_tool(), and returns the text payload.

Per-call sessions match the pattern in `matbrain_backup/mat-mcp/test_mcp_docker.py`
and avoid having to manage long-lived async-context-manager lifetimes inside
LangGraph nodes.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from mcp import ClientSession
from mcp.client.sse import sse_client

logger = logging.getLogger(__name__)

_SSE_OPEN_TIMEOUT = 60
_SSE_READ_TIMEOUT = 60 * 10
_INIT_TIMEOUT = 30


@dataclass
class ToolSpec:
    name: str
    server: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    content: str
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MultiMCPClient:
    """Aggregates multiple MCP SSE endpoints behind a unified tool registry."""

    def __init__(self, endpoints: dict[str, str], call_timeout: int = 480):
        self.endpoints = dict(endpoints)
        self.call_timeout = call_timeout
        self.registry: dict[str, ToolSpec] = {}

    async def discover(self) -> dict[str, ToolSpec]:
        """List tools across every endpoint and cache the registry.

        Servers that fail to respond are logged and skipped — the agent can still
        run with whatever did respond. Duplicate tool names across servers are
        flagged via a warning; the latest one wins.
        """
        results = await asyncio.gather(
            *(self._list_one(name, url) for name, url in self.endpoints.items()),
            return_exceptions=True,
        )
        for (server, _), res in zip(self.endpoints.items(), results):
            if isinstance(res, Exception):
                logger.warning("discover(%s) failed: %s", server, res)
                continue
            for spec in res:
                if spec.name in self.registry:
                    logger.warning(
                        "tool name collision: %s (already on %s, also on %s)",
                        spec.name,
                        self.registry[spec.name].server,
                        spec.server,
                    )
                self.registry[spec.name] = spec
        logger.info("discovered %d MCP tools across %d servers", len(self.registry), len(self.endpoints))
        return self.registry

    async def _list_one(self, server: str, url: str) -> list[ToolSpec]:
        async with sse_client(url, timeout=_SSE_OPEN_TIMEOUT, sse_read_timeout=_SSE_READ_TIMEOUT) as streams:
            async with ClientSession(*streams) as session:
                await asyncio.wait_for(session.initialize(), timeout=_INIT_TIMEOUT)
                res = await asyncio.wait_for(session.list_tools(), timeout=self.call_timeout)
                specs: list[ToolSpec] = []
                for t in res.tools:
                    schema = t.inputSchema if isinstance(t.inputSchema, dict) else {}
                    specs.append(
                        ToolSpec(
                            name=t.name,
                            server=server,
                            description=t.description or "",
                            input_schema=schema,
                        )
                    )
                return specs

    def get(self, tool_name: str) -> ToolSpec | None:
        return self.registry.get(tool_name)

    async def call(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        spec = self.registry.get(tool_name)
        if spec is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                content="",
                error=f"tool '{tool_name}' not in registry",
            )
        url = self.endpoints[spec.server]
        try:
            async with sse_client(url, timeout=_SSE_OPEN_TIMEOUT, sse_read_timeout=_SSE_READ_TIMEOUT) as streams:
                async with ClientSession(*streams) as session:
                    await asyncio.wait_for(session.initialize(), timeout=_INIT_TIMEOUT)
                    res = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=self.call_timeout,
                    )
                    text = ""
                    if res.content:
                        first = res.content[0]
                        text = getattr(first, "text", "") or ""
                    return ToolResult(
                        tool_name=tool_name,
                        success=not getattr(res, "isError", False),
                        content=text,
                        metadata={"server": spec.server},
                    )
        except asyncio.TimeoutError as e:
            return ToolResult(tool_name=tool_name, success=False, content="", error=f"timeout: {e}")
        except Exception as e:
            return ToolResult(tool_name=tool_name, success=False, content="", error=f"{type(e).__name__}: {e}")

    def server_names(self) -> list[str]:
        """Distinct MCP server names in registry order (alphabetical)."""
        return sorted({spec.server for spec in self.registry.values()})

    def tool_names_for_servers(self, servers: set[str]) -> set[str]:
        """Return all tool names belonging to any of the given servers."""
        return {name for name, spec in self.registry.items() if spec.server in servers}

    def render_inventory(self, enabled_servers: set[str] | None = None) -> str:
        """Compact tool block for the Mat-T1 system prompt.

        If `enabled_servers` is None, render every tool. Otherwise render only
        tools whose owning server is in the set — used by per-request tool
        filtering driven by the UI's per-server checkboxes.
        """
        lines: list[str] = []
        for name, spec in sorted(self.registry.items()):
            if enabled_servers is not None and spec.server not in enabled_servers:
                continue
            args = []
            props = spec.input_schema.get("properties", {}) if isinstance(spec.input_schema, dict) else {}
            required = set(spec.input_schema.get("required", []) or []) if isinstance(spec.input_schema, dict) else set()
            for arg_name, arg_schema in props.items():
                t = arg_schema.get("type", "any") if isinstance(arg_schema, dict) else "any"
                tag = f"{arg_name}:{t}"
                if arg_name in required:
                    tag = f"*{tag}"
                args.append(tag)
            desc = (spec.description or "").strip().splitlines()[0][:160] if spec.description else ""
            lines.append(f"- {name}({', '.join(args)})  [{spec.server}] {desc}")
        return "\n".join(lines)
