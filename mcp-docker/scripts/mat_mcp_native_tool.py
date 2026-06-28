"""verl 0.9 native tool wrapper for remote Mat-MCP SSE tools."""

from __future__ import annotations

import json
from typing import Any, Optional
from uuid import uuid4

from mcp import ClientSession
from mcp.client.sse import sse_client
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


class MatMCPTool(BaseTool):
    """Expose one Mat-MCP SSE tool as a verl native BaseTool.

    verl 0.9 currently supports native/function tools in the tool registry.  This
    wrapper keeps Mat-MCP as the actual execution backend while presenting each
    MCP function as a native verl tool.
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.server_url = str(config["server_url"])
        self.timeout = float(config.get("timeout", 120))

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        return instance_id or str(uuid4()), ToolResponse()

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        try:
            async with sse_client(
                self.server_url,
                timeout=self.timeout,
                sse_read_timeout=self.timeout,
            ) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, parameters or {})
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
            return ToolResponse(text=text), 0.0, {
                "status": "ok",
                "server_url": self.server_url,
                "tool_name": self.name,
            }
        except Exception as exc:
            message = f"Error executing Mat-MCP tool '{self.name}' at {self.server_url}: {exc}"
            return ToolResponse(text=message), 0.0, {
                "status": "error",
                "server_url": self.server_url,
                "tool_name": self.name,
                "api_request_error": str(exc),
            }

    async def release(self, instance_id: str, **kwargs) -> None:
        return None
