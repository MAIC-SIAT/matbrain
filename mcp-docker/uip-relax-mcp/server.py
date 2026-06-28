import asyncio
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List

import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server

from core import get_all_tools, get_tool_schema
from tools import (
    calculate_discovery_reward,
    compare_uip_relaxations,
    uip_force_stability_score,
    uip_relax_structure,
    uip_single_point_energy,
)


logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")), format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
app = Server("uip-relax-mcp")
TOOLS = get_all_tools(globals())


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    if name not in TOOLS:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    start = time.time()
    try:
        result = await asyncio.wait_for(
            TOOLS[name](**arguments),
            timeout=float(os.getenv("TOOL_CALL_TIMEOUT", "600")),
        )
        logger.info(f"{name} finished in {time.time() - start:.2f}s")
        if not isinstance(result, str):
            result = json.dumps(result, ensure_ascii=False, indent=2)
        return [types.TextContent(type="text", text=result)]
    except Exception as exc:
        text = f"Error calling {name}: {exc}\n{traceback.format_exc()}"
        logger.error(text)
        return [types.TextContent(type="text", text=text)]


@app.list_tools()
async def list_tools() -> List[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=get_tool_schema(func)["function"]["description"],
            inputSchema=get_tool_schema(func)["function"]["parameters"],
        )
        for name, func in TOOLS.items()
    ]


@click.command()
@click.option("--port", default=lambda: int(os.getenv("SERVER_PORT", "5679")))
@click.option("--transport", type=click.Choice(["stdio", "sse"]), default="sse")
def main(port: int, transport: str) -> int:
    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse, Response
        from starlette.routing import Mount, Route

        sse = SseServerTransport("/messages/")

        async def handle_sse(request):
            try:
                async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                    await app.run(streams[0], streams[1], app.create_initialization_options())
                return Response()
            except Exception as e:
                logger.error(f"SSE连接处理失败: {e}")
                return Response(status_code=500, content=f"SSE connection failed: {str(e)}")

        async def handle_root(request):
            return JSONResponse({"service": "uip-relax-mcp", "status": "running", "tools": list(TOOLS)})

        async def handle_health(request):
            return JSONResponse({
                "status": "healthy",
                "service": "uip-relax-mcp",
                "timestamp": str(datetime.now()),
                "tools_count": len(TOOLS),
                "device": os.getenv("UIP_DEVICE", "cuda"),
                "mace_model": os.getenv("MACE_MP_MODEL", "medium"),
            })

        starlette_app = Starlette(routes=[
            Route("/", endpoint=handle_root),
            Route("/health", endpoint=handle_health),
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message),
        ])
        import uvicorn
        uvicorn.run(starlette_app, host=os.getenv("SERVER_HOST", "0.0.0.0"), port=port, log_config=None)
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(streams[0], streams[1], app.create_initialization_options())

        anyio.run(arun)
    return 0


if __name__ == "__main__":
    print(f"uip-relax-mcp tools: {list(TOOLS.keys())}")
    main()
