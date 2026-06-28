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
    check_novelty_against_mp_db_router,
    compare_materials_across_databases_db_router,
    fetch_mp_structure_db_router,
    optimade_structure_search_db_router,
    search_materials_by_chemsys_db_router,
    search_materials_by_formula_db_router,
)


logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")), format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger(__name__)
app = Server("materials-db-router-mcp")
TOOLS = get_all_tools(globals())


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
    if name not in TOOLS:
        return [types.TextContent(type="text", text=f"Unknown tool: {name}")]
    start = time.time()
    try:
        result = await asyncio.wait_for(
            TOOLS[name](**arguments),
            timeout=float(os.getenv("TOOL_CALL_TIMEOUT", "120")),
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
@click.option("--port", default=lambda: int(os.getenv("SERVER_PORT", "5677")))
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
            return JSONResponse({"service": "materials-db-router-mcp", "status": "running", "tools": list(TOOLS)})

        async def handle_health(request):
            return JSONResponse({
                "status": "healthy",
                "service": "materials-db-router-mcp",
                "timestamp": str(datetime.now()),
                "tools_count": len(TOOLS),
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
    print(f"materials-db-router-mcp tools: {list(TOOLS.keys())}")
    main()
