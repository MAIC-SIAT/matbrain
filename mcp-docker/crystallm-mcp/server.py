"""CRYSTALLM-MCP服务器主文件"""

import anyio
import asyncio
import click
import json
import logging
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List
from core import get_all_tools,get_tool_schema
import mcp.types as types
from mcp.server.lowlevel import Server

from config import crystallm_config, server_config
# 导入工具函数模块，使工具函数在当前模块的全局命名空间中可用
from tools import (generate_crystal_structures_crystallm)


# 配置日志 - 简化格式，只显示消息内容
logging.basicConfig(
    level=getattr(logging, server_config.LOG_LEVEL),
    format="%(message)s",  # 只显示消息内容，去除时间戳和日志级别
    stream=sys.stdout,  # 明确指定输出到 stdout
    force=True  # 强制重新配置，避免被其他库覆盖
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
app = Server("crystallm-mcp")

# 工具函数映射 name-> function
# 这里使用get_all_tools()来获取所有工具函数
# 需要传入当前模块的全局变量字典
TOOLS = get_all_tools(globals())


def get_tool_icon(tool_name: str) -> str:
    """根据工具名称返回合适的图标"""
    if "crystal" in tool_name.lower() or "structure" in tool_name.lower():
        return "🧪"
    elif "generate" in tool_name.lower() or "generation" in tool_name.lower():
        return "⚡"
    elif "llm" in tool_name.lower():
        return "🤖"
    elif "analyze" in tool_name.lower():
        return "🔬"
    else:
        return "🛠️"


def format_json_content(data: Any, indent: int = 4) -> List[str]:
    """美化JSON内容显示"""
    if isinstance(data, (dict, list)):
        json_str = json.dumps(data, ensure_ascii=False, indent=2)
        lines = json_str.split('\n')
        # 为每行添加适当的缩进
        formatted_lines = []
        for line in lines:
            if line.strip():
                formatted_lines.append(" " * indent + line)
            else:
                formatted_lines.append("")
        return formatted_lines
    else:
        return [" " * indent + str(data)]


def log_formatted_message(title: str, content_lines: List[str], level: str = "info"):
    """输出格式化的消息（无边框版本）"""
    log_func = getattr(logger, level.lower())

    # 输出标题
    log_func(title)
    log_func("─" * 60)

    # 输出内容
    for content in content_lines:
        log_func(content)


async def call_tool_function(func_name: str, arguments: Dict[str, Any]) -> Any:
    """
    调用工具函数

    Args:
        func_name: 工具函数名称
        arguments: 工具函数参数

    Returns:
        工具函数的执行结果
    """
    if func_name not in TOOLS:
        raise ValueError(f"未知的工具函数: {func_name}")

    tool_func = TOOLS[func_name]
    tool_icon = get_tool_icon(func_name)
    start_time = time.time()

    # 美化的工具调用开始输出
    content_lines = [
        f"{tool_icon} 工具名称: {func_name}",
        f"⏰ 调用时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "📥 输入参数:"
    ]

    # 添加格式化的参数内容
    json_lines = format_json_content(arguments)
    content_lines.extend(json_lines)

    log_formatted_message("🚀 工具调用开始", content_lines)

    try:
        # 调用工具函数
        if asyncio.iscoroutinefunction(tool_func):
            result = await tool_func(**arguments)
        else:
            result = tool_func(**arguments)

        # 计算执行时间
        execution_time = time.time() - start_time

        # 美化的执行成功输出
        success_content = [
            f"{tool_icon} 工具名称: {func_name}",
            f"⏱️  执行时长: {execution_time:.2f}秒",
            "📤 返回结果:"
        ]

        # 添加格式化的结果内容
        result_lines = format_json_content(result)
        success_content.extend(result_lines)

        log_formatted_message("✅ 工具执行成功", success_content)

        return result

    except Exception as e:
        # 计算执行时间
        execution_time = time.time() - start_time

        # 美化的执行失败输出
        error_content = [
            f"{tool_icon} 工具名称: {func_name}",
            f"⏱️  执行时长: {execution_time:.2f}秒",
            f"🚫 错误类型: {type(e).__name__}",
            f"💬 错误信息: {str(e)}",
            "📋 错误堆栈:"
        ]

        # 添加格式化的堆栈信息
        stack_lines = traceback.format_exc().split('\n')
        for line in stack_lines:
            if line.strip():
                error_content.append(f"    {line}")

        log_formatted_message("❌ 工具执行失败", error_content, "error")
        raise


@app.call_tool()
async def call_tool(
    name: str, arguments: Dict[str, Any]
) -> List[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """
    MCP工具调用处理器

    Args:
        name: 工具函数名称
        arguments: 工具函数参数

    Returns:
        工具函数的执行结果
    """
    try:
        # 使用配置的超时时间调用工具函数
        result = await asyncio.wait_for(
            call_tool_function(name, arguments),
            timeout=server_config.TOOL_CALL_TIMEOUT
        )

        # 将结果转换为字符串
        if isinstance(result, (dict, list)):
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
        else:
            result_str = str(result)

        return [types.TextContent(type="text", text=result_str)]

    except asyncio.TimeoutError:
        error_msg = f"工具函数 {name} 调用超时 ({server_config.TOOL_CALL_TIMEOUT}秒)"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]
    except Exception as e:
        error_msg = f"调用工具函数 {name} 时出错: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        return [types.TextContent(type="text", text=error_msg)]


@app.list_tools()
async def list_tools() -> List[types.Tool]:
    """
    列出所有可用的工具函数

    Returns:
        工具函数列表
    """
    tools = []

    for func_name, tool_func in TOOLS.items():
        try:
            # 获取工具模式
            schema = get_tool_schema(tool_func)

            # 创建工具
            tool = types.Tool(
                name=func_name,
                description=schema["function"]["description"],
                inputSchema=schema["function"]["parameters"],
            )

            tools.append(tool)

        except Exception as e:
            logger.error(f"获取工具 {func_name} 的模式时出错: {e}")

    # 美化的工具列表输出
    content_lines = []

    for i, tool in enumerate(tools, 1):
        tool_icon = get_tool_icon(tool.name)
        content_lines.append(f"{i:2d}. {tool_icon} {tool.name}")
        content_lines.append(f"    📝 {tool.description}")

        # 如果不是最后一个工具，添加分隔线
        if i < len(tools):
            content_lines.append("─" * 58)

    log_formatted_message(f"🔧 可用工具列表 ({len(tools)}个)", content_lines)

    return tools


@click.command()
@click.option("--port", default=server_config.SERVER_PORT, help="SSE传输端口")
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="sse",
    help="传输类型",
)
def main(port: int, transport: str = 'sse') -> int:
    """
    CrystaLLM-MCP服务器主函数

    Args:
        port: SSE传输的端口号
        transport: 传输类型，stdio或sse

    Returns:
        退出码
    """
    # 验证配置
    crystallm_valid = crystallm_config.validate_config()
    server_valid = server_config.validate_config()

    if not crystallm_valid:
        logger.warning("CrystaLLM 配置验证失败，但服务器将继续启动")
    if not server_valid:
        logger.warning("服务器配置验证失败，但服务器将继续启动")

    logger.info(f"启动CRYSTALLM-MCP服务器，传输类型: {transport}, 端口: {port}")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse, Response

        sse = SseServerTransport("/messages/")

        async def handle_root(request):
            """根路径处理 - 返回服务状态信息"""
            return JSONResponse({
                "service": "crystallm-mcp",
                "status": "running",
                "version": "1.0.0",
                "endpoints": {
                    "sse": "/sse",
                    "messages": "/messages/",
                    "health": "/health"
                },
                "description": "CrystaLLM Crystal Structure Generation MCP Server",
                "tools": list(TOOLS.keys())
            })

        async def handle_health(request):
            """健康检查端点"""
            return JSONResponse({
                "status": "healthy",
                "service": "crystallm-mcp",
                "timestamp": str(datetime.now()),
                "tools_count": len(TOOLS)
            })
        async def handle_sse(request):
            try:
                async with sse.connect_sse(request.scope, request.receive, request._send) as streams:
                    await app.run(streams[0], streams[1], app.create_initialization_options())
                return Response()
            except Exception as e:
                logger.error(f"SSE连接处理失败: {e}")
                return Response(status_code=500, content=f"SSE connection failed: {str(e)}")

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/", endpoint=handle_root),
                Route("/health", endpoint=handle_health),
                Route("/sse", endpoint=handle_sse),
                Mount("/messages/", app=sse.handle_post_message),
            ],
        )

        import uvicorn

        # 配置 uvicorn 日志，确保不覆盖我们的日志配置
        uvicorn.run(
            starlette_app,
            host=server_config.SERVER_HOST,
            port=port,
            log_config=None,  # 禁用 uvicorn 的默认日志配置
            access_log=True,  # 启用访问日志
            use_colors=False,  # 禁用颜色输出，避免在容器中出现乱码
        )
    else:
        from mcp.server.stdio import stdio_server

        async def arun():
            async with stdio_server() as streams:
                await app.run(
                    streams[0], streams[1], app.create_initialization_options()
                )

        anyio.run(arun)

    return 0


if __name__ == "__main__":
    # 美化工具列表输出
    print("\n" + "="*60)
    print("🚀 CrystaLLM-MCP 服务器启动")
    print("="*60)
    print(f"📦 可用工具数量: {len(TOOLS)}")
    print("🔧 可用工具列表:")
    print("-"*60)

    for i, tool_name in enumerate(TOOLS.keys(), 1):
        print(f"  {i:2d}. 🧪  {tool_name}")

    print("-"*60)
    print("✅ 工具加载完成，正在启动服务器...")
    print("="*60 + "\n")

    main()
