"""MatterGen-MCP服务器主文件"""

import anyio
import asyncio
import click
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List
from core import get_all_tools, get_tool_schema
import mcp.types as types
from mcp.server.lowlevel import Server
from starlette.responses import Response
from config import mattergen_config, server_config
# 添加路径前缀配置
PATH_PREFIX = os.getenv('MCP_PATH_PREFIX', '/mattergen')  # 默认使用 /mattergen 前缀
# 导入工具函数模块，使工具函数在当前模块的全局命名空间中可用
from tools import (
    generate_material_unconditional_MatterGen,
    generate_material_by_dft_band_gap_MatterGen,
    generate_material_by_chemical_system_MatterGen,
    generate_material_by_space_group_MatterGen,
    generate_material_by_dft_mag_density_MatterGen,
    generate_material_by_dft_mag_density_and_hhi_score_MatterGen,
    generate_material_by_chemical_system_and_energy_above_hull_MatterGen,
)

# 配置日志 - 简化格式，只显示消息内容
logging.basicConfig(
    level=getattr(logging, server_config.LOG_LEVEL),
    format="%(message)s",  # 只显示消息内容，去除时间戳和日志级别
    stream=sys.stdout,  # 明确指定输出到 stdout
    force=True  # 强制重新配置，避免被其他库覆盖
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
app = Server("mattergen-mcp")

# 工具函数映射 name-> function
# 这里使用get_all_tools()来获取所有工具函数
# 需要传入当前模块的全局变量字典
TOOLS = get_all_tools(globals())


def get_tool_icon(tool_name: str) -> str:
    """根据工具名称返回合适的图标"""
    if "material" in tool_name.lower() or "mattergen" in tool_name.lower():
        return "🧪"
    elif "generate" in tool_name.lower() or "generation" in tool_name.lower():
        return "⚡"
    elif "crystal" in tool_name.lower() or "structure" in tool_name.lower():
        return "💎"
    elif "unconditional" in tool_name.lower():
        return "🎲"
    elif "band_gap" in tool_name.lower():
        return "🌈"
    elif "chemical_system" in tool_name.lower():
        return "⚛️"
    elif "space_group" in tool_name.lower():
        return "🔷"
    elif "formation_energy" in tool_name.lower():
        return "⚡"
    elif "bulk_modulus" in tool_name.lower():
        return "💪"
    elif "custom" in tool_name.lower():
        return "🎛️"
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
@click.option("--path-prefix", default="/mattergen", help="URL路径前缀，如 /mattergen")
def main(port: int, transport: str = 'sse', path_prefix: str = "/mattergen") -> int:
    """
    MatterGen-MCP服务器主函数

    Args:
        port: SSE传输的端口号
        transport: 传输类型，stdio或sse
        path_prefix: URL路径前缀

    Returns:
        退出码
    """
    # 如果没有通过命令行参数指定，使用环境变量（None 才回退；空字符串表示明确禁用前缀）
    if path_prefix is None:
        path_prefix = PATH_PREFIX

    # 验证配置
    mattergen_valid = mattergen_config.validate_config()
    server_valid = server_config.validate_config()

    if not mattergen_valid:
        logger.warning("MatterGen 配置验证失败，但服务器将继续启动")
    if not server_valid:
        logger.warning("服务器配置验证失败，但服务器将继续启动")

    logger.info(f"启动MatterGen-MCP服务器，传输类型: {transport}, 端口: {port}, 路径前缀: {path_prefix}")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse

        # 关键修改：使用带前缀的messages路径
        messages_path = f"{path_prefix}/messages/" if path_prefix else "/messages/"
        sse = SseServerTransport(messages_path)

        async def handle_root(request):
            """根路径处理 - 返回服务状态信息"""
            return JSONResponse({
                "service": "mattergen-mcp",
                "status": "running",
                "version": "1.0.0",
                "path_prefix": path_prefix,
                "endpoints": {
                    "sse": f"{path_prefix}/sse" if path_prefix else "/sse",
                    "messages": messages_path,
                    "health": f"{path_prefix}/health" if path_prefix else "/health"
                },
                "description": "MatterGen Crystal Structure Generation MCP Server",
                "tools": list(TOOLS.keys())
            })

        async def handle_health(request):
            """健康检查端点"""
            return JSONResponse({
                "status": "healthy",
                "service": "mattergen-mcp",
                "timestamp": str(datetime.now()),
                "tools_count": len(TOOLS),
                "path_prefix": path_prefix
            })

        async def handle_sse(request):
            """处理 SSE 连接 - 生产环境优化版"""
            try:
                async with sse.connect_sse(
                    request.scope,
                    request.receive,
                    request._send
                ) as streams:
                    await app.run(
                        streams[0], streams[1], app.create_initialization_options()
                    )

                # 返回适当的SSE响应
                return Response(
                    status_code=200,
                    headers={
                        'cache-control': 'no-cache',
                        'connection': 'keep-alive',
                        'access-control-allow-origin': '*',  # 如果需要跨域
                    }
                )
            except Exception as e:
                logger.error(f"SSE连接处理失败: {e}")
                return Response(
                    status_code=500,
                    content=f"SSE connection failed: {str(e)}",
                    headers={'content-type': 'text/plain'}
                )

        starlette_app = Starlette(
            debug=True,
            routes=[
                Route("/", endpoint=handle_root),
                Route(f"{path_prefix}/health" if path_prefix else "/health", endpoint=handle_health),
                Route(f"{path_prefix}/sse" if path_prefix else "/sse", endpoint=handle_sse),
                Mount(messages_path, app=sse.handle_post_message),
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
    print("🚀 MatterGen-MCP 服务器启动")
    print("="*60)
    print(f"📦 可用工具数量: {len(TOOLS)}")
    print(f"🔗 路径前缀: {PATH_PREFIX}")
    print("🔧 可用工具列表:")
    print("-"*60)

    for i, tool_name in enumerate(TOOLS.keys(), 1):
        tool_icon = get_tool_icon(tool_name)
        print(f"  {i:2d}. {tool_icon}  {tool_name}")

    print("-"*60)
    print("✅ 工具加载完成，正在启动服务器...")
    print("="*60 + "\n")

    import torch
    print("====cuda 能否使用:", torch.cuda.is_available())

    main()
