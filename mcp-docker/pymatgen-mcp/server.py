"""PyMatGen-MCP服务器主文件"""

import anyio
import asyncio
import click
import json
import logging
import os
import sys
import time
import traceback
import base64
import re
from datetime import datetime
from typing import Any, Dict, List
from core import get_all_tools, get_tool_schema
import mcp.types as types
from mcp.server.lowlevel import Server
from starlette.responses import Response
from config import server_config, pymatgen_config

# 添加路径前缀配置
PATH_PREFIX = os.getenv('MCP_PATH_PREFIX', '/pymatgen')  # 默认使用 /pymatgen 前缀

# 导入工具函数模块，使工具函数在当前模块的全局命名空间中可用
from tools import (analyze_thermodynamic_stability_pymatgen,
                   estimate_energy_above_hull_pymatgen,
                   check_chemical_formula_valence_pymatgen,
                   check_structure_atomic_geometry_pymatgen,
                   simulate_xrd_pattern_pymatgen,
                   analyze_thermodynamic_stability_with_phase_diagram_pymatgen,
                   substitute_elements_in_structure_pymatgen,
                   analyze_cif_structure_pymatgen,
                   compare_structures_pymatgen,
                   score_generated_cif_against_target_pymatgen,
                   apply_vegards_law_refinement_pymatgen,
                   enumerate_fractional_occupancy_structure_pymatgen,
                   build_bulk_supercell_or_slab_pymatgen,
                   standardize_structure_pymatgen,
                   extract_symmetry_and_wyckoff_pymatgen,
                   estimate_oxidation_states_pymatgen,
                   compute_bond_valence_pymatgen,
                   analyze_coordination_environment_pymatgen,
                   perturb_lattice_or_positions_pymatgen,
                   convert_cif_to_vasp_inputs_pymatgen,
                   parse_vasp_outputs_pymatgen,
                   generate_surface_slabs_pymatgen,
                   find_adsorption_sites_pymatgen,
                   place_adsorbate_on_slab_pymatgen,
                   generate_nrr_intermediates_on_surface_pymatgen,
                   analyze_collinear_magnetism_pymatgen,
                   suggest_initial_magmoms_pymatgen,
                   enumerate_magnetic_orderings_pymatgen)

# 配置日志 - 简化格式，只显示消息内容
logging.basicConfig(
    level=getattr(logging, server_config.LOG_LEVEL),
    format="%(message)s",  # 只显示消息内容，去除时间戳和日志级别
    stream=sys.stdout,  # 明确指定输出到 stdout
    force=True  # 强制重新配置，避免被其他库覆盖
)
logger = logging.getLogger(__name__)

# 创建MCP服务器实例
app = Server("pymatgen-mcp")

# 工具函数映射 name-> function
# 这里使用get_all_tools()来获取所有工具函数
# 需要传入当前模块的全局变量字典
TOOLS = get_all_tools(globals())


def get_tool_icon(tool_name: str) -> str:
    """根据工具名称返回合适的图标"""
    if "thermodynamic" in tool_name.lower() or "stability" in tool_name.lower():
        return "⚖️"
    elif "valence" in tool_name.lower() or "formula" in tool_name.lower():
        return "🧪"
    elif "geometry" in tool_name.lower() or "structure" in tool_name.lower():
        return "🔬"
    elif "crystal" in tool_name.lower():
        return "💎"
    elif "property" in tool_name.lower() or "properties" in tool_name.lower():
        return "📊"
    elif "query" in tool_name.lower() or "search" in tool_name.lower():
        return "🔍"
    elif "predict" in tool_name.lower():
        return "⚡"
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


def is_base64_image(data: str) -> bool:
    """
    检查字符串是否为base64编码的图像数据

    Args:
        data: 要检查的字符串

    Returns:
        bool: 如果是base64图像数据返回True，否则返回False
    """
    if not isinstance(data, str):
        return False

    # 移除可能的数据URL前缀
    if data.startswith('data:image/'):
        data = data.split(',', 1)[-1]

    # 检查是否为有效的base64字符串
    try:
        # base64字符串应该只包含A-Z, a-z, 0-9, +, /, = 字符
        if not re.match(r'^[A-Za-z0-9+/]*={0,2}$', data):
            return False

        # 尝试解码
        decoded = base64.b64decode(data)

        # 检查解码后的数据长度（图像数据通常比较大）
        if len(decoded) < 100:  # 太小的数据不太可能是图像
            return False

        # 检查是否有常见的图像文件头
        # JPEG: FF D8 FF
        # PNG: 89 50 4E 47
        # GIF: 47 49 46 38
        if (decoded.startswith(b'\xff\xd8\xff') or  # JPEG
            decoded.startswith(b'\x89PNG') or        # PNG
            decoded.startswith(b'GIF8')):            # GIF
            return True

        # 如果数据足够大且是有效的base64，也认为可能是图像
        return len(decoded) > 1000

    except Exception:
        return False


def extract_base64_from_markdown(text: str) -> str:
    """
    从Markdown文本中提取base64图像数据

    Args:
        text: 包含base64图像的Markdown文本

    Returns:
        str: 提取的base64图像数据，如果没有找到则返回空字符串
    """
    if not isinstance(text, str):
        return ""

    # 查找Base64编码部分的模式
    # 匹配 ```\n{base64_data}\n``` 格式
    base64_pattern = r'```\n([A-Za-z0-9+/=\n\r\s]+)\n```'
    matches = re.findall(base64_pattern, text)

    for match in matches:
        # 清理换行符和空格
        cleaned_data = re.sub(r'\s+', '', match)
        if is_base64_image(cleaned_data):
            return cleaned_data

    # 如果没有找到代码块格式，尝试查找直接的base64数据
    # 查找长的base64字符串（通常图像数据会很长）
    direct_pattern = r'([A-Za-z0-9+/]{500,}={0,2})'
    matches = re.findall(direct_pattern, text)

    for match in matches:
        if is_base64_image(match):
            return match

    return ""


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
            "📤 返回结果类型:"
        ]

        # 根据结果类型添加不同的描述
        if isinstance(result, str):
            # 检查是否包含base64图像数据
            base64_image = extract_base64_from_markdown(result)
            if base64_image:
                success_content.append("    📷 包含Base64编码图像数据的Markdown文本")
                success_content.append(f"    📏 图像数据长度: {len(base64_image)} 字符")
                success_content.append(f"    📄 文本长度: {len(result)} 字符")
                # 只显示文本的前200个字符（不包含base64部分）
                text_without_base64 = result.replace(base64_image, "[BASE64_IMAGE_DATA]")
                preview = text_without_base64[:200]
                if len(text_without_base64) > 200:
                    preview += "..."
                success_content.append(f"    📝 内容预览: {preview}")
            elif is_base64_image(result):
                success_content.append("    📷 Base64编码图像数据")
                success_content.append(f"    📏 数据长度: {len(result)} 字符")
            else:
                success_content.append("    📝 文本数据")
                # 只显示结果的前200个字符
                result_preview = str(result)[:200]
                if len(str(result)) > 200:
                    result_preview += "..."
                success_content.append(f"    📄 内容预览: {result_preview}")
        else:
            success_content.append("    📊 结构化数据")
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
        工具函数的执行结果，根据工具类型返回不同的内容类型
    """
    try:
        # 使用配置的超时时间调用工具函数
        result = await asyncio.wait_for(
            call_tool_function(name, arguments),
            timeout=server_config.TOOL_CALL_TIMEOUT
        )

        # 根据工具函数名称和返回结果类型决定返回格式
        if isinstance(result, str):
            # 检查是否包含base64图像数据
            base64_image = extract_base64_from_markdown(result)
            if base64_image:
                # 对于包含相图的工具，返回图像和文本内容
                # 创建不包含base64数据的文本版本
                text_without_base64 = result.replace(base64_image, "[图像数据已单独提供]")

                return [
                    types.TextContent(type="text", text=text_without_base64),
                    types.ImageContent(
                        type="image",
                        data=base64_image,
                        mimeType="image/png"
                    )
                ]
            elif is_base64_image(result):
                # 纯base64图像数据，返回ImageContent
                return [types.ImageContent(
                    type="image",
                    data=result,
                    mimeType="image/png"
                )]
            else:
                # 普通文本数据，返回TextContent
                return [types.TextContent(type="text", text=result)]
        else:
            # 对于结构化数据，转换为JSON字符串
            result_str = json.dumps(result, ensure_ascii=False, indent=2)
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
@click.option("--path-prefix", default="/pymatgen", help="URL路径前缀，如 /pymatgen")
def main(port: int, transport: str = 'sse', path_prefix: str = "/pymatgen") -> int:
    """
    PyMatGen-MCP服务器主函数

    Args:
        port: SSE传输的端口号
        transport: 传输类型，stdio或sse
        path_prefix: URL路径前缀

    Returns:
        退出码
    """
    # 如果没有通过命令行参数指定，使用环境变量
    if not path_prefix:
        path_prefix = PATH_PREFIX

    # 验证配置
    server_valid = server_config.validate_config()
    pymatgen_valid = pymatgen_config.validate_config()

    if not server_valid:
        logger.warning("服务器配置验证失败，但服务器将继续启动")

    if not pymatgen_valid:
        logger.warning("PyMatGen配置验证失败，某些功能可能受限")

    logger.info(f"启动PyMatGen-MCP服务器，传输类型: {transport}, 端口: {port}, 路径前缀: {path_prefix}")

    if transport == "sse":
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Mount, Route
        from starlette.responses import JSONResponse

        # 关键修改：使用带前缀的messages路径
        messages_path = f"{path_prefix}/messages/" if path_prefix else "/messages/"
        sse = SseServerTransport(messages_path)

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

        async def handle_root(request):
            """根路径处理 - 返回服务状态信息"""
            return JSONResponse({
                "service": "pymatgen-mcp",
                "status": "running",
                "version": "1.0.0",
                "path_prefix": path_prefix,
                "endpoints": {
                    "sse": f"{path_prefix}/sse" if path_prefix else "/sse",
                    "messages": messages_path,
                    "health": f"{path_prefix}/health" if path_prefix else "/health"
                },
                "description": "PyMatGen MCP Server - Materials Science Analysis Tools",
                "tools": list(TOOLS.keys()),
                "capabilities": [
                    "热力学稳定性分析",
                    "化学式价态验证",
                    "原子几何检查"
                ],
                "dependencies": {
                    "pymatgen": "PyMatGen材料科学库",
                    "materials_project": "Materials Project数据库集成",
                    "matgl_mcp": server_config.MATGL_MCP_SSE_URL
                }
            })

        async def handle_health(request):
            """健康检查端点"""
            # 检查配置状态
            config_status = {
                "server_config": server_valid,
                "pymatgen_config": pymatgen_valid,
                "mp_api_key": bool(pymatgen_config.MP_API_KEY),
                "matgl_connection": bool(server_config.MATGL_MCP_SSE_URL)
            }

            overall_health = all(config_status.values())

            return JSONResponse({
                "status": "healthy" if overall_health else "degraded",
                "service": "pymatgen-mcp",
                "timestamp": str(datetime.now()),
                "tools_count": len(TOOLS),
                "path_prefix": path_prefix,
                "config_status": config_status,
                "available_tools": list(TOOLS.keys())
            })

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
    print("🚀 PyMatGen-MCP 服务器启动")
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

    # 打印配置信息
    server_config.print_config()
    pymatgen_config.print_config()

    main()
