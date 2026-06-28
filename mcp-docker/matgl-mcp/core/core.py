"""MatGL MCP 核心模块

包含 LLM 工具装饰器和统一的错误处理机制，可复用于其他 MCP 项目。
"""

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Union


def handle_tool_error(func: Callable) -> Callable:
    """
    统一的工具函数错误处理装饰器

    将所有异常转换为统一格式的错误信息：Error：{异常信息}

    Args:
        func: 要包装的函数

    Returns:
        包装后的函数
    """
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                return f"Error：{str(e)}"
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                return f"Error：{str(e)}"
        return sync_wrapper


def llm_tool(name: str, description: str):
    """
    LLM工具函数装饰器

    自动为工具函数添加错误处理和元数据信息

    Args:
        name: 工具函数名称
        description: 工具函数描述
    """
    def decorator(func: Callable) -> Callable:
        # 保存原始函数信息
        func._tool_name = name
        func._tool_description = description

        # 获取函数签名
        sig = inspect.signature(func)
        func._tool_signature = sig

        # 应用错误处理装饰器
        error_handled_func = handle_tool_error(func)

        # 将工具信息附加到包装器
        error_handled_func._tool_name = name
        error_handled_func._tool_description = description
        error_handled_func._tool_signature = sig

        return error_handled_func

    return decorator


def get_tool_schema(func: Callable) -> Dict[str, Any]:
    """
    从装饰的函数生成工具模式

    Args:
        func: 被装饰的函数

    Returns:
        工具模式字典
    """
    if not hasattr(func, '_tool_name'):
        raise ValueError(f"函数 {func.__name__} 没有使用 @llm_tool 装饰器")

    # 获取函数签名
    sig = func._tool_signature

    # 构建参数模式
    properties = {}
    required = []

    for param_name, param in sig.parameters.items():
        # 获取参数类型
        param_type = "string"  # 默认类型
        if param.annotation != inspect.Parameter.empty:
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif hasattr(param.annotation, '__origin__'):
                # 处理泛型类型如 List, Dict 等
                if param.annotation.__origin__ == list:
                    param_type = "array"
                elif param.annotation.__origin__ == dict:
                    param_type = "object"

        # 构建参数属性
        prop = {
            "type": param_type,
            "description": f"参数 {param_name}"
        }

        properties[param_name] = prop

        # 检查是否为必需参数
        if param.default == inspect.Parameter.empty:
            required.append(param_name)

    # 构建完整的工具模式
    schema = {
        "function": {
            "name": func._tool_name,
            "description": func._tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

    return schema


def get_all_tool_schemas(module_globals: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    获取模块中所有工具函数的模式

    Args:
        module_globals: 模块的全局变量字典

    Returns:
        工具模式列表
    """
    schemas = []

    for name, obj in module_globals.items():
        if callable(obj) and hasattr(obj, '_tool_name'):
            try:
                schema = get_tool_schema(obj)
                schemas.append(schema)
            except Exception as e:
                print(f"获取工具 {name} 的模式时出错: {e}")

    return schemas


def get_all_tools(module_globals: Dict[str, Any]) -> Dict[str, Callable]:
    """
    获取模块中所有工具函数

    Args:
        module_globals: 模块的全局变量字典

    Returns:
        工具函数字典
    """
    tools = {}

    for name, obj in module_globals.items():
        if callable(obj) and hasattr(obj, '_tool_name'):
            tools[obj._tool_name] = obj

    return tools
