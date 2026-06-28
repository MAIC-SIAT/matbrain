import functools
import inspect
from typing import Any, Callable, Dict


def handle_tool_error(func: Callable) -> Callable:
    if inspect.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as exc:
                return {"ok": False, "error": str(exc)}
        return async_wrapper

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
    return wrapper


def llm_tool(name: str, description: str):
    def decorator(func: Callable) -> Callable:
        sig = inspect.signature(func)
        wrapped = handle_tool_error(func)
        wrapped._tool_name = name
        wrapped._tool_description = description
        wrapped._tool_signature = sig
        return wrapped
    return decorator


def _schema_type(annotation: Any) -> str:
    if annotation == int:
        return "integer"
    if annotation == float:
        return "number"
    if annotation == bool:
        return "boolean"
    origin = getattr(annotation, "__origin__", None)
    if origin == list:
        return "array"
    if origin == dict:
        return "object"
    return "string"


def get_tool_schema(func: Callable) -> Dict[str, Any]:
    sig = func._tool_signature
    properties = {}
    required = []
    for name, param in sig.parameters.items():
        properties[name] = {
            "type": _schema_type(param.annotation),
            "description": f"parameter {name}",
        }
        if param.default == inspect.Parameter.empty:
            required.append(name)
    return {
        "function": {
            "name": func._tool_name,
            "description": func._tool_description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }
    }


def get_all_tools(module_globals: Dict[str, Any]) -> Dict[str, Callable]:
    return {
        obj._tool_name: obj
        for obj in module_globals.values()
        if callable(obj) and hasattr(obj, "_tool_name")
    }
