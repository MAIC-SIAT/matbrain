"""Runtime parameter validation built dynamically from each MCP tool's inputSchema.

We convert a JSONSchema object into a Pydantic model via pydantic.create_model
and use it to coerce/validate the arguments Mat-T1 emitted inside <tool_call>.
The goal is to catch obvious type / required-key errors BEFORE dispatching to
the MCP server (per the manuscript's "Runtime Parameter Validation Layer").
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError, create_model

_PY_TYPES: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _jsonschema_type_to_python(schema: dict[str, Any]) -> Any:
    """Best-effort JSONSchema -> Python annotation. Falls back to Any."""
    if not isinstance(schema, dict):
        return Any
    if "anyOf" in schema or "oneOf" in schema:
        return Any
    js_type = schema.get("type")
    if isinstance(js_type, list):
        return Any
    if js_type == "array":
        items = schema.get("items", {})
        inner = _jsonschema_type_to_python(items) if isinstance(items, dict) else Any
        return list[inner] if inner is not Any else list
    if js_type == "object":
        return dict
    return _PY_TYPES.get(js_type, Any)


def model_from_input_schema(tool_name: str, input_schema: dict[str, Any]):
    """Build a Pydantic model for a single MCP tool's inputSchema."""
    if not isinstance(input_schema, dict):
        input_schema = {}
    properties: dict[str, Any] = input_schema.get("properties", {}) or {}
    required: set[str] = set(input_schema.get("required", []) or [])

    fields: dict[str, tuple[Any, Any]] = {}
    for name, prop in properties.items():
        py_type = _jsonschema_type_to_python(prop if isinstance(prop, dict) else {})
        default = ... if name in required else prop.get("default", None) if isinstance(prop, dict) else None
        fields[name] = (py_type, default)

    # Sanitize tool name for class name.
    safe = "".join(c if c.isalnum() else "_" for c in tool_name).strip("_") or "Tool"
    return create_model(f"Args_{safe}", **fields)


def validate_arguments(
    tool_name: str,
    input_schema: dict[str, Any],
    arguments: dict[str, Any],
) -> tuple[dict[str, Any] | None, str | None]:
    """Returns (validated_dict, None) on success or (None, error_message) on failure.

    LLMs frequently pass `null` for optional arguments meaning "use the default".
    JSONSchemas in our MCPs rarely list null as a valid type, so strict Pydantic
    validation would reject those calls. We pre-process the incoming args: any
    None value for a non-required field is dropped, and the MCP server's own
    default for that field will apply at the tool-execution layer.
    """
    required: set[str] = set()
    if isinstance(input_schema, dict):
        required = set(input_schema.get("required", []) or [])
    cleaned = {k: v for k, v in (arguments or {}).items() if not (v is None and k not in required)}

    try:
        Model = model_from_input_schema(tool_name, input_schema)
        instance = Model(**cleaned)
        # exclude_none drops any optional field still carrying None (because no
        # explicit default was specified in the schema). Keeps the args payload
        # sent to the MCP server tight.
        return instance.model_dump(exclude_none=True), None
    except ValidationError as e:
        return None, f"Pydantic validation failed for {tool_name}: {e.errors(include_url=False)}"
    except TypeError as e:
        return None, f"Schema construction failed for {tool_name}: {e}"
