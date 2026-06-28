from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from matbrain.llm import LLMRouter
from matbrain.mcp_client import MultiMCPClient, ToolResult
from matbrain.parsers import ExecutorOutput, parse_executor_output
from matbrain.prompts import render_t1_system
from matbrain.state import AgentState, HistoryEntry
from matbrain.validators import validate_arguments

logger = logging.getLogger(__name__)

_FORCE_NO_OPTIMIZE = os.environ.get("MATBRAIN_FORCE_NO_OPTIMIZE", "0") == "1"

_RETRY_TEMPLATE = (
    "Your previous response did not parse. Errors: {errors}\n\n"
    "Re-emit the response in the strict <think>/<tool_call>/<answer> format. "
    "Do not include any prose outside tags."
)


def _build_user_message(state: AgentState) -> str:
    lines: list[str] = []
    lines.append(f"# Original research query\n{state['original_query']}\n")

    if state.get("pending_instruction"):
        lines.append(f"# Current instruction (from Mat-R1)\n{state['pending_instruction']}\n")

    history = state.get("history", [])
    if len(history) > 1:
        lines.append("# Execution trail so far")
        for entry in history[1:]:
            role = entry.get("role", "?")
            content = entry.get("content", "")
            md = entry.get("metadata", {}) or {}
            if role == "tool":
                tname = md.get("tool_name", "?")
                status = "ok" if md.get("success") else "FAIL"
                lines.append(f"[tool {tname} {status}] {content}")
            else:
                lines.append(f"[{role}] {content}")
        lines.append("")

    lines.append("Emit your next response now (Pattern A or B).")
    return "\n".join(lines)


async def _dispatch_calls(
    client: MultiMCPClient,
    parsed: ExecutorOutput,
    enabled_tool_names: set[str] | None = None,
) -> list[HistoryEntry]:
    pre: list[tuple[Any, dict[str, Any] | None, str | None]] = []
    for call in parsed.tool_calls:
        spec = client.get(call.name)
        if spec is None:
            pre.append((call, None, f"unknown tool '{call.name}'"))
            continue
        if enabled_tool_names is not None and call.name not in enabled_tool_names:
            pre.append((
                call,
                None,
                f"tool '{call.name}' (server {spec.server}) is currently disabled "
                f"by user selection; available servers this turn = "
                f"{sorted({client.get(t).server for t in enabled_tool_names if client.get(t)})}",
            ))
            continue

        if _FORCE_NO_OPTIMIZE:
            props = (spec.input_schema or {}).get("properties", {}) or {}
            for flag in ("optimize_structure", "relax_structure"):
                if flag in props and call.arguments.get(flag) is not False:
                    logger.info("MATBRAIN_FORCE_NO_OPTIMIZE: forcing %s=False on %s", flag, call.name)
                    call.arguments[flag] = False
        validated, err = validate_arguments(call.name, spec.input_schema, call.arguments)
        pre.append((call, validated, err))

    async def run(call, args):
        return await client.call(call.name, args)

    coros = []
    coro_idx: list[int] = []
    for i, (call, args, err) in enumerate(pre):
        if err is None and args is not None:
            coros.append(run(call, args))
            coro_idx.append(i)

    results: list[ToolResult] = []
    if coros:
        gathered = await asyncio.gather(*coros, return_exceptions=True)
        results = [
            r
            if isinstance(r, ToolResult)
            else ToolResult(tool_name=pre[coro_idx[j]][0].name, success=False, content="", error=f"{type(r).__name__}: {r}")
            for j, r in enumerate(gathered)
        ]

    out: list[HistoryEntry] = []
    res_iter = iter(results)
    for call, args, err in pre:
        if err is not None:
            out.append(
                HistoryEntry(
                    role="tool",
                    content=err,
                    metadata={
                        "tool_name": call.name,
                        "tool_args": call.arguments,
                        "success": False,
                        "validation_error": err,
                    },
                )
            )
            continue
        r = next(res_iter)
        content = r.content
        snippet = content if len(content) < 8000 else content[:8000] + f"\n...[truncated, total {len(content)} chars]"
        if not r.success:
            snippet = (r.error or "") + ("\n" + snippet if snippet else "")
        out.append(
            HistoryEntry(
                role="tool",
                content=snippet,
                metadata={
                    "tool_name": call.name,
                    "tool_args": args,
                    "success": r.success,
                    "error": r.error,
                },
            )
        )
    return out


def make_executor_node(router: LLMRouter, client: MultiMCPClient, default_model: str | None = None):
    async def executor_node(state: AgentState, config=None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        model = configurable.get("t1_model") or default_model
        enabled_servers_raw = configurable.get("enabled_servers")
        enabled_servers: set[str] | None = None
        enabled_tool_names: set[str] | None = None
        if enabled_servers_raw:
            enabled_servers = set(enabled_servers_raw)
            enabled_tool_names = client.tool_names_for_servers(enabled_servers)

        tools_block = client.render_inventory(enabled_servers=enabled_servers)
        system = render_t1_system(tools_block)
        user_msg = _build_user_message(state)

        _t1_temp = float(os.environ.get("MATBRAIN_T1_TEMPERATURE", "0.6"))
        _t1_retry_temp = float(os.environ.get("MATBRAIN_T1_RETRY_TEMPERATURE", "0.3"))
        raw = await router.complete(model=model, system=system, user=user_msg, temperature=_t1_temp)
        parsed = parse_executor_output(raw)

        if parsed.parse_errors and not parsed.tool_calls and not parsed.answer:
            logger.warning("Mat-T1 parse errors: %s — retrying once", parsed.parse_errors)
            retry_user = user_msg + "\n\n" + _RETRY_TEMPLATE.format(errors="; ".join(parsed.parse_errors))
            raw = await router.complete(model=model, system=system, user=retry_user, temperature=_t1_retry_temp)
            parsed = parse_executor_output(raw)

        history_update: list[HistoryEntry] = [
            HistoryEntry(
                role="executor",
                content=raw,
                metadata={
                    "think": parsed.think,
                    "n_tool_calls": len(parsed.tool_calls),
                    "parse_errors": parsed.parse_errors,
                    "terminal": parsed.is_terminal,
                },
            )
        ]

        if parsed.tool_calls:
            tool_entries = await _dispatch_calls(
                client, parsed, enabled_tool_names=enabled_tool_names
            )
            history_update.extend(tool_entries)

        if parsed.is_terminal and parsed.answer:
            history_update.append(
                HistoryEntry(
                    role="executor",
                    content=parsed.answer,
                    metadata={"kind": "executor_answer"},
                )
            )

        new_history = state.get("history", []) + history_update
        update: dict[str, Any] = {
            "history": new_history,
            "pending_instruction": None,
            "iteration": state.get("iteration", 0) + 1,
        }
        return update

    return executor_node
