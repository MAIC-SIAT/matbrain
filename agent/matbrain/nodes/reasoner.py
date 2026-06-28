from __future__ import annotations

import json
import logging
import os
from typing import Any

from matbrain.llm import LLMRouter
from matbrain.parsers import parse_reasoner_output
from matbrain.prompts import MAT_R1_SYSTEM, render_finalize_contract
from matbrain.state import AgentState, HistoryEntry

logger = logging.getLogger(__name__)

def _build_user_message(state: AgentState, max_iter: int) -> str:
    lines: list[str] = []
    lines.append(f"# Original research query\n{state['original_query']}\n")
    lines.append(f"# Iteration {state.get('iteration', 0)} of {max_iter}")
    if state.get("iteration", 0) >= max_iter:
        lines.append(
            "NOTE: max iterations reached. You MUST produce Pattern B (<answer>) "
            "based on whatever evidence is available."
        )
    lines.append("")
    lines.append("# Full execution history")
    for entry in state.get("history", []):
        role = entry.get("role", "?")
        content = entry.get("content", "")
        md = entry.get("metadata", {}) or {}
        if role == "tool":
            tname = md.get("tool_name", "?")
            status = "ok" if md.get("success") else "FAIL"
            args_str = json.dumps(md.get("tool_args"), ensure_ascii=False)[:200]
            lines.append(f"[tool {tname} {status} args={args_str}]")
            lines.append(content)
        else:
            lines.append(f"[{role}] {content}")
    lines.append("")
    contract = render_finalize_contract(state.get("question_type"))
    if contract:
        lines.append("# 最终答案格式契约 (Pattern B 收尾时必须遵守)")
        lines.append(contract)
        lines.append("")
    lines.append("Decide: continue (Pattern A) or finalize (Pattern B). Emit your response now.")
    return "\n".join(lines)


def make_reasoner_node(router: LLMRouter, max_iterations: int, default_model: str | None = None):
    async def reasoner_node(state: AgentState, config=None) -> dict[str, Any]:
        configurable = (config or {}).get("configurable") or {}
        model = configurable.get("r1_model") or default_model
        user_msg = _build_user_message(state, max_iterations)
        _r1_temp = float(os.environ.get("MATBRAIN_R1_TEMPERATURE", "0.3"))
        raw = await router.complete(model=model, system=MAT_R1_SYSTEM, user=user_msg, temperature=_r1_temp)
        parsed = parse_reasoner_output(raw)

        force_terminate = state.get("iteration", 0) >= max_iterations
        terminated = parsed.is_terminal or force_terminate

        history_update = [
            HistoryEntry(
                role="reasoner",
                content=raw,
                metadata={
                    "think": parsed.think,
                    "parse_errors": parsed.parse_errors,
                    "terminal": terminated,
                    "forced_terminate": force_terminate and not parsed.is_terminal,
                },
            )
        ]

        update: dict[str, Any] = {
            "history": state.get("history", []) + history_update,
            "terminated": terminated,
        }
        if terminated:
            update["final_answer"] = parsed.answer or (raw if force_terminate else None)
            update["pending_instruction"] = None
        else:
            update["pending_instruction"] = parsed.next_instruction or ""

        return update

    return reasoner_node
