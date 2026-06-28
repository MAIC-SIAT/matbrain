from __future__ import annotations

from typing import Any

from matbrain.state import AgentState, HistoryEntry


async def finalizer_node(state: AgentState) -> dict[str, Any]:
    answer = state.get("final_answer")
    if not answer:
        for entry in reversed(state.get("history", [])):
            if entry.get("role") == "reasoner":
                answer = entry.get("content", "")
                break
    entry = HistoryEntry(role="reasoner", content=answer or "(no answer produced)", metadata={"kind": "finalizer"})
    return {
        "final_answer": answer or "(no answer produced)",
        "terminated": True,
        "history": state.get("history", []) + [entry],
    }
