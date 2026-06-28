"""Conditional edge logic for the MatBrain DCG."""

from __future__ import annotations

from typing import Literal

from matbrain.state import AgentState


def reasoner_router(state: AgentState) -> Literal["executor", "finalizer"]:
    """After Mat-R1 runs, decide whether to loop back to Mat-T1 or finalize."""
    if state.get("terminated"):
        return "finalizer"
    if state.get("final_answer"):
        return "finalizer"
    return "executor"
