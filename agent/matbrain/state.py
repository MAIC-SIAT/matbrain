"""Shared global state passed through the LangGraph DCG."""

from __future__ import annotations

from typing import Any, Literal, TypedDict


class HistoryEntry(TypedDict, total=False):
    role: Literal["user", "executor", "tool", "reasoner"]
    content: str
    metadata: dict[str, Any]


class AgentState(TypedDict, total=False):
    original_query: str
    question_type: str | None
    history: list[HistoryEntry]
    pending_instruction: str | None
    iteration: int
    final_answer: str | None
    terminated: bool


def initial_state(query: str, question_type: str | None = None) -> AgentState:
    return AgentState(
        original_query=query,
        question_type=question_type,
        history=[HistoryEntry(role="user", content=query, metadata={})],
        pending_instruction=None,
        iteration=0,
        final_answer=None,
        terminated=False,
    )
