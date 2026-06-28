"""Regex-based parsers for the tag-structured outputs of Mat-T1 and Mat-R1."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NEXT_INSTR_RE = re.compile(r"<next_instruction>(.*?)</next_instruction>", re.DOTALL)

# Fallbacks: thinking-mode models occasionally emit opening tag without
# closing one (truncation, formatting drift). Treat everything from the
# opening tag to end-of-string as the body in that case.
_ANSWER_OPEN_RE = re.compile(r"<answer>\s*(.+)\Z", re.DOTALL)
_NEXT_INSTR_OPEN_RE = re.compile(r"<next_instruction>\s*(.+)\Z", re.DOTALL)


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str


@dataclass
class ExecutorOutput:
    think: str
    tool_calls: list[ToolCall]
    answer: str | None
    parse_errors: list[str]

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def is_terminal(self) -> bool:
        return self.answer is not None and not self.tool_calls


@dataclass
class ReasonerOutput:
    think: str
    next_instruction: str | None
    answer: str | None
    parse_errors: list[str]

    @property
    def is_terminal(self) -> bool:
        return self.answer is not None


def _first(pat: re.Pattern[str], text: str) -> str | None:
    m = pat.search(text)
    return m.group(1).strip() if m else None


_NESTED_ANSWER_RE = re.compile(r"</?answer>", re.IGNORECASE)
_NESTED_NEXT_RE = re.compile(r"</?next_instruction>", re.IGNORECASE)


def _clean_nested(text: str | None, kind: str) -> str | None:
    """Mat-R1 fine-tunes occasionally double-wrap their output, e.g.
    `<answer><answer>X</answer></answer>`. After the outer regex extracts the
    inner span, residual nested tags may remain — strip them all."""
    if text is None:
        return None
    if kind == "answer":
        return _NESTED_ANSWER_RE.sub("", text).strip()
    if kind == "next_instruction":
        return _NESTED_NEXT_RE.sub("", text).strip()
    return text


def parse_executor_output(text: str) -> ExecutorOutput:
    errors: list[str] = []
    think = _first(_THINK_RE, text) or ""
    if not think:
        errors.append("missing <think> block")

    tool_calls: list[ToolCall] = []
    for m in _TOOL_CALL_RE.finditer(text):
        raw = m.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as e:
            errors.append(f"<tool_call> JSON parse failed: {e}")
            continue
        if not isinstance(payload, dict):
            errors.append("<tool_call> must be a JSON object")
            continue
        name = payload.get("name")
        args = payload.get("arguments", {})
        if not isinstance(name, str) or not name:
            errors.append("<tool_call> missing 'name'")
            continue
        if not isinstance(args, dict):
            errors.append(f"<tool_call> 'arguments' must be an object, got {type(args).__name__}")
            continue
        tool_calls.append(ToolCall(name=name, arguments=args, raw=raw))

    answer = _clean_nested(_first(_ANSWER_RE, text) or _first(_ANSWER_OPEN_RE, text), "answer")

    if not tool_calls and not answer:
        errors.append("response contains neither <tool_call> nor <answer>")

    return ExecutorOutput(
        think=think,
        tool_calls=tool_calls,
        answer=answer,
        parse_errors=errors,
    )


def parse_reasoner_output(text: str) -> ReasonerOutput:
    errors: list[str] = []
    think = _first(_THINK_RE, text) or ""
    if not think:
        errors.append("missing <think> block")

    answer = _clean_nested(_first(_ANSWER_RE, text) or _first(_ANSWER_OPEN_RE, text), "answer")
    next_instr = _clean_nested(
        _first(_NEXT_INSTR_RE, text) or _first(_NEXT_INSTR_OPEN_RE, text),
        "next_instruction",
    )

    if answer and next_instr:
        errors.append("response contains BOTH <answer> and <next_instruction>; treating as terminal")
        next_instr = None
    if not answer and not next_instr:
        errors.append("response contains neither <answer> nor <next_instruction>")

    return ReasonerOutput(
        think=think,
        next_instruction=next_instr,
        answer=answer,
        parse_errors=errors,
    )
