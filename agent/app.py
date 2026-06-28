"""Gradio web UI for the MatBrain dual-model collaborative agent.

Layout:
  ┌────────────────────────────────────────────┬──────────────────────────────┐
  │ Chat (Mat-T1 + Mat-R1 collaborative loop)  │ Live agent trace             │
  │                                            │  - iteration counter          │
  │  [user] ...                                │  - executor tool calls        │
  │  [assistant] ... final answer ...          │  - reasoner decision          │
  │                                            │                              │
  │  [Textbox]  [Send] [Stop] [Reset]          │ [Tool inventory accordion]   │
  └────────────────────────────────────────────┴──────────────────────────────┘

The UI streams LangGraph node-level events via `graph.astream(..., stream_mode="updates")`.
Multi-turn history is supported by prepending prior Q→A pairs to each new
agent invocation's `original_query` (the agent loop itself is stateless per query).
"""

from __future__ import annotations

import asyncio
import html
import json
import logging
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from matbrain.config import get_settings
from matbrain.graph import build_graph
from matbrain.state import initial_state

logger = logging.getLogger("matbrain.ui")

# Globals set at startup.
_GRAPH = None
_CLIENT = None
_SETTINGS = None


async def _startup() -> None:
    global _GRAPH, _CLIENT, _SETTINGS
    _SETTINGS = get_settings()
    logging.basicConfig(
        level=getattr(logging, _SETTINGS.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )
    _GRAPH, _CLIENT = await build_graph(_SETTINGS)
    logger.info("matbrain UI ready (%d MCP tools)", len(_CLIENT.registry))


def _render_tool_inventory() -> str:
    """Markdown table of all discovered MCP tools, grouped by server."""
    if _CLIENT is None or not _CLIENT.registry:
        return "_no tools loaded_"
    by_server: dict[str, list] = {}
    for spec in _CLIENT.registry.values():
        by_server.setdefault(spec.server, []).append(spec)
    lines: list[str] = []
    for server in sorted(by_server):
        tools = sorted(by_server[server], key=lambda s: s.name)
        lines.append(f"### `{server}` ({len(tools)} tools)")
        for spec in tools:
            desc = (spec.description or "").strip().splitlines()[0][:120]
            lines.append(f"- **{spec.name}** — {desc}")
        lines.append("")
    return "\n".join(lines)


def _build_query_with_context(new_question: str, chat_history: list[dict]) -> str:
    """Inline prior turns so the next agent run has memory of earlier answers."""
    prior_pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in chat_history:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            prior_pairs.append((pending_user, content))
            pending_user = None
    if not prior_pairs:
        return new_question
    ctx_lines = ["# Prior conversation in this session"]
    for q, a in prior_pairs:
        ctx_lines.append(f"User asked: {q}")
        ctx_lines.append(f"You answered: {a}")
        ctx_lines.append("")
    ctx_lines.append("# Current new question (answer this, leveraging prior context where useful)")
    ctx_lines.append(new_question)
    return "\n".join(ctx_lines)


def _expandable(text: str, preview_chars: int = 240) -> str:
    """Render text inline up to preview_chars; wrap the full content in a
    <details> block if it would otherwise be truncated. Safe inside markdown
    (gradio's markdown renderer passes <details>/<summary>/<pre> through)."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= preview_chars:
        return f"_{text}_"
    preview = text[:preview_chars].rstrip()
    full = html.escape(text)
    return (
        f"_{preview}…_\n\n"
        f'<details><summary>📖 展开完整 ({len(text)} chars)</summary>\n\n'
        f'<pre style="white-space: pre-wrap; font-size: 0.85em; '
        f'line-height: 1.4; padding: 8px; border-radius: 4px;">{full}</pre>\n'
        f"</details>"
    )


def _expandable_plain(text: str, preview_chars: int = 240, label: str = "完整内容") -> str:
    """Same as _expandable but without italic styling — for tool args / instructions."""
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= preview_chars:
        return text
    preview = text[:preview_chars].rstrip()
    full = html.escape(text)
    return (
        f"{preview}…\n\n"
        f'<details><summary>📖 展开{label} ({len(text)} chars)</summary>\n\n'
        f'<pre style="white-space: pre-wrap; font-size: 0.85em; '
        f'line-height: 1.4; padding: 8px; border-radius: 4px;">{full}</pre>\n'
        f"</details>"
    )


def _format_node_event(node_name: str, update: dict, last_history_len: int) -> tuple[str, int]:
    """Map an `astream` update chunk to a human-readable trace block.

    Returns (markdown_block, new_history_length).
    """
    history = update.get("history") or []
    new_entries = history[last_history_len:] if isinstance(history, list) else []
    new_len = len(history) if isinstance(history, list) else last_history_len

    if node_name == "executor":
        iteration = update.get("iteration", "?")
        tool_lines: list[str] = []
        executor_think: str | None = None
        executor_answer: str | None = None
        for entry in new_entries:
            role = entry.get("role")
            md = entry.get("metadata") or {}
            if role == "executor" and md.get("kind") != "executor_answer":
                executor_think = (md.get("think") or "").strip()
                if md.get("parse_errors"):
                    tool_lines.append(f"  ⚠️ parse errors: {md['parse_errors']}")
            elif role == "executor" and md.get("kind") == "executor_answer":
                executor_answer = entry.get("content", "")
            elif role == "tool":
                tname = md.get("tool_name", "?")
                success = "✓" if md.get("success") else "✗"
                args = md.get("tool_args")
                args_str = json.dumps(args, ensure_ascii=False) if args else ""
                result_text = (entry.get("content") or "").strip()
                tool_lines.append(
                    f"  {success} `{tname}`  args= {_expandable_plain(args_str, 160, label='参数')}"
                )
                if result_text:
                    label = "错误信息" if not md.get("success") else "返回结果"
                    tool_lines.append(
                        f"    └─ {label}: {_expandable_plain(result_text, 200, label=label)}"
                    )

        block = [f"### 🔧 Mat-T1 executor — iter {iteration}"]
        if executor_think:
            block.append(f"  > {_expandable(executor_think, 240)}")
        if tool_lines:
            block.append("  **Tools dispatched:**")
            block.extend(tool_lines)
        if executor_answer:
            block.append(f"  **Executor handoff:** {_expandable_plain(executor_answer, 200, label='handoff')}")
        if not tool_lines and not executor_answer:
            block.append("  _(no tool calls this turn)_")
        return "\n".join(block), new_len

    if node_name == "reasoner":
        terminated = update.get("terminated", False)
        verdict = "✅ **terminate**" if terminated else "🔁 **continue**"
        # Reasoner block is the last new entry; its metadata holds parse info.
        last_reasoner = next((e for e in reversed(new_entries) if e.get("role") == "reasoner"), None)
        think = ""
        if last_reasoner:
            md = last_reasoner.get("metadata") or {}
            think = (md.get("think") or "").strip()
            if md.get("forced_terminate"):
                verdict = "🛑 **forced terminate (max iter)**"
        block = [f"### 🧠 Mat-R1 reasoner — {verdict}"]
        if think:
            block.append(f"  > {_expandable(think, 240)}")
        if not terminated:
            instr = update.get("pending_instruction") or ""
            if instr:
                block.append(
                    f"  **Next instruction:** {_expandable_plain(instr, 240, label='指令')}"
                )
        return "\n".join(block), new_len

    if node_name == "finalizer":
        return "### 🏁 finalizer\n  answer ready", new_len

    return f"### {node_name}\n  {json.dumps(update, default=str)[:200]}", new_len


_SPINNER = "🤔 _agent is thinking..._"

def _all_available_models() -> list[str]:
    """Every model name the LLMRouter knows about — populated at agent startup
    from .env (each role's served-model-name). Both dropdowns show this same
    list so the user can pick either role's model freely."""
    if _CLIENT is None or not hasattr(_CLIENT, "llm_router"):
        return []
    return list(_CLIENT.llm_router.known_models())


def _server_checkbox_choices() -> list[tuple[str, str]]:
    """[(label_with_count, server_name), ...] for the per-server CheckboxGroup."""
    if _CLIENT is None or not _CLIENT.registry:
        return []
    counts: dict[str, int] = {}
    for spec in _CLIENT.registry.values():
        counts[spec.server] = counts.get(spec.server, 0) + 1
    return [(f"{srv} ({counts[srv]} tools)", srv) for srv in sorted(counts)]


async def respond(
    user_msg: str,
    chat_history: list[dict],
    t1_model: str,
    r1_model: str,
    enabled_servers: list[str],
    turn_records: list[dict],
):
    """Drive the agent; stream chat + trace; on completion append a structured
    TurnRecord (used by the export buttons)."""
    if not user_msg.strip():
        yield chat_history, "_(empty message)_", turn_records
        return

    if _GRAPH is None:
        yield chat_history, "**error**: agent not initialized", turn_records
        return

    started_at = datetime.now(timezone.utc).isoformat()
    turn_records = list(turn_records or [])

    chat_history = list(chat_history) + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": _SPINNER},
    ]
    yield chat_history, f"_starting agent (T1={t1_model}, R1={r1_model})..._", turn_records

    augmented = _build_query_with_context(user_msg, chat_history[:-2])
    state = initial_state(augmented)
    trace_blocks: list[str] = []
    last_history_len = len(state.get("history", []))

    # Accumulate the final agent-side state for the export record.
    final_history: list[dict] = list(state.get("history", []))
    final_iteration = 0
    final_answer: str | None = None
    error: str | None = None

    cancelled = False
    try:
        async for chunk in _GRAPH.astream(
            state,
            config={
                "recursion_limit": 30,
                "configurable": {
                    "t1_model": t1_model,
                    "r1_model": r1_model,
                    "enabled_servers": enabled_servers or None,
                },
            },
            stream_mode="updates",
        ):
            for node_name, update in chunk.items():
                block, last_history_len = _format_node_event(node_name, update, last_history_len)
                trace_blocks.append(block)
                # Each node returns the full new history; keep the latest seen.
                if isinstance(update.get("history"), list):
                    final_history = update["history"]
                if isinstance(update.get("iteration"), int):
                    final_iteration = update["iteration"]
                if update.get("final_answer"):
                    final_answer = update["final_answer"]
                # Keep spinner visible until finalizer produces the final answer.
                if node_name == "finalizer":
                    chat_history[-1]["content"] = update.get("final_answer") or "(no answer)"
                elif node_name == "reasoner" and update.get("terminated"):
                    chat_history[-1]["content"] = update.get("final_answer") or _SPINNER
                yield chat_history, "\n\n".join(trace_blocks), turn_records
    except asyncio.CancelledError:
        chat_history[-1]["content"] = "_(stopped by user)_"
        trace_blocks.append("### ⛔ cancelled by user")
        error = "cancelled"
        cancelled = True
    except Exception as e:
        logger.exception("agent run failed")
        chat_history[-1]["content"] = f"❌ **error**: {type(e).__name__}: {e}"
        trace_blocks.append(f"### ❌ exception\n  `{type(e).__name__}: {e}`")
        error = f"{type(e).__name__}: {e}"

    # Append the turn record OUTSIDE any try/finally — yielding from `finally`
    # collides with `GeneratorExit` when Gradio cancels the generator (e.g. user
    # hits stop or navigates away), producing "async generator ignored
    # GeneratorExit" RuntimeError AND swallowing the State update so the export
    # buttons see an empty list immediately after the run finishes.
    completed_at = datetime.now(timezone.utc).isoformat()
    turn_records.append(
        {
            "turn_index": len(turn_records),
            "started_at": started_at,
            "completed_at": completed_at,
            "t1_model": t1_model,
            "r1_model": r1_model,
            "user_query": user_msg,
            "iteration_count": final_iteration,
            "final_answer": final_answer or chat_history[-1].get("content"),
            "error": error,
            "history": final_history,
        }
    )
    yield chat_history, "\n\n".join(trace_blocks), turn_records
    if cancelled:
        raise asyncio.CancelledError()


_JSON_BTN_DEFAULT_LABEL = "📦 导出 JSON"
_MD_BTN_DEFAULT_LABEL = "📝 导出 Markdown"


def reset_conversation():
    """Reset chatbot, trace, the export records, AND the export buttons.

    Wiping the buttons is critical: gr.DownloadButton fires a browser download
    of whatever `value` is set when clicked. If we only cleared the records,
    the next click would re-download the previous session's file before the
    handler had a chance to regenerate from the empty (or new) state.
    """
    return (
        [],
        "",
        [],
        gr.DownloadButton(value=None, label=_JSON_BTN_DEFAULT_LABEL),
        gr.DownloadButton(value=None, label=_MD_BTN_DEFAULT_LABEL),
    )


def _export_json(turn_records: list[dict]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.gettempdir()) / f"matbrain_session_{ts}.json"
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "n_turns": len(turn_records),
        "turns": turn_records,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("exported JSON to %s (%d turns)", out, len(turn_records))
    return str(out)


def _export_markdown(turn_records: list[dict]) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(tempfile.gettempdir()) / f"matbrain_session_{ts}.md"
    lines: list[str] = [
        "# MatBrain conversation transcript",
        f"_exported at {datetime.now(timezone.utc).isoformat()}_",
        f"_turns: {len(turn_records)}_",
        "",
    ]
    for turn in turn_records:
        idx = turn.get("turn_index", "?")
        lines.append(f"## Turn {idx + 1 if isinstance(idx, int) else idx}")
        lines.append(f"- **started**: {turn.get('started_at')}")
        lines.append(f"- **completed**: {turn.get('completed_at')}")
        lines.append(f"- **models**: T1=`{turn.get('t1_model')}` · R1=`{turn.get('r1_model')}`")
        lines.append(f"- **iterations**: {turn.get('iteration_count')}")
        if turn.get("error"):
            lines.append(f"- **error**: `{turn['error']}`")
        lines.append("")
        lines.append("### User query")
        lines.append(f"> {turn.get('user_query', '')}")
        lines.append("")
        lines.append("### Agent execution trail")
        for entry in turn.get("history") or []:
            role = entry.get("role")
            content = entry.get("content", "") or ""
            md = entry.get("metadata") or {}
            if role == "user":
                continue  # already shown above
            if role == "executor" and md.get("kind") != "executor_answer":
                think = md.get("think") or ""
                n_calls = md.get("n_tool_calls", 0)
                lines.append(f"#### 🔧 Mat-T1 executor ({n_calls} tool call{'s' if n_calls != 1 else ''})")
                if think:
                    lines.append("**think:**")
                    lines.append("```")
                    lines.append(think)
                    lines.append("```")
                if md.get("parse_errors"):
                    lines.append(f"_parse errors: {md['parse_errors']}_")
            elif role == "executor" and md.get("kind") == "executor_answer":
                lines.append("**Executor handoff to Mat-R1:**")
                lines.append("```")
                lines.append(content)
                lines.append("```")
            elif role == "tool":
                tname = md.get("tool_name", "?")
                success = "✓" if md.get("success") else "✗"
                args = md.get("tool_args")
                lines.append(f"#### {success} tool call · `{tname}`")
                if args:
                    lines.append("**args:**")
                    lines.append("```json")
                    lines.append(json.dumps(args, ensure_ascii=False, indent=2))
                    lines.append("```")
                label = "error" if not md.get("success") else "result"
                lines.append(f"**{label}:**")
                lines.append("```")
                lines.append(content)
                lines.append("```")
            elif role == "reasoner":
                think = md.get("think") or ""
                terminal = md.get("terminal")
                forced = md.get("forced_terminate")
                verdict = "🛑 forced-terminate" if forced else ("✅ terminate" if terminal else "🔁 continue")
                lines.append(f"#### 🧠 Mat-R1 reasoner — {verdict}")
                if think:
                    lines.append("**think:**")
                    lines.append("```")
                    lines.append(think)
                    lines.append("```")
            lines.append("")
        lines.append("### Final answer")
        lines.append(turn.get("final_answer") or "_(no answer)_")
        lines.append("\n---\n")
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("exported Markdown to %s (%d turns)", out, len(turn_records))
    return str(out)


def _export_json_handler(turn_records):
    if not turn_records:
        gr.Warning("还没有对话记录可导出。请先发送一个问题让 agent 跑一轮。")
        # Clear value too — otherwise a stale path from a previous session
        # would be re-downloaded on the next click.
        return gr.DownloadButton(value=None, label=_JSON_BTN_DEFAULT_LABEL)
    path = _export_json(turn_records)
    return gr.DownloadButton(value=path, label=f"📦 JSON ({len(turn_records)} turns)")


def _export_md_handler(turn_records):
    if not turn_records:
        gr.Warning("还没有对话记录可导出。请先发送一个问题让 agent 跑一轮。")
        return gr.DownloadButton(value=None, label=_MD_BTN_DEFAULT_LABEL)
    path = _export_markdown(turn_records)
    return gr.DownloadButton(value=path, label=f"📝 Markdown ({len(turn_records)} turns)")


_TRACE_CSS = """
.matbrain-trace, .matbrain-trace * {
    overflow-wrap: anywhere !important;
    word-break: break-word !important;
}
.matbrain-trace pre {
    white-space: pre-wrap !important;
    max-width: 100% !important;
    overflow-x: hidden !important;
}
.matbrain-trace details {
    max-width: 100% !important;
    margin-bottom: 4px;
}
.matbrain-trace summary {
    cursor: pointer;
}
"""


def build_ui() -> gr.Blocks:
    settings = _SETTINGS
    all_choices = _all_available_models()
    default_t1 = settings.mat_t1_model if settings else (all_choices[0] if all_choices else "")
    default_r1 = settings.mat_r1_model if settings else (all_choices[0] if all_choices else "")
    if default_t1 not in all_choices:
        default_t1 = all_choices[0] if all_choices else default_t1
    if default_r1 not in all_choices:
        default_r1 = all_choices[0] if all_choices else default_r1
    t1_label = "Mat-T1 model (executor)"
    r1_label = "Mat-R1 model (reasoner)"

    with gr.Blocks(title="MatBrain — collaborative crystal materials agent") as demo:
        gr.Markdown(
            "# MatBrain\n"
            "Dual-model collaborative agent over the Mat-MCP toolset "
            "(Mat-T1 executor + Mat-R1 reasoner, LangGraph state machine).\n\n"
            "_Mat-T1 (executor) and Mat-R1 (reasoner) each use the provider "
            "configured in `.env` via `MAT_T*_PROVIDER`. The dropdowns below "
            "show the models served by each role's provider._"
        )
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="Conversation", height=600)
                msg = gr.Textbox(
                    label="Question",
                    placeholder="例：查询 CsPbBr3 的晶体结构并预测其形成能",
                    lines=2,
                )
                with gr.Row():
                    send_btn = gr.Button("发送", variant="primary")
                    stop_btn = gr.Button("停止", variant="stop")
                    reset_btn = gr.Button("重置对话")
                with gr.Row():
                    export_json_btn = gr.DownloadButton(
                        label="📦 导出 JSON",
                        size="md",
                    )
                    export_md_btn = gr.DownloadButton(
                        label="📝 导出 Markdown",
                        size="md",
                    )
                turn_records_state = gr.State(value=[])
            with gr.Column(scale=2):
                with gr.Row():
                    t1_model_dd = gr.Dropdown(
                        label=t1_label,
                        choices=all_choices,
                        value=default_t1,
                        interactive=True,
                    )
                    r1_model_dd = gr.Dropdown(
                        label=r1_label,
                        choices=all_choices,
                        value=default_r1,
                        interactive=True,
                    )
                _server_choices = _server_checkbox_choices()
                _server_values = [v for _, v in _server_choices]
                with gr.Accordion(
                    "🧰 启用的 MCP server (per-request, 默认全开)",
                    open=False,
                ):
                    servers_cb = gr.CheckboxGroup(
                        choices=_server_choices,
                        value=_server_values,
                        label=None,
                        show_label=False,
                        interactive=True,
                    )
                gr.Markdown("### 🪞 Agent trace (live)")
                trace = gr.Markdown(
                    value="_send a question to start_",
                    height=520,
                    sanitize_html=False,
                    elem_classes=["matbrain-trace"],
                )
                with gr.Accordion(
                    f"🧰 Tool inventory ({len(_CLIENT.registry) if _CLIENT else 0} tools)",
                    open=False,
                ):
                    gr.Markdown(_render_tool_inventory())

        send_event = send_btn.click(
            fn=respond,
            inputs=[msg, chatbot, t1_model_dd, r1_model_dd, servers_cb, turn_records_state],
            outputs=[chatbot, trace, turn_records_state],
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        submit_event = msg.submit(
            fn=respond,
            inputs=[msg, chatbot, t1_model_dd, r1_model_dd, servers_cb, turn_records_state],
            outputs=[chatbot, trace, turn_records_state],
        ).then(
            fn=lambda: "",
            outputs=[msg],
        )

        stop_btn.click(
            fn=None,
            inputs=None,
            outputs=None,
            cancels=[send_event, submit_event],
        )

        reset_btn.click(
            fn=reset_conversation,
            inputs=None,
            outputs=[chatbot, trace, turn_records_state, export_json_btn, export_md_btn],
        )

        export_json_btn.click(
            fn=_export_json_handler,
            inputs=[turn_records_state],
            outputs=[export_json_btn],
        )
        export_md_btn.click(
            fn=_export_md_handler,
            inputs=[turn_records_state],
            outputs=[export_md_btn],
        )

    return demo


def main() -> None:
    asyncio.run(_startup())
    ui = build_ui()
    ui.queue(default_concurrency_limit=4)
    ui.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=_TRACE_CSS,
    )


if __name__ == "__main__":
    main()
