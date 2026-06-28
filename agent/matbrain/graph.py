"""Assemble the MatBrain LangGraph DCG: executor (Mat-T1) <-> reasoner (Mat-R1).

Each node receives an `LLMRouter` built from environment-configured model
endpoints. At request time the picked model identifier selects which
underlying client is used.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from matbrain.config import Settings, get_settings
from matbrain.llm import LLMRouter, ModelSpec
from matbrain.mcp_client import MultiMCPClient
from matbrain.nodes.executor import make_executor_node
from matbrain.nodes.finalizer import finalizer_node
from matbrain.nodes.reasoner import make_reasoner_node
from matbrain.routing import reasoner_router
from matbrain.state import AgentState

def build_model_registry(s: Settings) -> dict[str, ModelSpec]:
    """Build the (model name -> spec) registry from .env settings.

    The served model names of the configured T1/R1 endpoints are registered
    when a provider and model identifier are supplied.
    """
    reg: dict[str, ModelSpec] = {}

    # Register each role's configured endpoint by its served-model-name.
    # Same model name from both roles (rare) won't double-register.
    for role in ("t1", "r1"):
        provider = getattr(s, f"mat_{role}_provider")
        model = getattr(s, f"mat_{role}_model")
        base_url = getattr(s, f"mat_{role}_base_url")
        api_key = getattr(s, f"mat_{role}_api_key")
        if model and provider.lower() in ("openai", "vllm", "openai-compatible"):
            reg[model] = ModelSpec(
                name=model,
                provider=provider,
                base_url=base_url,
                api_key=api_key,
            )

    return reg


async def build_graph(settings: Settings | None = None):
    s = settings or get_settings()

    client = MultiMCPClient(endpoints=s.mcp_endpoints(), call_timeout=s.tool_call_timeout)
    await client.discover()

    # Optional safety control: drop tools whose names contain any substring in
    # MATBRAIN_DISABLE_TOOLS (comma-separated).
    import os as _os

    blocklist = [t.strip() for t in _os.environ.get("MATBRAIN_DISABLE_TOOLS", "").split(",") if t.strip()]
    if blocklist:
        dropped = [n for n in list(client.registry) if any(b in n for b in blocklist)]
        for n in dropped:
            client.registry.pop(n, None)
        import logging as _logging

        _logging.getLogger("matbrain.graph").warning(
            "MATBRAIN_DISABLE_TOOLS pruned %d tools: %s", len(dropped), dropped
        )

    # Optional allowlist control: if set, keep only the exact tool names listed
    # in MATBRAIN_ALLOW_TOOLS (comma-separated).
    allowlist_raw = _os.environ.get("MATBRAIN_ALLOW_TOOLS", "").strip()
    if allowlist_raw:
        allowlist = {t.strip() for t in allowlist_raw.split(",") if t.strip()}
        import logging as _logging
        dropped_allow = [n for n in list(client.registry) if n not in allowlist]
        for n in dropped_allow:
            client.registry.pop(n, None)
        _logging.getLogger("matbrain.graph").warning(
            "MATBRAIN_ALLOW_TOOLS kept %d tools, pruned %d: %s",
            len(allowlist), len(dropped_allow), dropped_allow,
        )

    registry = build_model_registry(s)
    router = LLMRouter(registry=registry)

    executor = make_executor_node(router, client, default_model=s.mat_t1_model)
    reasoner = make_reasoner_node(router, max_iterations=s.max_iterations, default_model=s.mat_r1_model)

    g = StateGraph(AgentState)
    g.add_node("executor", executor)
    g.add_node("reasoner", reasoner)
    g.add_node("finalizer", finalizer_node)

    g.add_edge(START, "executor")
    g.add_edge("executor", "reasoner")
    g.add_conditional_edges("reasoner", reasoner_router, ["executor", "finalizer"])
    g.add_edge("finalizer", END)

    compiled = g.compile()
    # Expose the router on the client so the UI can list available models.
    client.llm_router = router  # type: ignore[attr-defined]
    return compiled, client
