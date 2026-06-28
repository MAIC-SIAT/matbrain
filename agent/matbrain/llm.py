"""LLM client for Mat-T1 (executor) and Mat-R1 (reasoner).

The agent always invokes `client.complete(system, user, temperature, model)`
and gets back a plain string in the strict tag format
`<think>...</think><tool_call>...</tool_call>` or `<think>...</think><answer>...</answer>`.

The backend is any OpenAI-compatible chat-completions endpoint (e.g. vLLM).
"""

from __future__ import annotations

import logging
from typing import Protocol

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    default_model: str

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.6,
        model: str | None = None,
    ) -> str: ...


class OpenAILLM:
    def __init__(self, api_key: str, base_url: str, default_model: str, max_tokens: int = 8192):
        # Reasoning-tuned models (Qwen3-Thinking) routinely produce 2-4K tokens of
        # CoT before the answer; cap at 8K by default.
        self.client = AsyncOpenAI(api_key=api_key or "dummy", base_url=base_url)
        self.default_model = default_model
        self.max_tokens = max_tokens

    async def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.6,
        model: str | None = None,
    ) -> str:
        resp = await self.client.chat.completions.create(
            model=model or self.default_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=self.max_tokens,
        )
        msg = resp.choices[0].message
        content = msg.content or ""
        reasoning = (
            getattr(msg, "reasoning", None)
            or getattr(msg, "reasoning_content", None)
            or ""
        )
        # We deliberately do NOT splice `reasoning` into the response stream.
        # vLLM's reasoning_parser already separates the model's CoT from the
        # final assistant content; splicing it back as <think>...</think>
        # confuses our agent-side parser because reasoning text often contains
        # literal "<answer>" examples (the model thinking about its own format)
        # which then get matched ahead of the real <answer> in the content.
        # The CoT is preserved in logs for debugging.
        if reasoning:
            logger.debug("OpenAILLM CoT (reasoning_len=%d): %s", len(reasoning), reasoning[:200])
        return content


def make_llm(
    provider: str,
    api_key: str,
    base_url: str,
    default_model: str,
    max_tokens: int = 4096,
) -> LLMClient:
    provider = (provider or "").lower().strip()
    if provider in ("openai", "vllm", "openai-compatible"):
        return OpenAILLM(api_key=api_key, base_url=base_url, default_model=default_model, max_tokens=max_tokens)
    raise ValueError(f"unknown LLM provider: {provider!r}; expected 'openai-compatible'")


class ModelSpec:
    """Static info about a model the agent can target."""

    def __init__(self, name: str, provider: str, base_url: str, api_key: str):
        self.name = name
        self.provider = provider
        self.base_url = base_url
        self.api_key = api_key


class LLMRouter:
    """Maps any model name to a ready LLM client. Each unique
    (provider, base_url) tuple shares one client; the per-call `model`
    argument selects the model within that endpoint.

    At request time, the picked model identifier is looked up here.
    """

    def __init__(self, registry: dict[str, ModelSpec], max_tokens: int = 8192):
        self.registry = registry
        self.max_tokens = max_tokens
        self._clients: dict[tuple[str, str], LLMClient] = {}

    def known_models(self) -> list[str]:
        return list(self.registry.keys())

    def get_client(self, model: str) -> LLMClient:
        spec = self.registry.get(model)
        if spec is None:
            raise KeyError(f"model {model!r} not in registry; known={list(self.registry)}")
        key = (spec.provider, spec.base_url)
        client = self._clients.get(key)
        if client is None:
            client = make_llm(
                provider=spec.provider,
                api_key=spec.api_key,
                base_url=spec.base_url,
                default_model=model,
                max_tokens=self.max_tokens,
            )
            self._clients[key] = client
        return client

    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.6,
    ) -> str:
        client = self.get_client(model)
        return await client.complete(system=system, user=user, temperature=temperature, model=model)


# Alias for OpenAILLM.
LLM = OpenAILLM
