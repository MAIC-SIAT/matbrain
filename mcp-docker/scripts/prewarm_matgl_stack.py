#!/usr/bin/env python3
"""Prewarm MatGL MCP model sessions to avoid first-call latency during rollout."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import httpx
from mcp import ClientSession
from mcp.client.sse import sse_client


LOCAL_NO_PROXY = "localhost,127.0.0.1,::1,your-mcp-host"


def force_local_no_proxy() -> None:
    existing = os.environ.get("NO_PROXY") or os.environ.get("no_proxy") or ""
    merged = ",".join(part for part in [existing, LOCAL_NO_PROXY] if part).strip(",")
    os.environ["NO_PROXY"] = merged
    os.environ["no_proxy"] = merged
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        os.environ.pop(key, None)


def local_mcp_httpx_client_factory(
    headers: dict[str, object] | None = None,
    timeout: httpx.Timeout | None = None,
    auth: httpx.Auth | None = None,
) -> httpx.AsyncClient:
    kwargs: dict[str, object] = {
        "follow_redirects": True,
        "trust_env": False,
    }
    if timeout is not None:
        kwargs["timeout"] = timeout
    if headers is not None:
        kwargs["headers"] = headers
    if auth is not None:
        kwargs["auth"] = auth
    return httpx.AsyncClient(**kwargs)


async def _prewarm_once(server_url: str, model_refs: str, timeout: float) -> dict[str, object]:
    async with sse_client(
        server_url,
        timeout=timeout,
        sse_read_timeout=timeout,
        httpx_client_factory=local_mcp_httpx_client_factory,
    ) as streams:
        async with ClientSession(*streams) as session:
            await asyncio.wait_for(session.initialize(), timeout=timeout)
            result = await asyncio.wait_for(
                session.call_tool("prewarm_matgl_models_MatGL", {"model_refs": model_refs}),
                timeout=timeout,
            )
    content = "\n".join(getattr(part, "text", str(part)) for part in (result.content or []))
    return {
        "ok": True,
        "error": None,
        "content": content,
    }


async def main_async(args: argparse.Namespace) -> int:
    force_local_no_proxy()
    server_url = args.server_url
    model_refs = args.model_refs
    timeout = float(args.timeout)
    concurrency = max(1, int(args.concurrency))

    tasks = [
        asyncio.create_task(_prewarm_once(server_url, model_refs, timeout))
        for _ in range(concurrency)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    summary = Counter()
    failures: list[str] = []
    for item in results:
        if isinstance(item, Exception):
            summary["exception"] += 1
            failures.append(repr(item))
            continue
        if item["ok"]:
            summary["ok"] += 1
        else:
            summary["failed"] += 1
            failures.append(str(item["error"] or item["content"]))

    print(
        json.dumps(
            {
                "server_url": server_url,
                "requested_concurrency": concurrency,
                "model_refs": model_refs,
                "summary": dict(summary),
                "failures": failures[:10],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not failures else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:5668/sse")
    parser.add_argument("--model-refs", default="EFORM,BAND_GAP_MFI")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    raise SystemExit(asyncio.run(main_async(parse_args())))


if __name__ == "__main__":
    main()
