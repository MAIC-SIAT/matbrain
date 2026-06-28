from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import AsyncIterator
from urllib.parse import parse_qs, urlparse

import httpx
import uvicorn
from httpx_sse import aconnect_sse
from sse_starlette import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.routing import Route


def _env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]


SERVICE_NAME = os.getenv("SERVICE_NAME", "mcp-service")
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
SERVER_PORT = int(os.getenv("SERVER_PORT", "5668"))
BACKEND_URLS = _env_list("BACKEND_URLS")
BACKEND_SSE_PATH = os.getenv("BACKEND_SSE_PATH", "/sse")
BACKEND_MESSAGE_PATH = os.getenv("BACKEND_MESSAGE_PATH", "/messages/")
BACKEND_HEALTH_PATH = os.getenv("BACKEND_HEALTH_PATH", "/health")
PUBLIC_SSE_PATH = os.getenv("PUBLIC_SSE_PATH", "/sse")
PUBLIC_MESSAGE_PATH = os.getenv("PUBLIC_MESSAGE_PATH", "/messages/")
PUBLIC_HEALTH_PATH = os.getenv("PUBLIC_HEALTH_PATH", "/health")
PUBLIC_LOAD_PATH = os.getenv("PUBLIC_LOAD_PATH", "/load")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
CONNECT_TIMEOUT = float(os.getenv("ROUTER_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("ROUTER_READ_TIMEOUT", "600"))
WRITE_TIMEOUT = float(os.getenv("ROUTER_WRITE_TIMEOUT", "60"))
HEALTH_TIMEOUT = float(os.getenv("ROUTER_HEALTH_TIMEOUT", "5"))
RECENT_EVENT_LIMIT = int(os.getenv("ROUTER_RECENT_EVENT_LIMIT", "200"))

logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO), format="%(message)s")
logger = logging.getLogger("mcp-sse-router")


def _extract_session_id(endpoint_url: str) -> str | None:
    params = parse_qs(urlparse(endpoint_url).query)
    return params.get("session_id", [None])[0] or params.get("sessionId", [None])[0]


def _extract_request_meta(body: bytes) -> dict[str, str | None]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {"rpc_method": None, "tool_name": None}
    if not isinstance(payload, dict):
        return {"rpc_method": None, "tool_name": None}
    method = payload.get("method")
    params = payload.get("params") or {}
    tool_name = None
    if isinstance(params, dict):
        tool_name = params.get("name") or params.get("tool") or params.get("tool_name")
    return {
        "rpc_method": str(method) if method is not None else None,
        "tool_name": str(tool_name) if tool_name is not None else None,
    }


def _http_timeout(connect: float | None = None, read: float | None = None) -> httpx.Timeout:
    return httpx.Timeout(
        connect=CONNECT_TIMEOUT if connect is None else connect,
        read=READ_TIMEOUT if read is None else read,
        write=WRITE_TIMEOUT,
        pool=CONNECT_TIMEOUT if connect is None else connect,
    )


@dataclass
class RouterState:
    backends: list[str]
    sessions: dict[str, str] = field(default_factory=dict)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    rr_index: int = 0
    active_sse_by_backend: dict[str, int] = field(default_factory=dict)
    inflight_post_by_backend: dict[str, int] = field(default_factory=dict)
    total_sse_by_backend: dict[str, int] = field(default_factory=dict)
    total_post_by_backend: dict[str, int] = field(default_factory=dict)
    failed_connects_by_backend: dict[str, int] = field(default_factory=dict)
    failed_posts_by_backend: dict[str, int] = field(default_factory=dict)
    recent_events: deque[dict[str, object]] = field(default_factory=lambda: deque(maxlen=RECENT_EVENT_LIMIT))

    def __post_init__(self) -> None:
        if not self.backends:
            raise RuntimeError("BACKEND_URLS is empty")
        for backend in self.backends:
            self.active_sse_by_backend.setdefault(backend, 0)
            self.inflight_post_by_backend.setdefault(backend, 0)
            self.total_sse_by_backend.setdefault(backend, 0)
            self.total_post_by_backend.setdefault(backend, 0)
            self.failed_connects_by_backend.setdefault(backend, 0)
            self.failed_posts_by_backend.setdefault(backend, 0)

    async def ordered_backends(self) -> list[str]:
        async with self.lock:
            if not self.backends:
                return []
            start = self.rr_index % len(self.backends)
            self.rr_index = (self.rr_index + 1) % len(self.backends)
            return self.backends[start:] + self.backends[:start]

    async def bind_session(self, session_id: str, backend: str) -> None:
        async with self.lock:
            self.sessions[session_id] = backend

    async def get_backend_for_session(self, session_id: str) -> str | None:
        async with self.lock:
            return self.sessions.get(session_id)

    async def begin_sse(self, backend: str) -> None:
        async with self.lock:
            self.active_sse_by_backend[backend] = self.active_sse_by_backend.get(backend, 0) + 1
            self.total_sse_by_backend[backend] = self.total_sse_by_backend.get(backend, 0) + 1

    async def end_sse(self, backend: str) -> None:
        async with self.lock:
            self.active_sse_by_backend[backend] = max(0, self.active_sse_by_backend.get(backend, 0) - 1)

    async def begin_post(self, backend: str) -> None:
        async with self.lock:
            self.inflight_post_by_backend[backend] = self.inflight_post_by_backend.get(backend, 0) + 1
            self.total_post_by_backend[backend] = self.total_post_by_backend.get(backend, 0) + 1

    async def end_post(self, backend: str) -> None:
        async with self.lock:
            self.inflight_post_by_backend[backend] = max(0, self.inflight_post_by_backend.get(backend, 0) - 1)

    async def note_failed_connect(self, backend: str, error: str) -> None:
        async with self.lock:
            self.failed_connects_by_backend[backend] = self.failed_connects_by_backend.get(backend, 0) + 1
            self.recent_events.append(
                {
                    "ts": time.time(),
                    "kind": "backend_connect_failed",
                    "backend": backend,
                    "error": error,
                }
            )

    async def note_failed_post(self, backend: str, session_id: str, rpc_method: str | None, tool_name: str | None, error: str) -> None:
        async with self.lock:
            self.failed_posts_by_backend[backend] = self.failed_posts_by_backend.get(backend, 0) + 1
            self.recent_events.append(
                {
                    "ts": time.time(),
                    "kind": "post_failed",
                    "backend": backend,
                    "session_id": session_id,
                    "rpc_method": rpc_method,
                    "tool_name": tool_name,
                    "error": error,
                }
            )

    async def note_event(self, event: dict[str, object]) -> None:
        async with self.lock:
            self.recent_events.append({"ts": time.time(), **event})

    async def load_snapshot(self) -> dict[str, object]:
        async with self.lock:
            sessions_by_backend = {backend: 0 for backend in self.backends}
            for backend in self.sessions.values():
                sessions_by_backend[backend] = sessions_by_backend.get(backend, 0) + 1
            return {
                "service": f"{SERVICE_NAME}-router",
                "backend_count": len(self.backends),
                "sessions_tracked": len(self.sessions),
                "backends": [
                    {
                        "backend": backend,
                        "active_sse": self.active_sse_by_backend.get(backend, 0),
                        "tracked_sessions": sessions_by_backend.get(backend, 0),
                        "inflight_post": self.inflight_post_by_backend.get(backend, 0),
                        "total_sse": self.total_sse_by_backend.get(backend, 0),
                        "total_post": self.total_post_by_backend.get(backend, 0),
                        "failed_connects": self.failed_connects_by_backend.get(backend, 0),
                        "failed_posts": self.failed_posts_by_backend.get(backend, 0),
                    }
                    for backend in self.backends
                ],
                "recent_events": list(self.recent_events),
            }


state = RouterState(BACKEND_URLS)


async def root(_: Request) -> JSONResponse:
    return JSONResponse(
        {
            "service": f"{SERVICE_NAME}-router",
            "status": "running",
            "public_endpoints": {
                "sse": PUBLIC_SSE_PATH,
                "messages": PUBLIC_MESSAGE_PATH,
                "health": PUBLIC_HEALTH_PATH,
                "load": PUBLIC_LOAD_PATH,
            },
            "backend_count": len(state.backends),
            "backends": state.backends,
        }
    )


async def health(_: Request) -> JSONResponse:
    checks = []
    healthy = 0
    async with httpx.AsyncClient(timeout=_http_timeout(connect=HEALTH_TIMEOUT, read=HEALTH_TIMEOUT), trust_env=False) as client:
        for backend in state.backends:
            url = f"{backend}{BACKEND_HEALTH_PATH}"
            ok = False
            detail = None
            try:
                resp = await client.get(url)
                ok = resp.status_code < 400
                detail = resp.json() if ok else resp.text[:500]
            except Exception as exc:  # noqa: BLE001
                detail = repr(exc)
            checks.append({"backend": backend, "ok": ok, "detail": detail})
            healthy += int(ok)
    status = 200 if healthy > 0 else 503
    return JSONResponse(
        {
            "service": f"{SERVICE_NAME}-router",
            "status": "healthy" if healthy > 0 else "unhealthy",
            "healthy_backends": healthy,
            "backend_count": len(state.backends),
            "checks": checks,
        },
        status_code=status,
    )


async def load(_: Request) -> JSONResponse:
    return JSONResponse(await state.load_snapshot())


async def sse(request: Request) -> EventSourceResponse:
    async def event_generator() -> AsyncIterator[dict[str, str]]:
        ordered = await state.ordered_backends()
        if not ordered:
            raise RuntimeError("router has no backends configured")
        last_exc: Exception | None = None
        for backend in ordered:
            backend_sse_url = f"{backend}{BACKEND_SSE_PATH}"
            logger.info(f"[router] new SSE session -> backend={backend_sse_url}")
            try:
                await state.begin_sse(backend)
                await state.note_event({"kind": "sse_open", "backend": backend})
                async with httpx.AsyncClient(timeout=_http_timeout(), trust_env=False, follow_redirects=True) as client:
                    async with aconnect_sse(client, "GET", backend_sse_url) as event_source:
                        event_source.response.raise_for_status()
                        async for sse_event in event_source.aiter_sse():
                            event_name = sse_event.event or "message"
                            data = sse_event.data or ""
                            if event_name == "endpoint":
                                endpoint_url = f"{backend}{data}" if data.startswith("/") else data
                                session_id = _extract_session_id(endpoint_url)
                                if not session_id:
                                    raise RuntimeError(
                                        f"router could not extract session_id from endpoint event: {endpoint_url}"
                                    )
                                await state.bind_session(session_id, backend)
                                data = f"{PUBLIC_MESSAGE_PATH}?session_id={session_id}"
                                logger.info(f"[router] bound session_id={session_id} -> backend={backend}")
                                await state.note_event(
                                    {
                                        "kind": "session_bound",
                                        "backend": backend,
                                        "session_id": session_id,
                                    }
                                )
                            yield {"event": event_name, "data": data}
                return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                await state.note_failed_connect(backend, repr(exc))
                logger.warning(f"[router] backend connect failed backend={backend_sse_url} error={exc!r}")
                continue
            finally:
                await state.end_sse(backend)
                await state.note_event({"kind": "sse_close", "backend": backend})
        if last_exc is not None:
            raise last_exc

    return EventSourceResponse(event_generator())


async def post_message(request: Request) -> Response:
    session_id = request.query_params.get("session_id") or request.query_params.get("sessionId")
    if not session_id:
        return PlainTextResponse("session_id is required", status_code=400)
    backend = await state.get_backend_for_session(session_id)
    if not backend:
        return PlainTextResponse("unknown session_id", status_code=404)

    target_url = f"{backend}{BACKEND_MESSAGE_PATH}?session_id={session_id}"
    body = await request.body()
    request_meta = _extract_request_meta(body)
    headers = {}
    content_type = request.headers.get("content-type")
    if content_type:
        headers["content-type"] = content_type
    logger.info(
        f"[router] POST session_id={session_id} backend={backend} rpc_method={request_meta['rpc_method']} tool={request_meta['tool_name']}"
    )
    await state.begin_post(backend)
    await state.note_event(
        {
            "kind": "post_start",
            "backend": backend,
            "session_id": session_id,
            "rpc_method": request_meta["rpc_method"],
            "tool_name": request_meta["tool_name"],
        }
    )
    try:
        async with httpx.AsyncClient(timeout=_http_timeout(), trust_env=False, follow_redirects=True) as client:
            resp = await client.post(target_url, content=body, headers=headers)
    except Exception as exc:  # noqa: BLE001
        await state.note_failed_post(
            backend,
            session_id,
            request_meta["rpc_method"],
            request_meta["tool_name"],
            repr(exc),
        )
        raise
    finally:
        await state.end_post(backend)
        await state.note_event(
            {
                "kind": "post_end",
                "backend": backend,
                "session_id": session_id,
                "rpc_method": request_meta["rpc_method"],
                "tool_name": request_meta["tool_name"],
            }
        )
    hop_by_hop = {"content-length", "transfer-encoding", "connection", "content-encoding"}
    forwarded_headers = {k: v for k, v in resp.headers.items() if k.lower() not in hop_by_hop}
    return Response(content=resp.content, status_code=resp.status_code, headers=forwarded_headers)


app = Starlette(
    debug=False,
    routes=[
        Route("/", root),
        Route(PUBLIC_HEALTH_PATH, health),
        Route(PUBLIC_LOAD_PATH, load),
        Route(PUBLIC_SSE_PATH, sse),
        Route(PUBLIC_MESSAGE_PATH, post_message, methods=["POST"]),
    ],
)


if __name__ == "__main__":
    uvicorn.run(app, host=SERVER_HOST, port=SERVER_PORT, log_config=None, access_log=True)
