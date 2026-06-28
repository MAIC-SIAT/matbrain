"""CLI entry point.

Usage:
    .venv/bin/python agent/scripts/run_query.py "查询 CsPbBr3 的晶体结构并预测其形成能"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

# Make `matbrain` importable when running this file directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from matbrain.config import get_settings
from matbrain.graph import build_graph
from matbrain.state import initial_state


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stderr,
    )


async def amain(query: str, dump_state: bool, recursion_limit: int) -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    graph, client = await build_graph(settings)

    print(f"[matbrain] {len(client.registry)} tools discovered across {len(client.endpoints)} MCP servers", file=sys.stderr)

    state = initial_state(query)
    final = await graph.ainvoke(state, config={"recursion_limit": recursion_limit})

    print("\n========== FINAL ANSWER ==========")
    print(final.get("final_answer") or "(no answer)")
    print("==================================\n")

    if dump_state:
        print("\n========== FULL STATE ==========", file=sys.stderr)
        print(json.dumps(final, indent=2, ensure_ascii=False, default=str), file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Run a single MatBrain query end-to-end.")
    p.add_argument("query", help="research query in natural language")
    p.add_argument("--dump-state", action="store_true", help="print final state JSON to stderr")
    p.add_argument(
        "--recursion-limit",
        type=int,
        default=30,
        help="LangGraph recursion limit (each iteration = executor + reasoner = 2 steps; default 30)",
    )
    args = p.parse_args()
    asyncio.run(amain(args.query, args.dump_state, args.recursion_limit))


if __name__ == "__main__":
    main()
