#!/usr/bin/env python3
"""Validate that Mat-T1 RL parquet files contain the expected prompt contract."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow.parquet as pq


REQUIRED_SYSTEM_SNIPPETS = [
    "Every intermediate tool-use step must use <think> followed by <tool_call>.",
    "produce a final <think> followed by a final <answer>.",
]


def read_first_system_prompt(parquet_path: Path) -> str:
    table = pq.read_table(parquet_path, columns=["prompt"])
    if len(table) == 0:
        raise ValueError(f"{parquet_path} is empty")
    prompt = table.column("prompt")[0].as_py()
    if not isinstance(prompt, list) or not prompt:
        raise ValueError(f"{parquet_path} prompt column is not a chat-message list")
    first_msg = prompt[0]
    if not isinstance(first_msg, dict) or first_msg.get("role") != "system":
        raise ValueError(f"{parquet_path} first prompt message is not a system message")
    content = first_msg.get("content")
    if not isinstance(content, str):
        raise ValueError(f"{parquet_path} system content is not a string")
    return content


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("parquet", nargs="+", type=Path)
    args = parser.parse_args()

    failures: list[str] = []
    for parquet_path in args.parquet:
        system_prompt = read_first_system_prompt(parquet_path)
        missing = [snippet for snippet in REQUIRED_SYSTEM_SNIPPETS if snippet not in system_prompt]
        if missing:
            failures.append(
                f"{parquet_path}: missing required system-prompt snippets: {missing}"
            )

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        print(
            "Regenerate RL data with:\n"
            "  python3 train/rl/prepare_t1_rl_data.py --output-dir dataset/rl/t1_mcp --max-train-per-task 30000\n"
            "  python3 train/rl/convert_jsonl_to_verl_parquet.py dataset/rl/t1_mcp/t1_mcp_train.jsonl dataset/rl/t1_mcp/t1_mcp_train.parquet\n"
            "  python3 train/rl/convert_jsonl_to_verl_parquet.py dataset/rl/t1_mcp/t1_mcp_eval.jsonl dataset/rl/t1_mcp/t1_mcp_eval.parquet",
            file=sys.stderr,
        )
        return 2

    for parquet_path in args.parquet:
        print(f"OK: {parquet_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
