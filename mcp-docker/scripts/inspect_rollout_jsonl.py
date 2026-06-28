#!/usr/bin/env python3
"""Inspect verl rollout JSONL dumps."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def tail_path(path: Path) -> Path:
    if path.is_file():
        return path
    files = sorted(path.glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise SystemExit(f"No jsonl files found under {path}")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, nargs="?", default=Path("/workspace/run/rollouts/t1_mcp_grpo_sglang"))
    parser.add_argument("--show", type=int, default=3)
    parser.add_argument("--chars", type=int, default=2500)
    args = parser.parse_args()

    path = tail_path(args.path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    print(f"file: {path}")
    print(f"rows: {len(rows)}")
    if not rows:
        return

    answer_count = 0
    tool_call_counts = []
    output_lengths = []
    scores = []
    for row in rows:
        out = str(row.get("output", ""))
        output_lengths.append(len(out))
        tool_call_counts.append(out.count("<tool_call>"))
        if "<answer>" in out and "</answer>" in out:
            answer_count += 1
        try:
            scores.append(float(row.get("score", 0.0)))
        except Exception:
            scores.append(0.0)

    print(f"score mean/min/max: {sum(scores)/len(scores):.4f} / {min(scores):.4f} / {max(scores):.4f}")
    print(f"answer rate: {answer_count}/{len(rows)} = {answer_count/len(rows):.3f}")
    print(f"tool calls mean/max: {sum(tool_call_counts)/len(tool_call_counts):.2f} / {max(tool_call_counts)}")
    print(f"output chars mean/max: {sum(output_lengths)/len(output_lengths):.0f} / {max(output_lengths)}")

    ranked = sorted(enumerate(rows), key=lambda item: float(item[1].get("score", 0.0)), reverse=True)
    for rank, (idx, row) in enumerate(ranked[: args.show], start=1):
        out = str(row.get("output", ""))
        print("\n" + "=" * 100)
        print(f"sample rank={rank} row={idx} score={row.get('score')}")
        print(
            f"has_answer={'<answer>' in out and '</answer>' in out} "
            f"tool_calls={out.count('<tool_call>')} "
            f"think_blocks={len(re.findall(r'<think>.*?</think>', out, re.DOTALL))}"
        )
        print("-" * 100)
        print(out[: args.chars])
        if len(out) > args.chars:
            print("\n...[truncated]...\n")
            print(out[-args.chars:])


if __name__ == "__main__":
    main()
