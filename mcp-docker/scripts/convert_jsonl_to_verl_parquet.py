#!/usr/bin/env python3
"""Convert MatBrain RL JSONL data into a verl-friendly parquet file.

The source JSONL keeps rich Python/JSON objects for readability.  This converter
normalizes the variable ground-truth dictionaries into JSON strings while keeping
the chat prompt and fixed extra_info metadata as structured parquet columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


PROMPT_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("role", pa.string()),
            pa.field("content", pa.string()),
        ]
    )
)

EXTRA_INFO_TYPE = pa.struct(
    [
        pa.field("source_path", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("source_pair_index", pa.int64()),
        pa.field("mp_id", pa.string()),
        pa.field("material_formula_key", pa.string()),
        pa.field("tool_profile", pa.string()),
        pa.field("expected_output", pa.string()),
        pa.field("reward_spec", pa.string()),
        pa.field("input_cif", pa.string()),
    ]
)

REWARD_MODEL_TYPE = pa.struct(
    [
        pa.field("style", pa.string()),
        pa.field("ground_truth", pa.string()),
    ]
)

SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string()),
        pa.field("ability", pa.string()),
        pa.field("prompt", PROMPT_TYPE),
        pa.field("ground_truth", pa.string()),
        pa.field("reward_model", REWARD_MODEL_TYPE),
        pa.field("agent_name", pa.string()),
        pa.field("extra_info", EXTRA_INFO_TYPE),
    ]
)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    data_source = str(row.get("data_source") or "")
    ground_truth = _json_dumps(row.get("ground_truth") or {})
    extra = row.get("extra_info") or {}
    return {
        "data_source": data_source,
        "ability": data_source,
        "prompt": [
            {
                "role": str(msg.get("role") or ""),
                "content": str(msg.get("content") or ""),
            }
            for msg in (row.get("prompt") or [])
            if isinstance(msg, dict)
        ],
        "ground_truth": ground_truth,
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "agent_name": "tool_agent",
        "extra_info": {
            "source_path": str(extra.get("source_path") or ""),
            "source_row_index": int(extra.get("source_row_index") or 0),
            "source_pair_index": int(extra.get("source_pair_index") or 0),
            "mp_id": str(extra.get("mp_id") or ""),
            "material_formula_key": str(extra.get("material_formula_key") or ""),
            "tool_profile": str(extra.get("tool_profile") or ""),
            "expected_output": str(extra.get("expected_output") or ""),
            "reward_spec": _json_dumps(extra.get("reward_spec") or {}),
            "input_cif": str(extra.get("input_cif") or ""),
        },
    }


def convert_jsonl_to_parquet(input_path: Path, output_path: Path, batch_size: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    rows: list[dict[str, Any]] = []
    count = 0
    try:
        with input_path.open("r", encoding="utf-8") as src:
            for line in src:
                if not line.strip():
                    continue
                rows.append(normalize_row(json.loads(line)))
                if len(rows) >= batch_size:
                    table = pa.Table.from_pylist(rows, schema=SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(output_path, SCHEMA, compression="zstd")
                    writer.write_table(table)
                    count += len(rows)
                    rows = []
            if rows:
                table = pa.Table.from_pylist(rows, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, SCHEMA, compression="zstd")
                writer.write_table(table)
                count += len(rows)
    finally:
        if writer is not None:
            writer.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    count = convert_jsonl_to_parquet(args.input, args.output, args.batch_size)
    print(json.dumps({"input": str(args.input), "output": str(args.output), "rows": count}, ensure_ascii=False))


if __name__ == "__main__":
    main()
