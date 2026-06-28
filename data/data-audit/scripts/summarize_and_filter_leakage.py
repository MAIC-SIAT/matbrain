#!/usr/bin/env python3
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from leakage_common import (
    CLEAN_TEST_DIR,
    DETAILS_DIR,
    SUMMARY_DIR,
    TEST_PATH,
    iter_jsonl,
    write_json,
    write_jsonl,
)


COMPOSITION_DETAILS = DETAILS_DIR / "composition_leakage_details.jsonl"
STRUCTURE_DETAILS = DETAILS_DIR / "structure_similarity_leakage_details.jsonl"
PROTOTYPE_DETAILS = DETAILS_DIR / "prototype_leakage_details.jsonl"

ROW_FLAGS_PATH = DETAILS_DIR / "leakage_row_flags.jsonl"
SUMMARY_JSON_PATH = SUMMARY_DIR / "leakage_overlap_report_summary.json"
SUMMARY_CSV_PATH = SUMMARY_DIR / "leakage_metric_summary_table.csv"
TOP_ITEMS_JSON_PATH = SUMMARY_DIR / "leakage_top_overlap_items.json"
TOP_ITEMS_CSV_PATH = SUMMARY_DIR / "leakage_top_overlap_items.csv"

STRICT_CLEAN_TEST_PATH = CLEAN_TEST_DIR / "mp_cif_design_property_test_no_identity_structure_full_protostructure_overlap.jsonl"
CONSERVATIVE_CLEAN_TEST_PATH = CLEAN_TEST_DIR / "mp_cif_design_property_test_no_any_audit_overlap.jsonl"


METRICS = [
    {
        "name": "mp_id_overlap",
        "source": "composition",
        "flag": "mp_id_overlap",
        "top_key": "test_mp_id",
        "label": "MP-ID",
        "family": "identity",
    },
    {
        "name": "exact_cif_hash_overlap",
        "source": "composition",
        "flag": "exact_cif_hash_overlap",
        "top_key": "test_cif_hash",
        "label": "Exact CIF hash",
        "family": "identity",
    },
    {
        "name": "composition_overlap",
        "source": "composition",
        "flag": "composition_overlap",
        "top_key": "formula_raw_normalized",
        "label": "Raw formula",
        "family": "composition",
    },
    {
        "name": "structure_similarity_overlap",
        "source": "structure",
        "flag": "structure_similarity_overlap",
        "top_key": "matched_train_mp_id",
        "label": "Fingerprint-retrieved StructureMatcher match",
        "family": "structure",
    },
    {
        "name": "protostructure_label_overlap",
        "source": "prototype",
        "flag": "protostructure_label_overlap",
        "top_key": "protostructure_label",
        "label": "Matbench Discovery-style full protostructure label",
        "family": "prototype_full",
    },
]

STRICT_REMOVE_FLAGS = {
    "mp_id_overlap",
    "exact_cif_hash_overlap",
    "structure_similarity_overlap",
    "protostructure_label_overlap",
}
CONSERVATIVE_REMOVE_FLAGS = {metric["name"] for metric in METRICS}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_details() -> dict[str, dict[int, dict[str, Any]]]:
    paths = {
        "composition": COMPOSITION_DETAILS,
        "structure": STRUCTURE_DETAILS,
        "prototype": PROTOTYPE_DETAILS,
    }
    details: dict[str, dict[int, dict[str, Any]]] = {}
    for name, path in paths.items():
        rows_by_index: dict[int, dict[str, Any]] = {}
        for row in load_jsonl(path):
            rows_by_index[int(row["test_row_index"])] = row
        details[name] = rows_by_index
    return details


def counter_top(counter: Counter[str], n: int = 20) -> list[dict[str, Any]]:
    return [{"item": item, "count": count} for item, count in counter.most_common(n)]


def write_summary_csv(rows: Iterable[dict[str, Any]]) -> None:
    fieldnames = ["metric", "count", "ratio_over_test", "description"]
    with SUMMARY_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_top_items_csv(rows_by_metric: dict[str, list[dict[str, Any]]]) -> None:
    fieldnames = ["metric", "rank", "item", "count"]
    with TOP_ITEMS_CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for metric, rows in rows_by_metric.items():
            for rank, row in enumerate(rows, start=1):
                writer.writerow(
                    {
                        "metric": metric,
                        "rank": rank,
                        "item": row["item"],
                        "count": row["count"],
                    }
                )


def main() -> None:
    details = load_details()
    test_rows = list(iter_jsonl(TEST_PATH))
    test_size = len(test_rows)

    row_flags: list[dict[str, Any]] = []
    metric_counts: dict[str, int] = {}
    metric_top_items: dict[str, list[dict[str, Any]]] = {}

    counters_by_metric: dict[str, Counter[str]] = {metric["name"]: Counter() for metric in METRICS}
    row_indices_by_metric: dict[str, set[int]] = {metric["name"]: set() for metric in METRICS}

    for row_index, test_row in test_rows:
        comp = details["composition"].get(row_index, {})
        struct = details["structure"].get(row_index, {})
        proto = details["prototype"].get(row_index, {})
        source_rows = {"composition": comp, "structure": struct, "prototype": proto}

        flags: dict[str, bool] = {}
        for metric in METRICS:
            source = source_rows[metric["source"]]
            hit = bool(source.get(metric["flag"], False))
            flags[metric["name"]] = hit
            if hit:
                row_indices_by_metric[metric["name"]].add(row_index)
                top_value = source.get(metric["top_key"])
                if top_value is not None:
                    counters_by_metric[metric["name"]][str(top_value)] += 1

        strict_remove = any(flags[name] for name in STRICT_REMOVE_FLAGS)
        conservative_remove = any(flags[name] for name in CONSERVATIVE_REMOVE_FLAGS)
        row_flags.append(
            {
                "test_row_index": row_index,
                "test_mp_id": test_row.get("mp_id"),
                "test_cif_hash": test_row.get("cif_split_meta", {}).get("target_cif_hash"),
                "formula_raw_normalized": comp.get("formula_raw_normalized"),
                "protostructure_label": proto.get("protostructure_label"),
                "flags": flags,
                "remove_in_strict_clean_test": strict_remove,
                "remove_in_conservative_clean_test": conservative_remove,
            }
        )

    for metric in METRICS:
        name = metric["name"]
        metric_counts[name] = len(row_indices_by_metric[name])
        metric_top_items[name] = counter_top(counters_by_metric[name], n=20)

    strict_remove_indices = {
        row["test_row_index"] for row in row_flags if row["remove_in_strict_clean_test"]
    }
    conservative_remove_indices = {
        row["test_row_index"] for row in row_flags if row["remove_in_conservative_clean_test"]
    }

    strict_clean_rows = [row for idx, row in test_rows if idx not in strict_remove_indices]
    conservative_clean_rows = [row for idx, row in test_rows if idx not in conservative_remove_indices]

    summary_rows = [
        {
            "metric": metric["name"],
            "count": metric_counts[metric["name"]],
            "ratio_over_test": round(metric_counts[metric["name"]] / test_size, 6),
            "description": metric["label"],
        }
        for metric in METRICS
    ]

    summary = {
        "test_path": str(TEST_PATH),
        "test_rows": test_size,
        "metric_summary": summary_rows,
        "clean_test_definitions": {
            "strict_clean_test": {
                "remove_flags": sorted(STRICT_REMOVE_FLAGS),
                "removed_rows": len(strict_remove_indices),
                "remaining_rows": len(strict_clean_rows),
                "output_path": str(STRICT_CLEAN_TEST_PATH),
            },
            "conservative_clean_test": {
                "remove_flags": sorted(CONSERVATIVE_REMOVE_FLAGS),
                "removed_rows": len(conservative_remove_indices),
                "remaining_rows": len(conservative_clean_rows),
                "output_path": str(CONSERVATIVE_CLEAN_TEST_PATH),
            },
        },
        "overlap_set_sizes": {
            "strict_union": len(strict_remove_indices),
            "conservative_union": len(conservative_remove_indices),
        },
        "outputs": {
            "row_flags": str(ROW_FLAGS_PATH),
            "metric_summary_csv": str(SUMMARY_CSV_PATH),
            "top_items_json": str(TOP_ITEMS_JSON_PATH),
            "top_items_csv": str(TOP_ITEMS_CSV_PATH),
        },
    }

    write_jsonl(ROW_FLAGS_PATH, row_flags)
    write_json(SUMMARY_JSON_PATH, summary)
    write_json(TOP_ITEMS_JSON_PATH, metric_top_items)
    write_summary_csv(summary_rows)
    write_top_items_csv(metric_top_items)
    write_jsonl(STRICT_CLEAN_TEST_PATH, strict_clean_rows)
    write_jsonl(CONSERVATIVE_CLEAN_TEST_PATH, conservative_clean_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
