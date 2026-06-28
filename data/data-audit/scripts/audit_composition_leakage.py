#!/usr/bin/env python3
import json
from collections import Counter
from typing import Any

from leakage_common import (
    DETAILS_DIR,
    SUMMARY_DIR,
    TEST_PATH,
    TRAIN_PATH,
    iter_jsonl,
    load_entries,
    normalize_formula_raw,
    parse_cif_metadata,
    write_json,
    write_jsonl,
)


def build_train_sets() -> tuple[set[str], set[str], set[str], Counter[str]]:
    mp_ids: set[str] = set()
    cif_hashes: set[str] = set()
    formula_raws: set[str] = set()
    formula_counts: Counter[str] = Counter()

    for _, row in iter_jsonl(TRAIN_PATH):
        mp_ids.add(row["mp_id"])
        cif_hashes.add(row["cif_split_meta"]["target_cif_hash"])

        meta = parse_cif_metadata(row["cif_split_meta"]["target_cif"])
        formula = normalize_formula_raw(meta["formula_raw"])
        if formula:
            formula_raws.add(formula)
            formula_counts[formula] += 1

    return mp_ids, cif_hashes, formula_raws, formula_counts


def main() -> None:
    train_mp_ids, train_cif_hashes, train_formula_raws, train_formula_counts = build_train_sets()
    test_entries = load_entries(TEST_PATH, split="test", need_structure=False)

    counters = {
        "mp_id_overlap": 0,
        "exact_cif_hash_overlap": 0,
        "composition_overlap": 0,
    }
    details: list[dict[str, Any]] = []

    for entry in test_entries:
        mp_id_overlap = entry.mp_id in train_mp_ids
        exact_cif_hash_overlap = entry.cif_hash in train_cif_hashes
        composition_overlap = (
            entry.formula_raw_normalized in train_formula_raws
            if entry.formula_raw_normalized
            else False
        )

        counters["mp_id_overlap"] += int(mp_id_overlap)
        counters["exact_cif_hash_overlap"] += int(exact_cif_hash_overlap)
        counters["composition_overlap"] += int(composition_overlap)

        details.append(
            {
                "test_row_index": entry.row_index,
                "test_mp_id": entry.mp_id,
                "test_cif_hash": entry.cif_hash,
                "formula_raw": entry.formula_raw,
                "formula_raw_normalized": entry.formula_raw_normalized,
                "mp_id_overlap": mp_id_overlap,
                "exact_cif_hash_overlap": exact_cif_hash_overlap,
                "composition_overlap": composition_overlap,
                "train_rows_with_same_raw_formula": train_formula_counts.get(
                    entry.formula_raw_normalized or "", 0
                ),
            }
        )

    test_size = len(test_entries)
    summary = {
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "test_rows": test_size,
        "definitions": {
            "mp_id_overlap": "test mp_id appears in the train split",
            "exact_cif_hash_overlap": "test CIF hash appears in the train split",
            "composition_overlap": "test raw structural formula string appears in the train split after whitespace removal only",
        },
        "counts": counters,
        "ratios_over_test": {key: round(value / test_size, 6) for key, value in counters.items()},
        "examples": {
            key: [row for row in details if row[key]][:20]
            for key in counters
        },
    }

    summary_path = SUMMARY_DIR / "composition_leakage_summary.json"
    details_path = DETAILS_DIR / "composition_leakage_details.jsonl"
    write_json(summary_path, summary)
    write_jsonl(details_path, details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {summary_path}")
    print(f"wrote: {details_path}")


if __name__ == "__main__":
    main()
