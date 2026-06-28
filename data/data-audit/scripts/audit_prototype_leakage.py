#!/usr/bin/env python3
import argparse
import json
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from pymatgen.analysis.prototypes import get_protostructure_label
from tqdm import tqdm

from leakage_common import (
    CACHE_DIR,
    DETAILS_DIR,
    SUMMARY_DIR,
    TEST_PATH,
    TRAIN_PATH,
    iter_jsonl,
    parse_structure,
    read_symmetrized_cif_text,
    suppress_fd_stderr,
    symmetrized_cif_path,
    write_json,
    write_jsonl,
)


CACHE_PATH = CACHE_DIR / "train_protostructure_label_cache.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit prototype-level test-to-train leakage using pymatgen "
            "AFLOW-style protostructure labels generated with spglib."
        )
    )
    parser.add_argument("--workers", type=int, default=1, help="Number of worker processes.")
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Rebuild the train protostructure-label cache instead of resuming from it.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional debug limit on unique train mp-ids to process.",
    )
    return parser.parse_args()


def is_valid_protostructure_label(label: str | None) -> bool:
    if not label or ":" not in label:
        return False
    aflow_label, chemsys = label.split(":", 1)
    return bool(aflow_label.strip() and chemsys.strip())


def get_spglib_protostructure_label_from_cif_text(cif_text: str | None) -> str | None:
    if cif_text is None:
        return None

    structure = parse_structure(cif_text)
    if structure is None:
        return None

    try:
        with warnings.catch_warnings(), suppress_fd_stderr(True):
            warnings.simplefilter("ignore")
            label = get_protostructure_label(
                structure,
                method="spglib",
                raise_errors=False,
                init_symprec=0.1,
                fallback_symprec=1e-5,
            )
    except Exception:
        return None

    if not is_valid_protostructure_label(label):
        return None
    return label


def compute_cache_record(mp_id: str) -> dict[str, Any]:
    cif_text = read_symmetrized_cif_text(mp_id)
    label = get_spglib_protostructure_label_from_cif_text(cif_text)
    return {
        "mp_id": mp_id,
        "symmetrized_cif_path": str(symmetrized_cif_path(mp_id)),
        "symmetrized_cif_found": cif_text is not None,
        "protostructure_label": label,
        "protostructure_label_valid": bool(label),
    }


def load_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}

    cache: dict[str, dict[str, Any]] = {}
    with cache_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            cache[row["mp_id"]] = row
    return cache


def collect_unique_train_mp_ids(existing_cache: dict[str, dict[str, Any]], limit: int | None) -> tuple[list[str], int]:
    missing_mp_ids: list[str] = []
    cached_seen: set[str] = set(existing_cache)
    all_seen: set[str] = set()

    print("Scanning train split for unique mp-ids.", flush=True)
    for _, row in tqdm(iter_jsonl(TRAIN_PATH), desc="Scanning train rows", unit="row"):
        mp_id = row["mp_id"]
        if mp_id in all_seen:
            continue
        all_seen.add(mp_id)
        if mp_id in cached_seen:
            continue
        missing_mp_ids.append(mp_id)
        if limit is not None and len(missing_mp_ids) >= limit:
            break

    return missing_mp_ids, len(all_seen)


def build_or_resume_train_cache(
    cache_path: Path, workers: int, force: bool, limit: int | None
) -> tuple[dict[str, dict[str, Any]], int]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if force and cache_path.exists():
        cache_path.unlink()

    cache = load_cache(cache_path)
    print(f"Loaded protostructure-label cache: {len(cache)} cached train mp-ids.", flush=True)

    missing_mp_ids, unique_train_mp_ids_seen = collect_unique_train_mp_ids(cache, limit=limit)
    print(
        f"Loaded protostructure-label cache: {len(cache)}/{unique_train_mp_ids_seen} "
        "scanned unique train mp-ids cached.",
        flush=True,
    )
    if not missing_mp_ids:
        print("Protostructure-label cache is complete for scanned train mp-ids.", flush=True)
        return cache, unique_train_mp_ids_seen

    print(
        f"Building protostructure labels for {len(missing_mp_ids)} missing train mp-ids "
        f"with workers={workers}.",
        flush=True,
    )
    mode = "a" if cache else "w"
    with cache_path.open(mode, encoding="utf-8") as handle:
        if workers > 1:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                results = executor.map(compute_cache_record, missing_mp_ids, chunksize=8)
                iterator = tqdm(
                    results,
                    total=len(missing_mp_ids),
                    desc="Building protostructure cache",
                    unit="mp-id",
                    dynamic_ncols=True,
                    mininterval=0.5,
                )
                for row in iterator:
                    cache[row["mp_id"]] = row
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            for mp_id in tqdm(
                missing_mp_ids,
                total=len(missing_mp_ids),
                desc="Building protostructure cache",
                unit="mp-id",
                dynamic_ncols=True,
                mininterval=0.5,
            ):
                row = compute_cache_record(mp_id)
                cache[mp_id] = row
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    return cache, unique_train_mp_ids_seen


def collect_train_labels(cache: dict[str, dict[str, Any]]) -> set[str]:
    full_labels: set[str] = set()
    for row in cache.values():
        label = row.get("protostructure_label")
        if label:
            full_labels.add(label)
    return full_labels


def main() -> None:
    args = parse_args()
    workers = max(1, args.workers)
    train_cache, unique_train_mp_ids_seen = build_or_resume_train_cache(
        CACHE_PATH, workers, args.force_rebuild_cache, args.limit
    )
    train_full_labels = collect_train_labels(train_cache)

    details: list[dict[str, Any]] = []
    counters = {
        "protostructure_label_overlap": 0,
        "test_rows_with_valid_protostructure_label": 0,
        "test_rows_without_valid_protostructure_label": 0,
        "test_rows_missing_symmetrized_cif": 0,
        "train_cached_mp_ids_with_valid_protostructure_label": sum(
            int(bool(row.get("protostructure_label"))) for row in train_cache.values()
        ),
        "train_cached_mp_ids_without_valid_protostructure_label": sum(
            int(not bool(row.get("protostructure_label"))) for row in train_cache.values()
        ),
        "train_cached_mp_ids_missing_symmetrized_cif": sum(
            int(not row.get("symmetrized_cif_found", False)) for row in train_cache.values()
        ),
    }

    test_rows = list(iter_jsonl(TEST_PATH))
    print(f"Matching protostructure labels for {len(test_rows)} test rows.", flush=True)
    for row_index, row in tqdm(
        test_rows,
        desc="Matching test protostructures",
        unit="test",
        dynamic_ncols=True,
        mininterval=0.5,
    ):
        mp_id = row["mp_id"]
        cif_text = read_symmetrized_cif_text(mp_id)
        label = get_spglib_protostructure_label_from_cif_text(cif_text)
        full_label_overlap = bool(label and label in train_full_labels)

        counters["protostructure_label_overlap"] += int(full_label_overlap)
        counters["test_rows_with_valid_protostructure_label"] += int(bool(label))
        counters["test_rows_without_valid_protostructure_label"] += int(not bool(label))
        counters["test_rows_missing_symmetrized_cif"] += int(cif_text is None)

        details.append(
            {
                "test_row_index": row_index,
                "test_mp_id": mp_id,
                "test_cif_hash": row["cif_split_meta"]["target_cif_hash"],
                "symmetrized_cif_path": str(symmetrized_cif_path(mp_id)),
                "symmetrized_cif_found": cif_text is not None,
                "protostructure_label": label,
                "protostructure_label_overlap": full_label_overlap,
            }
        )

    test_size = len(details)
    valid_test = counters["test_rows_with_valid_protostructure_label"]
    summary = {
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "symmetrized_cif_source": str(SYMMETRIZED_CIF_DIR / "mp-*_symmetrized.cif"),
        "test_rows": test_size,
        "train_unique_mp_ids_scanned": unique_train_mp_ids_seen,
        "train_cached_mp_ids": len(train_cache),
        "train_unique_protostructure_labels": len(train_full_labels),
        "train_protostructure_cache_path": str(CACHE_PATH),
        "definitions": {
            "protostructure_label": (
                "Full pymatgen.analysis.prototypes.get_protostructure_label(..., "
                "method='spglib') result, formatted as aflow_sym_label:chemsys."
            ),
            "protostructure_label_overlap": (
                "Exact full protostructure_label appears in the train split. This is the primary "
                "prototype-aware leakage metric, following the Matbench Discovery-style AFLOW "
                "protostructure-label definition."
            ),
        },
        "counts": counters,
        "ratios_over_test": {
            "protostructure_label_overlap": round(counters["protostructure_label_overlap"] / test_size, 6),
            "protostructure_label_assignment_coverage": round(valid_test / test_size, 6),
        },
        "ratios_over_valid_test_labels": {
            "protostructure_label_overlap": round(counters["protostructure_label_overlap"] / valid_test, 6)
            if valid_test
            else None,
        },
        "examples": {
            "protostructure_label_overlap": [
                row for row in details if row["protostructure_label_overlap"]
            ][:20],
            "invalid_or_unassigned": [row for row in details if not row["protostructure_label"]][:20],
        },
    }

    summary_path = SUMMARY_DIR / "prototype_leakage_summary.json"
    details_path = DETAILS_DIR / "prototype_leakage_details.jsonl"
    write_json(summary_path, summary)
    write_jsonl(details_path, details)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {summary_path}")
    print(f"wrote: {details_path}")


if __name__ == "__main__":
    main()
