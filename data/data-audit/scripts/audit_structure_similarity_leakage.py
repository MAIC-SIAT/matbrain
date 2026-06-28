#!/usr/bin/env python3
import argparse
import json
import os
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
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

try:
    from sklearn.neighbors import NearestNeighbors
except Exception:  # pragma: no cover - fallback path for minimal environments
    NearestNeighbors = None


TRAIN_META_CACHE = CACHE_DIR / "train_structure_fingerprint_metadata.jsonl"
TRAIN_FP_CACHE = CACHE_DIR / "train_structure_fingerprints.npy"
TEST_META_CACHE = CACHE_DIR / "test_structure_fingerprint_metadata.jsonl"
TEST_FP_CACHE = CACHE_DIR / "test_structure_fingerprints.npy"

_FINGERPRINT_FEATURIZER = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit structure-similarity test-to-train leakage using MP-style "
            "CrystalNN/SiteStats structure fingerprints for candidate retrieval, "
            "followed by pymatgen StructureMatcher confirmation."
        )
    )
    parser.add_argument("--workers", type=int, default=1, help="Worker processes for fingerprint generation.")
    parser.add_argument("--top-k", type=int, default=200, help="Always retrieve this many nearest fingerprint candidates per test row.")
    parser.add_argument(
        "--fingerprint-distance-threshold",
        type=float,
        default=0.9,
        help="Also retrieve all train structures with fingerprint distance <= this threshold. Use a negative value to disable.",
    )
    parser.add_argument(
        "--force-rebuild-cache",
        action="store_true",
        help="Rebuild train/test fingerprint caches from symmetrized CIFs.",
    )
    parser.add_argument("--limit-train", type=int, default=None, help="Debug limit on unique train mp-ids.")
    parser.add_argument("--limit-test", type=int, default=None, help="Debug limit on test rows.")
    parser.add_argument(
        "--no-sklearn",
        action="store_true",
        help="Disable sklearn NearestNeighbors and use the numpy fallback retrieval.",
    )
    return parser.parse_args()


def fingerprint_featurizer():
    global _FINGERPRINT_FEATURIZER
    if _FINGERPRINT_FEATURIZER is None:
        from matminer.featurizers.site import CrystalNNFingerprint
        from matminer.featurizers.structure import SiteStatsFingerprint

        _FINGERPRINT_FEATURIZER = SiteStatsFingerprint(
            CrystalNNFingerprint.from_preset("ops", distance_cutoffs=None, x_diff_weight=0),
            stats=("mean", "std_dev", "minimum", "maximum"),
        )
    return _FINGERPRINT_FEATURIZER


def collect_unique_train_mp_ids(limit: int | None = None) -> list[str]:
    seen: set[str] = set()
    mp_ids: list[str] = []
    for _, row in tqdm(iter_jsonl(TRAIN_PATH), desc="Scanning train mp-ids", unit="row"):
        mp_id = row["mp_id"]
        if mp_id in seen:
            continue
        seen.add(mp_id)
        mp_ids.append(mp_id)
        if limit is not None and len(mp_ids) >= limit:
            break
    return mp_ids


def collect_test_rows(limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, row in iter_jsonl(TEST_PATH):
        rows.append(
            {
                "row_index": row_index,
                "mp_id": row["mp_id"],
                "test_cif_hash": row.get("cif_split_meta", {}).get("target_cif_hash"),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def compute_fingerprint_record(item: tuple[int | None, str]) -> tuple[dict[str, Any], list[float] | None]:
    row_index, mp_id = item
    cif_text = read_symmetrized_cif_text(mp_id)
    record: dict[str, Any] = {
        "row_index": row_index,
        "mp_id": mp_id,
        "symmetrized_cif_path": str(symmetrized_cif_path(mp_id)),
        "symmetrized_cif_found": cif_text is not None,
        "structure_parse_success": False,
        "fingerprint_success": False,
        "num_sites": None,
        "fingerprint_index": None,
        "failure_reason": None,
    }
    if cif_text is None:
        record["failure_reason"] = "missing_symmetrized_cif"
        return record, None

    structure = parse_structure(cif_text)
    if structure is None:
        record["failure_reason"] = "structure_parse_failed"
        return record, None

    record["structure_parse_success"] = True
    record["num_sites"] = len(structure)
    try:
        with warnings.catch_warnings(), suppress_fd_stderr(True):
            warnings.simplefilter("ignore")
            vector = fingerprint_featurizer().featurize(structure)
    except Exception as exc:
        record["failure_reason"] = f"fingerprint_failed:{type(exc).__name__}"
        return record, None

    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim != 1 or not np.all(np.isfinite(arr)):
        record["failure_reason"] = "fingerprint_non_finite"
        return record, None

    record["fingerprint_success"] = True
    return record, arr.tolist()


def write_metadata(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_or_load_fingerprint_cache(
    name: str,
    items: list[tuple[int | None, str]],
    meta_path: Path,
    fp_path: Path,
    workers: int,
    force: bool,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if not force and meta_path.exists() and fp_path.exists():
        metadata = read_metadata(meta_path)
        fingerprints = np.load(fp_path)
        print(f"Loaded {name} fingerprint cache: {len(metadata)} metadata rows, {len(fingerprints)} fingerprints.", flush=True)
        return metadata, fingerprints

    print(f"Building {name} structure fingerprints for {len(items)} structures with workers={workers}.", flush=True)
    metadata: list[dict[str, Any]] = []
    vectors: list[list[float]] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            iterator = executor.map(compute_fingerprint_record, items, chunksize=16)
            for record, vector in tqdm(iterator, total=len(items), desc=f"Featurizing {name}", unit="structure"):
                if vector is not None:
                    record["fingerprint_index"] = len(vectors)
                    vectors.append(vector)
                metadata.append(record)
    else:
        for item in tqdm(items, desc=f"Featurizing {name}", unit="structure"):
            record, vector = compute_fingerprint_record(item)
            if vector is not None:
                record["fingerprint_index"] = len(vectors)
                vectors.append(vector)
            metadata.append(record)

    fingerprints = np.asarray(vectors, dtype=np.float32)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fp = fp_path.with_suffix(fp_path.suffix + ".tmp")
    with tmp_fp.open("wb") as handle:
        np.save(handle, fingerprints)
    tmp_fp.replace(fp_path)
    write_metadata(meta_path, metadata)
    print(f"Built {name} fingerprint cache: {len(metadata)} metadata rows, {len(fingerprints)} fingerprints.", flush=True)
    return metadata, fingerprints


def valid_records_by_fp_index(metadata: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid = [row for row in metadata if row.get("fingerprint_index") is not None]
    valid.sort(key=lambda row: int(row["fingerprint_index"]))
    return valid


def retrieve_candidates_sklearn(
    train_fp: np.ndarray,
    test_fp: np.ndarray,
    top_k: int,
    threshold: float | None,
    workers: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if NearestNeighbors is None:
        raise RuntimeError("sklearn is not available")
    n_neighbors = min(max(1, top_k), len(train_fp))
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute", metric="euclidean", n_jobs=max(1, workers))
    nn.fit(train_fp)
    top_distances, top_indices = nn.kneighbors(test_fp, return_distance=True)

    if threshold is None:
        return [(top_indices[i], top_distances[i], top_indices[i]) for i in range(len(test_fp))]

    radius_distances, radius_indices = nn.radius_neighbors(test_fp, radius=threshold, return_distance=True, sort_results=True)
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(len(test_fp)):
        index_to_distance: dict[int, float] = {
            int(idx): float(dist) for idx, dist in zip(top_indices[i], top_distances[i], strict=False)
        }
        for idx, dist in zip(radius_indices[i], radius_distances[i], strict=False):
            index_to_distance[int(idx)] = min(index_to_distance.get(int(idx), float("inf")), float(dist))
        merged = sorted(index_to_distance.items(), key=lambda item: item[1])
        candidate_indices = np.asarray([idx for idx, _ in merged], dtype=np.int64)
        candidate_distances = np.asarray([dist for _, dist in merged], dtype=np.float32)
        results.append((candidate_indices, candidate_distances, np.asarray(top_indices[i], dtype=np.int64)))
    return results


def retrieve_candidates_numpy(
    train_fp: np.ndarray,
    test_fp: np.ndarray,
    top_k: int,
    threshold: float | None,
    chunk_size: int = 8192,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    results: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    n_train = len(train_fp)
    k = min(max(1, top_k), n_train)
    for vector in tqdm(test_fp, desc="Retrieving fingerprint candidates", unit="test"):
        selected: dict[int, float] = {}
        top_candidates: list[tuple[int, float]] = []
        for start in range(0, n_train, chunk_size):
            stop = min(start + chunk_size, n_train)
            diff = train_fp[start:stop] - vector
            distances = np.sqrt(np.einsum("ij,ij->i", diff, diff, optimize=True))
            if threshold is not None:
                within = np.flatnonzero(distances <= threshold)
                for local_idx in within:
                    idx = start + int(local_idx)
                    selected[idx] = min(selected.get(idx, float("inf")), float(distances[local_idx]))
            if len(distances) <= k:
                local_top = np.arange(len(distances))
            else:
                local_top = np.argpartition(distances, k - 1)[:k]
            top_candidates.extend((start + int(local_idx), float(distances[local_idx])) for local_idx in local_top)
        top_candidates = sorted(top_candidates, key=lambda item: item[1])[:k]
        for idx, dist in top_candidates:
            selected[idx] = min(selected.get(idx, float("inf")), dist)
        merged = sorted(selected.items(), key=lambda item: item[1])
        results.append(
            (
                np.asarray([idx for idx, _ in merged], dtype=np.int64),
                np.asarray([dist for _, dist in merged], dtype=np.float32),
                np.asarray([idx for idx, _ in top_candidates], dtype=np.int64),
            )
        )
    return results


@lru_cache(maxsize=20000)
def load_symmetrized_structure(mp_id: str):
    cif_text = read_symmetrized_cif_text(mp_id)
    if cif_text is None:
        return None
    return parse_structure(cif_text)


def count_metadata(metadata: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in metadata:
        if row.get("symmetrized_cif_found"):
            counts["symmetrized_cif_found"] += 1
        else:
            counts["symmetrized_cif_missing"] += 1
        if row.get("structure_parse_success"):
            counts["structure_parse_success"] += 1
        else:
            counts["structure_parse_failure"] += 1
        if row.get("fingerprint_success"):
            counts["fingerprint_success"] += 1
        else:
            counts["fingerprint_failure"] += 1
    return dict(counts)


def main() -> None:
    args = parse_args()
    workers = max(1, args.workers)
    threshold = args.fingerprint_distance_threshold if args.fingerprint_distance_threshold >= 0 else None

    train_mp_ids = collect_unique_train_mp_ids(limit=args.limit_train)
    test_rows = collect_test_rows(limit=args.limit_test)
    train_items = [(None, mp_id) for mp_id in train_mp_ids]
    test_items = [(row["row_index"], row["mp_id"]) for row in test_rows]

    train_meta, train_fp = build_or_load_fingerprint_cache(
        "train",
        train_items,
        TRAIN_META_CACHE,
        TRAIN_FP_CACHE,
        workers=workers,
        force=args.force_rebuild_cache,
    )
    test_meta, test_fp = build_or_load_fingerprint_cache(
        "test",
        test_items,
        TEST_META_CACHE,
        TEST_FP_CACHE,
        workers=workers,
        force=args.force_rebuild_cache,
    )

    train_valid = valid_records_by_fp_index(train_meta)
    test_valid = valid_records_by_fp_index(test_meta)
    if len(train_valid) != len(train_fp):
        raise RuntimeError("train metadata and fingerprint cache are misaligned")
    if len(test_valid) != len(test_fp):
        raise RuntimeError("test metadata and fingerprint cache are misaligned")

    if len(train_fp) == 0 or len(test_fp) == 0:
        raise RuntimeError("no valid fingerprints available for retrieval")

    print(
        f"Retrieving fingerprint candidates for {len(test_fp)} valid test structures against "
        f"{len(train_fp)} valid train structures; top_k={args.top_k}, threshold={threshold}.",
        flush=True,
    )
    if NearestNeighbors is not None and not args.no_sklearn:
        candidate_results = retrieve_candidates_sklearn(train_fp, test_fp, args.top_k, threshold, workers)
        retrieval_backend = "sklearn.NearestNeighbors(brute, euclidean)"
    else:
        candidate_results = retrieve_candidates_numpy(train_fp, test_fp, args.top_k, threshold)
        retrieval_backend = "numpy_chunked_euclidean"

    matcher = StructureMatcher(
        ltol=0.2,
        stol=0.3,
        angle_tol=5.0,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )

    test_meta_by_row = {row["row_index"]: row for row in test_meta}
    valid_result_by_row: dict[int, dict[str, Any]] = {}
    candidate_counts: list[int] = []
    nearest_distances: list[float] = []
    threshold_candidate_counts: list[int] = []
    structurematcher_exceptions = 0

    for test_record, (candidate_indices, candidate_distances, top_indices) in tqdm(
        zip(test_valid, candidate_results, strict=True),
        total=len(test_valid),
        desc="StructureMatcher confirming candidates",
        unit="test",
    ):
        row_index = int(test_record["row_index"])
        test_mp_id = test_record["mp_id"]
        candidate_counts.append(len(candidate_indices))
        nearest_distances.append(float(candidate_distances[0]) if len(candidate_distances) else float("nan"))
        if threshold is None:
            threshold_candidate_counts.append(0)
        else:
            threshold_candidate_counts.append(int(np.sum(candidate_distances <= threshold)))

        test_structure = load_symmetrized_structure(test_mp_id)
        overlap = False
        matched_train_mp_id = None
        matched_train_distance = None
        matched_train_symmetrized_cif_path = None
        nearest_train_mp_id = train_valid[int(candidate_indices[0])]["mp_id"] if len(candidate_indices) else None
        nearest_fingerprint_distance = float(candidate_distances[0]) if len(candidate_distances) else None

        if test_structure is not None:
            for fp_idx, distance in zip(candidate_indices, candidate_distances, strict=False):
                train_record = train_valid[int(fp_idx)]
                train_structure = load_symmetrized_structure(train_record["mp_id"])
                if train_structure is None:
                    continue
                try:
                    if matcher.fit(test_structure, train_structure):
                        overlap = True
                        matched_train_mp_id = train_record["mp_id"]
                        matched_train_distance = float(distance)
                        matched_train_symmetrized_cif_path = train_record["symmetrized_cif_path"]
                        break
                except Exception:
                    structurematcher_exceptions += 1
                    continue

        valid_result_by_row[row_index] = {
            "fingerprint_candidate_count": len(candidate_indices),
            "fingerprint_threshold_candidate_count": threshold_candidate_counts[-1],
            "fingerprint_top_k": min(args.top_k, len(train_fp)),
            "nearest_train_mp_id": nearest_train_mp_id,
            "nearest_fingerprint_distance": nearest_fingerprint_distance,
            "structure_similarity_overlap": overlap,
            "matched_train_mp_id": matched_train_mp_id,
            "matched_train_fingerprint_distance": matched_train_distance,
            "matched_train_symmetrized_cif_path": matched_train_symmetrized_cif_path,
        }

    details: list[dict[str, Any]] = []
    counters = Counter()
    test_row_lookup = {row["row_index"]: row for row in test_rows}
    for row in test_rows:
        row_index = row["row_index"]
        meta = test_meta_by_row.get(row_index, {})
        result = valid_result_by_row.get(row_index)
        base = {
            "test_row_index": row_index,
            "test_mp_id": row["mp_id"],
            "test_cif_hash": row.get("test_cif_hash"),
            "test_symmetrized_cif_path": str(symmetrized_cif_path(row["mp_id"])),
            "test_symmetrized_cif_found": bool(meta.get("symmetrized_cif_found", False)),
            "test_structure_parse_success": bool(meta.get("structure_parse_success", False)),
            "test_fingerprint_success": bool(meta.get("fingerprint_success", False)),
            "test_fingerprint_failure_reason": meta.get("failure_reason"),
        }
        if result is None:
            base.update(
                {
                    "fingerprint_candidate_count": 0,
                    "fingerprint_threshold_candidate_count": 0,
                    "fingerprint_top_k": min(args.top_k, len(train_fp)),
                    "nearest_train_mp_id": None,
                    "nearest_fingerprint_distance": None,
                    "structure_similarity_overlap": False,
                    "matched_train_mp_id": None,
                    "matched_train_fingerprint_distance": None,
                    "matched_train_symmetrized_cif_path": None,
                }
            )
        else:
            base.update(result)
        counters["structure_similarity_overlap"] += int(base["structure_similarity_overlap"])
        counters["test_rows_with_fingerprint_failure"] += int(not base["test_fingerprint_success"])
        details.append(base)

    candidate_counts_array = np.asarray(candidate_counts, dtype=np.float32)
    nearest_distances_array = np.asarray(nearest_distances, dtype=np.float32)
    threshold_counts_array = np.asarray(threshold_candidate_counts, dtype=np.float32)
    summary = {
        "train_path": str(TRAIN_PATH),
        "test_path": str(TEST_PATH),
        "symmetrized_cif_source": str(SYMMETRIZED_CIF_DIR / "mp-*_symmetrized.cif"),
        "test_rows": len(test_rows),
        "train_unique_mp_ids_scanned": len(train_mp_ids),
        "train_fingerprint_cache": str(TRAIN_FP_CACHE),
        "train_fingerprint_metadata": str(TRAIN_META_CACHE),
        "test_fingerprint_cache": str(TEST_FP_CACHE),
        "test_fingerprint_metadata": str(TEST_META_CACHE),
        "definitions": {
            "fingerprint_candidate_retrieval": (
                "MP-style structure fingerprints generated with matminer SiteStatsFingerprint "
                "and CrystalNNFingerprint.from_preset('ops', distance_cutoffs=None, x_diff_weight=0). "
                "Candidates are retrieved globally from the full train split by Euclidean fingerprint distance, "
                "without formula-based prefiltering."
            ),
            "structure_similarity_overlap": (
                "At least one fingerprint-retrieved train candidate is confirmed as matching the test structure "
                "by pymatgen StructureMatcher."
            ),
            "fingerprint_distance_threshold": threshold,
            "top_k_nearest_candidates": min(args.top_k, len(train_fp)),
            "retrieval_backend": retrieval_backend,
            "StructureMatcher_parameters": {
                "ltol": 0.2,
                "stol": 0.3,
                "angle_tol": 5.0,
                "primitive_cell": True,
                "scale": True,
                "attempt_supercell": False,
                "allow_subset": False,
            },
        },
        "counts": {
            **dict(counters),
            "test_rows_with_valid_fingerprint": len(test_fp),
            "train_rows_with_valid_fingerprint": len(train_fp),
            "structurematcher_exceptions": structurematcher_exceptions,
        },
        "fingerprint_cache_counts": {
            "train": count_metadata(train_meta),
            "test": count_metadata(test_meta),
        },
        "fingerprint_candidate_statistics": {
            "mean_candidate_count": round(float(np.mean(candidate_counts_array)), 6) if len(candidate_counts_array) else None,
            "median_candidate_count": round(float(np.median(candidate_counts_array)), 6) if len(candidate_counts_array) else None,
            "max_candidate_count": int(np.max(candidate_counts_array)) if len(candidate_counts_array) else None,
            "mean_threshold_candidate_count": round(float(np.mean(threshold_counts_array)), 6) if len(threshold_counts_array) else None,
            "median_nearest_fingerprint_distance": round(float(np.median(nearest_distances_array)), 6) if len(nearest_distances_array) else None,
            "min_nearest_fingerprint_distance": round(float(np.min(nearest_distances_array)), 6) if len(nearest_distances_array) else None,
        },
        "ratios_over_test": {
            "structure_similarity_overlap": round(counters["structure_similarity_overlap"] / len(test_rows), 6) if test_rows else None,
            "test_fingerprint_assignment_coverage": round(len(test_fp) / len(test_rows), 6) if test_rows else None,
        },
        "examples": {
            "structure_similarity_overlap": [row for row in details if row["structure_similarity_overlap"]][:20],
            "nearest_fingerprint_candidates": sorted(
                [row for row in details if row["nearest_fingerprint_distance"] is not None],
                key=lambda row: row["nearest_fingerprint_distance"],
            )[:20],
            "fingerprint_failures": [row for row in details if not row["test_fingerprint_success"]][:20],
        },
    }

    summary_path = SUMMARY_DIR / "structure_similarity_leakage_summary.json"
    details_path = DETAILS_DIR / "structure_similarity_leakage_details.jsonl"
    write_json(summary_path, summary)
    write_jsonl(details_path, details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"wrote: {summary_path}")
    print(f"wrote: {details_path}")


if __name__ == "__main__":
    main()
