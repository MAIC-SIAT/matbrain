#!/usr/bin/env python3
import json
import os
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from pymatgen.core import Composition, Structure


ROOT = Path(os.environ.get("MATBRAIN_ROOT", ".")).resolve()
TRAIN_PATH = ROOT / "dataset/sft_rl_data/mp_cif_design_property_train.jsonl"
TEST_PATH = ROOT / "dataset/sft_rl_data/mp_cif_design_property_test.jsonl"
OUTPUT_DIR = ROOT / "rebuttal/data_leakage"
SUMMARY_DIR = OUTPUT_DIR / "summaries"
DETAILS_DIR = OUTPUT_DIR / "details"
CACHE_DIR = OUTPUT_DIR / "cache"
CLEAN_TEST_DIR = OUTPUT_DIR / "clean_tests"
SYMMETRIZED_CIF_DIR = Path(
    os.environ.get("MATBRAIN_SYMMETRIZED_CIF_DIR", "data/external/symmetrized_cifs")
).resolve()


@dataclass
class Entry:
    split: str
    row_index: int
    mp_id: str
    cif_hash: str
    formula_raw: str | None
    formula_raw_normalized: str | None
    reduced_formula: str | None
    anonymous_formula: str | None
    spacegroup: str | None
    cif_text: str
    structure: Structure | None = None


@contextmanager
def suppress_fd_stderr(enabled: bool = True) -> Iterator[None]:
    if not enabled:
        yield
        return

    old_stderr = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(old_stderr)
        os.close(devnull)


def parse_cif_metadata(cif_text: str) -> dict[str, str | None]:
    formula_raw = None
    spacegroup = None
    for line in cif_text.splitlines():
        s = line.strip()
        if formula_raw is None and s.startswith("_chemical_formula_structural"):
            parts = s.split(None, 1)
            if len(parts) == 2:
                formula_raw = parts[1].strip().strip("'").strip('"')
        elif spacegroup is None and s.startswith("_symmetry_space_group_name_H-M"):
            parts = s.split(None, 1)
            if len(parts) == 2:
                spacegroup = parts[1].strip().strip("'").strip('"')
        if formula_raw is not None and spacegroup is not None:
            break
    return {"formula_raw": formula_raw, "spacegroup": spacegroup}


def normalize_formula_raw(formula_raw: str | None) -> str | None:
    if not formula_raw:
        return None
    return "".join(formula_raw.split())


def normalize_formula(formula_raw: str | None) -> tuple[str | None, str | None]:
    if not formula_raw:
        return None, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            comp = Composition(formula_raw)
            return comp.reduced_formula, comp.anonymized_formula
        except Exception:
            return formula_raw, None


def parse_structure(cif_text: str, suppress_stderr: bool = True) -> Structure | None:
    with warnings.catch_warnings(), suppress_fd_stderr(suppress_stderr):
        warnings.simplefilter("ignore")
        try:
            return Structure.from_str(cif_text, fmt="cif")
        except Exception:
            return None


def symmetrized_cif_path(mp_id: str) -> Path:
    return SYMMETRIZED_CIF_DIR / f"{mp_id}_symmetrized.cif"


def read_symmetrized_cif_text(mp_id: str) -> str | None:
    path = symmetrized_cif_path(mp_id)
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def entry_from_row(row: dict, row_index: int, split: str, need_structure: bool = False) -> Entry:
    cif_text = row["cif_split_meta"]["target_cif"]
    meta = parse_cif_metadata(cif_text)
    reduced_formula, anonymous_formula = normalize_formula(meta["formula_raw"])
    structure = parse_structure(cif_text) if need_structure else None

    return Entry(
        split=split,
        row_index=row_index,
        mp_id=row["mp_id"],
        cif_hash=row["cif_split_meta"]["target_cif_hash"],
        formula_raw=meta["formula_raw"],
        formula_raw_normalized=normalize_formula_raw(meta["formula_raw"]),
        reduced_formula=reduced_formula,
        anonymous_formula=anonymous_formula,
        spacegroup=meta["spacegroup"],
        cif_text=cif_text,
        structure=structure,
    )


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict]]:
    with path.open("r", encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            if not line.strip():
                continue
            yield row_index, json.loads(line)


def load_entries(path: Path, split: str, need_structure: bool = False) -> list[Entry]:
    return [
        entry_from_row(row, row_index, split=split, need_structure=need_structure)
        for row_index, row in iter_jsonl(path)
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
