#!/usr/bin/env python3
"""Convert leakage-cleaned SFT data into Mat-T1 RL prompt/ground-truth data.

This intentionally uses only the structure-design/property-prediction SFT file.
Synthesis-path planning is left out because its free-form outputs are harder to
reward robustly within the revision window.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_SYSTEM = (
    "You are Mat-T1, the executive tool-use model in MatBrain. Use the available "
    "Mat-MCP tools when they are needed. Keep tool calls necessary and "
    "non-redundant. Every intermediate tool-use step must use <think> followed by "
    "<tool_call>. Once enough evidence is collected, you must stop calling tools "
    "and produce a final <think> followed by a final <answer>. Ground the final "
    "answer in tool observations, and return the requested CIF and/or JSON in the "
    "final answer."
)

PROPERTY_KEYS = {
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "composition",
    "elements",
    "symmetry",
    "nelements",
    "nsites",
    "volume",
    "density",
    "density_atomic",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "efermi",
    "band_gap",
    "is_gap_direct",
    "is_metal",
    "is_magnetic",
    "ordering",
    "total_magnetization",
    "num_magnetic_sites",
}

DEFAULT_TOLERANCES = {
    "formation_energy_per_atom": 0.15,
    "energy_above_hull": 0.05,
    "band_gap": 0.25,
    "efermi": 0.30,
    "density": 0.20,
    "total_magnetization": 0.30,
}


def iter_sft_pairs(row: dict[str, Any]) -> list[tuple[int, str, str, str | None]]:
    conv = row.get("conversations") or []
    meta = row.get("record_meta") or {}
    kinds_by_human_index = {}
    human_indices = meta.get("source_human_indices") or []
    pair_kinds = meta.get("pair_kinds") or []
    for idx, kind in zip(human_indices, pair_kinds):
        kinds_by_human_index[int(idx)] = str(kind)

    pairs = []
    i = 0
    while i < len(conv) - 1:
        cur = conv[i]
        nxt = conv[i + 1]
        if cur.get("from") == "human" and nxt.get("from") == "gpt":
            pairs.append((i, str(cur.get("value", "")), str(nxt.get("value", "")), kinds_by_human_index.get(i)))
            i += 2
        else:
            i += 1
    return pairs


def extract_json_objects(text: str) -> list[dict[str, Any]]:
    objects: list[dict[str, Any]] = []
    text = text or ""
    for match in re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL | re.IGNORECASE):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict):
                objects.append(obj)
        except Exception:
            pass
    # Fallback: scan balanced-ish object spans. This is conservative enough for
    # the model-generated property JSON blocks in this dataset.
    for start in [m.start() for m in re.finditer(r"\{", text)]:
        depth = 0
        for pos in range(start, len(text)):
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
                if depth == 0:
                    raw = text[start : pos + 1]
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict) and obj not in objects:
                            objects.append(obj)
                    except Exception:
                        pass
                    break
    return objects


def best_property_json(text: str) -> dict[str, Any] | None:
    best = None
    best_score = 0
    for obj in extract_json_objects(text):
        score = len(PROPERTY_KEYS & set(obj.keys()))
        if isinstance(obj.get("symmetry"), dict):
            score += 2
        if score > best_score:
            best = obj
            best_score = score
    return best if best_score >= 2 else None


def extract_cif(text: str) -> str | None:
    text = text or ""
    fenced = re.findall(r"```(?:cif)?\s*(data_.*?)(?:```|$)", text, re.DOTALL | re.IGNORECASE)
    for block in fenced:
        if "_cell_length_" in block and "_atom_site" in block:
            return block.strip()
    match = re.search(r"(data_[\s\S]*?_atom_site_occupancy[\s\S]*?)(?:\n\s*\n[A-Z#*_`]|$)", text)
    if match:
        block = match.group(1).strip()
        if "_cell_length_" in block and "_atom_site" in block:
            return block
    return None


def compact_target_properties(obj: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(obj, dict):
        return {}
    out = {}
    for key in PROPERTY_KEYS:
        if key in obj:
            out[key] = obj[key]
    symmetry = out.get("symmetry")
    if isinstance(symmetry, dict):
        if "symbol" in symmetry:
            out["space_group_symbol"] = symmetry["symbol"]
        if "number" in symmetry:
            out["space_group_number"] = symmetry["number"]
        if "crystal_system" in symmetry:
            out["crystal_system"] = symmetry["crystal_system"]
    if "formula_pretty" in out:
        out["formula"] = out["formula_pretty"]
    return out


def parse_prompt_targets(text: str) -> dict[str, Any]:
    text = text or ""
    out: dict[str, Any] = {}
    patterns = {
        "formation_energy_per_atom": r"formation energy(?: per atom)?(?:\s*(?:is|of|near|around|approximately|=|:))?\s*([-+]?\d+(?:\.\d+)?)",
        "energy_above_hull": r"energy above (?:the )?hull(?:\s*(?:is|of|near|around|approximately|=|:))?\s*([-+]?\d+(?:\.\d+)?)",
        "band_gap": r"band gap(?:\s*(?:is|of|near|around|approximately|=|:))?\s*([-+]?\d+(?:\.\d+)?)",
        "efermi": r"(?:fermi energy|fermi level)(?:\s*(?:is|of|near|around|approximately|=|:))?\s*([-+]?\d+(?:\.\d+)?)",
        "density": r"density(?:\s*(?:is|of|near|around|approximately|=|:))?\s*([-+]?\d+(?:\.\d+)?)",
        "nsites": r"(?:containing|consider|needs?|with|have)\s*(\d+)\s*(?:distinct )?(?:atomic )?sites?",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            out[key] = int(value) if key == "nsites" else value

    sg_match = re.search(r"space group(?:\s*\(number\s*(\d+)\))?\s*(?:is|of|with|in|:)?\s*([A-Za-z0-9_/\-]+)", text, re.IGNORECASE)
    if sg_match:
        if sg_match.group(1):
            out["space_group_number"] = int(sg_match.group(1))
        symbol = sg_match.group(2).strip(" .,)(")
        if symbol.lower() not in {"number", "symmetry", "and"}:
            out["space_group_symbol"] = symbol

    if re.search(r"\bnon[- ]?magnetic\b", text, re.IGNORECASE):
        out["is_magnetic"] = False
        out["ordering"] = "NM"
    elif re.search(r"\bferromagnetic\b", text, re.IGNORECASE):
        out["is_magnetic"] = True
        out["ordering"] = "FM"
    elif re.search(r"\bferrimagnetic\b", text, re.IGNORECASE):
        out["is_magnetic"] = True
        out["ordering"] = "FiM"

    if re.search(r"\bmetallic\b|\bmetal\b", text, re.IGNORECASE):
        out["is_metal"] = True
    elif re.search(r"\bnon[- ]?metallic\b|\binsulating\b|\bsemiconduct", text, re.IGNORECASE):
        out["is_metal"] = False

    formula_match = re.search(
        r"(?:composition|formula|stoichiometry|compound|material)\s+(?:of\s+|is\s+|with\s+)?([A-Z][A-Za-z0-9₀-₉₂₃₄₅₆₇₈₉{}()._\-]+)",
        text,
    )
    if formula_match:
        out["formula"] = formula_match.group(1).strip(" .,:;)")
    return out


def classify_pair(human: str, gpt: str, meta_kind: str | None) -> str | None:
    if meta_kind in {"structure_design", "property_prediction"}:
        return meta_kind
    human_has_cif = "data_" in human and "_atom_site" in human
    gpt_has_cif = extract_cif(gpt) is not None
    gpt_has_json = best_property_json(gpt) is not None
    if human_has_cif and gpt_has_json:
        return "property_prediction"
    if gpt_has_cif:
        return "structure_design"
    return None


def make_prompt(task: str, human: str, input_cif: str | None = None) -> list[dict[str, str]]:
    content = human.strip()
    if task == "property_prediction" and input_cif and "data_" not in human:
        content += f"\n\nCIF:\n```cif\n{input_cif.strip()}\n```"
    return [
        {"role": "system", "content": DEFAULT_SYSTEM},
        {"role": "user", "content": content},
    ]


def convert_pair(
    source_path: str,
    row_index: int,
    pair_index: int,
    row: dict[str, Any],
    human: str,
    gpt: str,
    task: str,
    fallback_input_cif: str | None = None,
) -> dict[str, Any] | None:
    prop_json = best_property_json(gpt)
    target_properties = compact_target_properties(prop_json)
    prompt_targets = parse_prompt_targets(human)
    for key, value in prompt_targets.items():
        target_properties.setdefault(key, value)
    reference_cif = extract_cif(gpt) if task == "structure_design" else None
    input_cif = (extract_cif(human) or fallback_input_cif) if task == "property_prediction" else None

    if task == "property_prediction" and not target_properties:
        return None
    if task == "structure_design" and not reference_cif:
        return None

    if task == "structure_design":
        gt: dict[str, Any] = {
            "reference_cif": reference_cif,
        }
    else:
        gt = target_properties

    reward_spec: dict[str, Any] = {
        "target_properties": target_properties,
        "tolerances": dict(DEFAULT_TOLERANCES),
    }

    meta = row.get("record_meta") or {}
    return {
        "data_source": "structure_design" if task == "structure_design" else "property_prediction",
        "prompt": make_prompt(task, human, input_cif=input_cif),
        "ground_truth": gt,
        "extra_info": {
            "source_path": source_path,
            "source_row_index": row_index,
            "source_pair_index": pair_index,
            "mp_id": row.get("mp_id"),
            "material_formula_key": meta.get("material_formula_key"),
            "tool_profile": "structure_property_tools",
            "expected_output": "cif" if task == "structure_design" else "property_json",
            "reward_spec": reward_spec,
            "input_cif": input_cif if task == "property_prediction" else None,
        },
    }


def convert_file(path: Path, output: Path, max_rows: int | None, seed: int, max_per_task: int | None) -> dict[str, int]:
    rng = random.Random(seed)
    counts = {"rows": 0, "pairs": 0, "written": 0, "structure_design": 0, "property_prediction": 0, "skipped": 0}
    per_task = {"structure_design": 0, "property_prediction": 0}
    output.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r", encoding="utf-8") as src, output.open("w", encoding="utf-8") as dst:
        for row_index, line in enumerate(src):
            if max_rows is not None and counts["rows"] >= max_rows:
                break
            if max_per_task is not None and all(per_task[task] >= max_per_task for task in per_task):
                break
            if not line.strip():
                continue
            counts["rows"] += 1
            try:
                row = json.loads(line)
            except Exception:
                counts["skipped"] += 1
                continue
            last_cif: str | None = None
            for pair_index, human, gpt, meta_kind in iter_sft_pairs(row):
                counts["pairs"] += 1
                task = classify_pair(human, gpt, meta_kind)
                if task not in {"structure_design", "property_prediction"}:
                    counts["skipped"] += 1
                    continue
                if max_per_task is not None and per_task[task] >= max_per_task:
                    continue
                # Deterministic mild shuffling pressure for very large files: keep
                # all rows until cap, but do not bias toward only record-local first
                # pairs when max_per_task is set.
                if max_per_task is not None and rng.random() < 0.0:
                    continue
                item = convert_pair(str(path), row_index, pair_index, row, human, gpt, task, fallback_input_cif=last_cif)
                current_cif = extract_cif(gpt)
                if current_cif:
                    last_cif = current_cif
                if item is None:
                    counts["skipped"] += 1
                    continue
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                counts["written"] += 1
                counts[task] += 1
                per_task[task] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, default=Path("dataset/sft/mp_cif_design_property_train.jsonl"))
    parser.add_argument("--eval-input", type=Path, default=Path("dataset/sft/mp_cif_design_property_eval.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset/rl/t1_mcp"))
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-eval-rows", type=int, default=None)
    parser.add_argument("--max-train-per-task", type=int, default=30000)
    parser.add_argument("--max-eval-per-task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260605)
    args = parser.parse_args()

    train_out = args.output_dir / "t1_mcp_train.jsonl"
    eval_out = args.output_dir / "t1_mcp_eval.jsonl"
    train_counts = convert_file(args.train_input, train_out, args.max_train_rows, args.seed, args.max_train_per_task)
    eval_counts = convert_file(args.eval_input, eval_out, args.max_eval_rows, args.seed, args.max_eval_per_task)
    print(json.dumps({"train_output": str(train_out), "train": train_counts}, ensure_ascii=False))
    print(json.dumps({"eval_output": str(eval_out), "eval": eval_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
