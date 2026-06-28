import json
from typing import Any, Dict, List

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Structure

from core import llm_tool


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _load_structure(cif_string: str) -> Structure:
    return Structure.from_str(cif_string, fmt="cif")


def _simulate_pattern(
    cif_string: str,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    intensity_threshold: float = 0.1,
) -> Dict[str, Any]:
    structure = _load_structure(cif_string)
    calculator = XRDCalculator(wavelength=wavelength)
    pattern = calculator.get_pattern(structure, two_theta_range=(float(two_theta_min), float(two_theta_max)))
    peaks = []
    for x, y, hkls, d_hkl in zip(pattern.x, pattern.y, pattern.hkls, pattern.d_hkls):
        if float(y) < float(intensity_threshold):
            continue
        peaks.append({
            "two_theta": float(x),
            "intensity": float(y),
            "d_hkl": float(d_hkl),
            "hkls": hkls,
        })
    peaks.sort(key=lambda item: item["two_theta"])
    return {
        "formula": structure.composition.reduced_formula,
        "wavelength": wavelength,
        "two_theta_range": [float(two_theta_min), float(two_theta_max)],
        "peaks": peaks,
    }


def _pattern_to_grid(peaks: List[Dict[str, Any]], two_theta_min: float, two_theta_max: float, step: float, sigma: float) -> np.ndarray:
    grid = np.arange(float(two_theta_min), float(two_theta_max) + float(step), float(step))
    signal = np.zeros_like(grid)
    for peak in peaks:
        center = float(peak["two_theta"])
        intensity = float(peak["intensity"])
        signal += intensity * np.exp(-0.5 * ((grid - center) / max(float(sigma), 1e-6)) ** 2)
    norm = np.linalg.norm(signal)
    if norm > 0:
        signal = signal / norm
    return signal


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@llm_tool(
    name="simulate_xrd_pattern_characterization",
    description="Simulate powder XRD peaks from a CIF using pymatgen XRDCalculator.",
)
async def simulate_xrd_pattern_characterization(
    cif_string: str,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    intensity_threshold: float = 0.1,
) -> str:
    pattern = _simulate_pattern(cif_string, wavelength, two_theta_min, two_theta_max, intensity_threshold)
    payload = {
        "ok": True,
        **pattern,
        "num_peaks": len(pattern["peaks"]),
    }
    return _json_text(payload)


@llm_tool(
    name="extract_xrd_peak_table_characterization",
    description="Extract a compact top-k XRD peak table sorted by intensity for phase-evidence summaries.",
)
async def extract_xrd_peak_table_characterization(
    cif_string: str,
    top_k: int = 12,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
) -> str:
    pattern = _simulate_pattern(cif_string, wavelength, two_theta_min, two_theta_max, 0.0)
    peaks = sorted(pattern["peaks"], key=lambda item: item["intensity"], reverse=True)[: max(1, int(top_k))]
    payload = {
        "ok": True,
        "formula": pattern["formula"],
        "wavelength": wavelength,
        "top_peaks": peaks,
    }
    return _json_text(payload)


@llm_tool(
    name="compare_xrd_patterns_characterization",
    description="Compare two CIF-derived XRD patterns using Gaussian-broadened cosine similarity and peak-position overlap.",
)
async def compare_xrd_patterns_characterization(
    cif_string_a: str,
    cif_string_b: str,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    grid_step: float = 0.02,
    peak_sigma: float = 0.08,
    match_tolerance: float = 0.20,
) -> str:
    a = _simulate_pattern(cif_string_a, wavelength, two_theta_min, two_theta_max, 0.5)
    b = _simulate_pattern(cif_string_b, wavelength, two_theta_min, two_theta_max, 0.5)
    vec_a = _pattern_to_grid(a["peaks"], two_theta_min, two_theta_max, grid_step, peak_sigma)
    vec_b = _pattern_to_grid(b["peaks"], two_theta_min, two_theta_max, grid_step, peak_sigma)
    sim = _cosine(vec_a, vec_b)
    a_positions = [float(p["two_theta"]) for p in a["peaks"]]
    b_positions = [float(p["two_theta"]) for p in b["peaks"]]
    matched = 0
    for pos in a_positions:
        if any(abs(pos - other) <= float(match_tolerance) for other in b_positions):
            matched += 1
    overlap = matched / max(len(a_positions), 1)
    payload = {
        "ok": True,
        "formula_a": a["formula"],
        "formula_b": b["formula"],
        "cosine_similarity": sim,
        "peak_position_overlap": float(overlap),
        "matched_peak_count": matched,
        "num_peaks_a": len(a_positions),
        "num_peaks_b": len(b_positions),
        "reward_components": {
            "xrd_similarity": sim,
            "xrd_peak_overlap": float(overlap),
        },
    }
    return _json_text(payload)


@llm_tool(
    name="match_xrd_to_candidate_structures_characterization",
    description="Rank candidate CIF structures by similarity to a reference CIF-derived XRD pattern.",
)
async def match_xrd_to_candidate_structures_characterization(
    reference_cif: str,
    candidate_cifs: List[str],
    candidate_labels: List[str] = None,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    grid_step: float = 0.02,
    peak_sigma: float = 0.08,
    top_k: int = 10,
) -> str:
    reference = _simulate_pattern(reference_cif, wavelength, two_theta_min, two_theta_max, 0.5)
    ref_vec = _pattern_to_grid(reference["peaks"], two_theta_min, two_theta_max, grid_step, peak_sigma)
    labels = candidate_labels or [f"candidate_{idx}" for idx in range(len(candidate_cifs))]
    rows = []
    for idx, cif in enumerate(candidate_cifs):
        pattern = _simulate_pattern(cif, wavelength, two_theta_min, two_theta_max, 0.5)
        vec = _pattern_to_grid(pattern["peaks"], two_theta_min, two_theta_max, grid_step, peak_sigma)
        rows.append({
            "label": labels[idx] if idx < len(labels) else f"candidate_{idx}",
            "formula": pattern["formula"],
            "cosine_similarity": _cosine(ref_vec, vec),
            "num_peaks": len(pattern["peaks"]),
        })
    rows.sort(key=lambda item: item["cosine_similarity"], reverse=True)
    payload = {
        "ok": True,
        "reference_formula": reference["formula"],
        "ranked_candidates": rows[: max(1, int(top_k))],
        "reward_components": {
            "best_xrd_match": rows[0]["cosine_similarity"] if rows else 0.0,
        },
    }
    return _json_text(payload)


@llm_tool(
    name="score_phase_purity_from_xrd_characterization",
    description="Score whether target peaks explain an observed peak list and flag unexplained impurity peaks.",
)
async def score_phase_purity_from_xrd_characterization(
    target_cif: str,
    observed_two_theta: List[float],
    observed_intensity: List[float] = None,
    tolerance: float = 0.25,
    wavelength: str = "CuKa",
) -> str:
    target = _simulate_pattern(target_cif, wavelength, min(observed_two_theta or [5.0]) - 2.0, max(observed_two_theta or [90.0]) + 2.0, 1.0)
    obs_int = observed_intensity or [1.0] * len(observed_two_theta)
    target_positions = [float(p["two_theta"]) for p in target["peaks"]]
    explained = []
    unexplained = []
    for pos, inten in zip(observed_two_theta, obs_int):
        nearest = min((abs(float(pos) - t), t) for t in target_positions) if target_positions else (999.0, None)
        row = {"two_theta": float(pos), "intensity": float(inten), "nearest_target_peak": nearest[1], "delta": float(nearest[0])}
        if nearest[0] <= float(tolerance):
            explained.append(row)
        else:
            unexplained.append(row)
    weighted_total = sum(float(x) for x in obs_int) or 1.0
    weighted_explained = sum(float(row["intensity"]) for row in explained)
    purity_score = float(weighted_explained / weighted_total)
    payload = {
        "ok": True,
        "target_formula": target["formula"],
        "purity_score": purity_score,
        "explained_peaks": explained,
        "unexplained_peaks": unexplained,
        "reward_components": {
            "xrd_phase_purity": purity_score,
            "impurity_penalty": float(1.0 - purity_score),
        },
    }
    return _json_text(payload)
