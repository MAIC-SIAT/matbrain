import json
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from matminer.featurizers.composition import ElementProperty, Stoichiometry, ValenceOrbital
from matminer.featurizers.structure import DensityFeatures, GlobalSymmetryFeatures, RadialDistributionFunction
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Structure

from core import llm_tool


def _json_block(payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


def _finite_float(value: Any) -> Any:
    try:
        val = float(value)
        if np.isfinite(val):
            return val
        return None
    except Exception:
        return value


def _load_structure(cif_string: str) -> Structure:
    return Structure.from_str(cif_string, fmt="cif")


@llm_tool(
    name="featurize_composition_matminer",
    description="Compute deterministic matminer composition descriptors for a formula.",
)
async def featurize_composition_matminer(formula: str, preset: str = "magpie") -> str:
    comp = Composition(formula)
    featurizers = [
        Stoichiometry(),
        ElementProperty.from_preset(preset),
        ValenceOrbital(props=["avg"]),
    ]
    features: Dict[str, Any] = {}
    for featurizer in featurizers:
        labels = featurizer.feature_labels()
        values = featurizer.featurize(comp)
        for label, value in zip(labels, values):
            features[label] = _finite_float(value)
    payload = {
        "formula": comp.reduced_formula,
        "preset": preset,
        "num_features": len(features),
        "features": features,
    }
    return "# Matminer Composition Features\n\n" + _json_block(payload)


@llm_tool(
    name="featurize_structure_matminer",
    description="Compute deterministic matminer structure descriptors from a CIF.",
)
async def featurize_structure_matminer(
    cif_string: str,
    include_rdf: bool = False,
    rdf_cutoff: float = 10.0,
    rdf_bin_size: float = 0.2,
) -> str:
    structure = _load_structure(cif_string)
    featurizers = [DensityFeatures(), GlobalSymmetryFeatures()]
    if include_rdf:
        featurizers.append(RadialDistributionFunction(cutoff=float(rdf_cutoff), bin_size=float(rdf_bin_size)))
    features: Dict[str, Any] = {}
    for featurizer in featurizers:
        labels = featurizer.feature_labels()
        values = featurizer.featurize(structure)
        for label, value in zip(labels, values):
            if isinstance(value, (list, tuple, np.ndarray)):
                features[label] = [_finite_float(v) for v in list(value)[:256]]
            else:
                features[label] = _finite_float(value)
    payload = {
        "formula": structure.composition.reduced_formula,
        "num_sites": len(structure),
        "num_features": len(features),
        "features": features,
    }
    return "# Matminer Structure Features\n\n" + _json_block(payload)


@llm_tool(
    name="compare_structure_features_matminer",
    description="Compare two CIF structures using composition, density, symmetry, and StructureMatcher features.",
)
async def compare_structure_features_matminer(
    cif_string_a: str,
    cif_string_b: str,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
) -> str:
    a = _load_structure(cif_string_a)
    b = _load_structure(cif_string_b)
    matcher = StructureMatcher(ltol=float(ltol), stol=float(stol), angle_tol=float(angle_tol))
    comp_a = a.composition.fractional_composition
    comp_b = b.composition.fractional_composition
    elements = sorted({el.symbol for el in comp_a.elements} | {el.symbol for el in comp_b.elements})
    comp_l1 = sum(abs(float(comp_a.get_atomic_fraction(el)) - float(comp_b.get_atomic_fraction(el))) for el in elements)
    payload = {
        "formula_a": a.composition.reduced_formula,
        "formula_b": b.composition.reduced_formula,
        "num_sites_a": len(a),
        "num_sites_b": len(b),
        "density_a": float(a.density),
        "density_b": float(b.density),
        "density_delta": float(abs(a.density - b.density)),
        "volume_per_atom_a": float(a.volume / max(len(a), 1)),
        "volume_per_atom_b": float(b.volume / max(len(b), 1)),
        "composition_l1_distance": float(comp_l1),
        "structure_match": bool(matcher.fit(a, b)),
        "anonymous_structure_match": bool(matcher.fit_anonymous(a, b)),
    }
    return "# Matminer Structure Feature Comparison\n\n" + _json_block(payload)


@llm_tool(
    name="rank_candidates_by_descriptor_distance_matminer",
    description="Rank candidate formulas by matminer descriptor distance to a target formula.",
)
async def rank_candidates_by_descriptor_distance_matminer(
    target_formula: str,
    candidate_formulas: List[str],
    preset: str = "magpie",
    top_k: int = 10,
) -> str:
    target = Composition(target_formula)
    featurizer = ElementProperty.from_preset(preset)
    labels = featurizer.feature_labels()
    target_vec = np.array([_finite_float(x) or 0.0 for x in featurizer.featurize(target)], dtype=float)
    rows = []
    for formula in candidate_formulas:
        comp = Composition(formula)
        vec = np.array([_finite_float(x) or 0.0 for x in featurizer.featurize(comp)], dtype=float)
        dist = float(np.linalg.norm(target_vec - vec))
        rows.append({"formula": comp.reduced_formula, "descriptor_l2_distance": dist})
    rows = sorted(rows, key=lambda item: item["descriptor_l2_distance"])[: max(1, int(top_k))]
    payload = {
        "target_formula": target.reduced_formula,
        "preset": preset,
        "feature_count": len(labels),
        "ranked_candidates": rows,
    }
    return "# Matminer Descriptor-Distance Ranking\n\n" + _json_block(payload)
