import json
from typing import Any, Dict, List

import numpy as np
from pymatgen.core import Element, Lattice, Structure

from core import llm_tool


def _json_text(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def _load_structure(cif_string: str) -> Structure:
    return Structure.from_str(cif_string, fmt="cif")


def _to_cif(structure: Structure) -> str:
    return structure.to(fmt="cif")


def _site_species(site) -> str:
    return site.specie.symbol if hasattr(site, "specie") else str(site.species_string)


def _min_distance_to_sites(structure: Structure, frac_coords: List[float]) -> float:
    if len(structure) == 0:
        return 999.0
    distances = [float(structure.lattice.get_distance_and_image(frac_coords, site.frac_coords)[0]) for site in structure]
    return min(distances) if distances else 999.0


@llm_tool(
    name="generate_vacancy_defects",
    description="Generate vacancy-defect CIF candidates by removing selected sites from a structure.",
)
async def generate_vacancy_defects(
    cif_string: str,
    species_filter: str = "",
    max_structures: int = 16,
) -> str:
    structure = _load_structure(cif_string)
    rows = []
    for idx, site in enumerate(structure):
        symbol = _site_species(site)
        if species_filter and symbol != species_filter:
            continue
        defect = structure.copy()
        defect.remove_sites([idx])
        rows.append({
            "removed_site_index": idx,
            "removed_species": symbol,
            "removed_frac_coords": [float(x) for x in site.frac_coords],
            "formula": defect.composition.reduced_formula,
            "num_sites": len(defect),
            "cif": _to_cif(defect),
        })
        if len(rows) >= max(1, int(max_structures)):
            break
    return _json_text({
        "ok": True,
        "input_formula": structure.composition.reduced_formula,
        "species_filter": species_filter,
        "candidate_count": len(rows),
        "candidates": rows,
    })


@llm_tool(
    name="generate_substitutional_dopants",
    description="Generate substitutional dopant CIF candidates by replacing host sites with a dopant element.",
)
async def generate_substitutional_dopants(
    cif_string: str,
    dopant_element: str,
    host_species: str = "",
    max_structures: int = 16,
) -> str:
    structure = _load_structure(cif_string)
    Element(dopant_element)
    rows = []
    for idx, site in enumerate(structure):
        symbol = _site_species(site)
        if host_species and symbol != host_species:
            continue
        doped = structure.copy()
        doped.replace(idx, dopant_element)
        rows.append({
            "substituted_site_index": idx,
            "host_species": symbol,
            "dopant_element": dopant_element,
            "site_frac_coords": [float(x) for x in site.frac_coords],
            "formula": doped.composition.reduced_formula,
            "num_sites": len(doped),
            "cif": _to_cif(doped),
        })
        if len(rows) >= max(1, int(max_structures)):
            break
    return _json_text({
        "ok": True,
        "input_formula": structure.composition.reduced_formula,
        "dopant_element": dopant_element,
        "host_species": host_species,
        "candidate_count": len(rows),
        "candidates": rows,
    })


@llm_tool(
    name="generate_interstitial_defects",
    description="Generate interstitial-defect CIF candidates by inserting an element at candidate fractional coordinates.",
)
async def generate_interstitial_defects(
    cif_string: str,
    interstitial_element: str,
    candidate_frac_coords: List[List[float]] = None,
    min_distance_angstrom: float = 1.0,
    max_structures: int = 16,
) -> str:
    structure = _load_structure(cif_string)
    Element(interstitial_element)
    candidate_frac_coords = candidate_frac_coords or [
        [0.5, 0.5, 0.5],
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
        [0.5, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5],
    ]
    rows = []
    for coords in candidate_frac_coords:
        frac = [float(x) % 1.0 for x in coords]
        min_dist = _min_distance_to_sites(structure, frac)
        if min_dist < float(min_distance_angstrom):
            continue
        defect = structure.copy()
        defect.append(interstitial_element, frac, coords_are_cartesian=False)
        rows.append({
            "interstitial_element": interstitial_element,
            "frac_coords": frac,
            "nearest_neighbor_distance_A": min_dist,
            "formula": defect.composition.reduced_formula,
            "num_sites": len(defect),
            "cif": _to_cif(defect),
        })
        if len(rows) >= max(1, int(max_structures)):
            break
    return _json_text({
        "ok": True,
        "input_formula": structure.composition.reduced_formula,
        "interstitial_element": interstitial_element,
        "min_distance_angstrom": float(min_distance_angstrom),
        "candidate_count": len(rows),
        "candidates": rows,
    })


@llm_tool(
    name="calculate_lattice_mismatch",
    description="Calculate in-plane lattice mismatch between two structures using their a/b lattice constants.",
)
async def calculate_lattice_mismatch(
    substrate_cif: str,
    film_cif: str,
    substrate_multipliers: List[int] = None,
    film_multipliers: List[int] = None,
) -> str:
    substrate = _load_structure(substrate_cif)
    film = _load_structure(film_cif)
    substrate_multipliers = substrate_multipliers or [1, 1]
    film_multipliers = film_multipliers or [1, 1]
    sub_a = float(substrate.lattice.a) * int(substrate_multipliers[0])
    sub_b = float(substrate.lattice.b) * int(substrate_multipliers[1])
    film_a = float(film.lattice.a) * int(film_multipliers[0])
    film_b = float(film.lattice.b) * int(film_multipliers[1])
    mismatch_a = (film_a - sub_a) / max(sub_a, 1e-8)
    mismatch_b = (film_b - sub_b) / max(sub_b, 1e-8)
    score = float(max(0.0, 1.0 - (abs(mismatch_a) + abs(mismatch_b)) / 0.20))
    return _json_text({
        "ok": True,
        "substrate_formula": substrate.composition.reduced_formula,
        "film_formula": film.composition.reduced_formula,
        "substrate_inplane_lengths_A": [sub_a, sub_b],
        "film_inplane_lengths_A": [film_a, film_b],
        "mismatch_a": float(mismatch_a),
        "mismatch_b": float(mismatch_b),
        "mean_abs_mismatch": float((abs(mismatch_a) + abs(mismatch_b)) / 2.0),
        "reward_components": {"lattice_match_score": score},
    })


@llm_tool(
    name="generate_heterostructure_interface",
    description="Build a simple stacked heterostructure by placing a film above a substrate along c.",
)
async def generate_heterostructure_interface(
    substrate_cif: str,
    film_cif: str,
    gap_angstrom: float = 2.0,
    vacuum_angstrom: float = 12.0,
) -> str:
    substrate = _load_structure(substrate_cif)
    film = _load_structure(film_cif)
    a = max(float(substrate.lattice.a), float(film.lattice.a))
    b = max(float(substrate.lattice.b), float(film.lattice.b))
    c_sub = float(substrate.lattice.c)
    c_film = float(film.lattice.c)
    c_total = c_sub + float(gap_angstrom) + c_film + float(vacuum_angstrom)
    lattice = Lattice.from_parameters(a, b, c_total, 90, 90, 90)
    species = []
    cart_coords = []
    for site in substrate:
        frac = site.frac_coords
        cart_coords.append([frac[0] * a, frac[1] * b, frac[2] * c_sub])
        species.append(site.species_string)
    z_offset = c_sub + float(gap_angstrom)
    for site in film:
        frac = site.frac_coords
        cart_coords.append([frac[0] * a, frac[1] * b, z_offset + frac[2] * c_film])
        species.append(site.species_string)
    interface = Structure(lattice, species, cart_coords, coords_are_cartesian=True)
    mismatch = json.loads(await calculate_lattice_mismatch(substrate_cif, film_cif))
    return _json_text({
        "ok": True,
        "substrate_formula": substrate.composition.reduced_formula,
        "film_formula": film.composition.reduced_formula,
        "interface_formula": interface.composition.reduced_formula,
        "num_sites": len(interface),
        "gap_angstrom": float(gap_angstrom),
        "vacuum_angstrom": float(vacuum_angstrom),
        "lattice_mismatch": mismatch,
        "cif": _to_cif(interface),
    })


@llm_tool(
    name="generate_diffusion_path",
    description="Generate interpolated migrating-ion positions and optional CIF images for a simple diffusion path.",
)
async def generate_diffusion_path(
    cif_string: str,
    migrating_element: str,
    start_frac_coords: List[float],
    end_frac_coords: List[float],
    n_images: int = 5,
    include_cifs: bool = True,
) -> str:
    structure = _load_structure(cif_string)
    Element(migrating_element)
    n_images = max(2, int(n_images))
    start = np.array(start_frac_coords, dtype=float)
    end = np.array(end_frac_coords, dtype=float)
    rows = []
    for idx in range(n_images):
        t = idx / max(n_images - 1, 1)
        frac = ((1 - t) * start + t * end) % 1.0
        row = {
            "image_index": idx,
            "t": float(t),
            "frac_coords": [float(x) for x in frac],
            "nearest_neighbor_distance_A": _min_distance_to_sites(structure, [float(x) for x in frac]),
        }
        if include_cifs:
            image = structure.copy()
            image.append(migrating_element, [float(x) for x in frac], coords_are_cartesian=False)
            row["cif"] = _to_cif(image)
        rows.append(row)
    return _json_text({
        "ok": True,
        "base_formula": structure.composition.reduced_formula,
        "migrating_element": migrating_element,
        "n_images": n_images,
        "path": rows,
        "reward_components": {
            "min_path_clearance_A": min(row["nearest_neighbor_distance_A"] for row in rows) if rows else 0.0,
        },
    })
