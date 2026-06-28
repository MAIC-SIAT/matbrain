"""MatGL-backed tools for structure relaxation and property prediction."""

import asyncio
import gc
import math
import json
import os
import threading
import warnings
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import matcalc as mtc
import matgl
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from matgl.ext.ase import Relaxer, MolecularDynamics, PESCalculator
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from core import llm_tool
from core.utils import (
    load_structure_from_cif_string,
    format_basic_structure_info,
    format_optimization_status,
    validate_numeric_parameter)
from config import get_model_ref, load_model, matgl_config, resolve_model_ref

# To suppress warnings for clearer output
warnings.simplefilter("ignore")

_GPA_PER_EV_A3 = 160.2176621
_QET_VARIANT_TO_MODEL_KEY = {
    "PBE": "QET_PES_PBE",
    "R2SCAN": "QET_PES_R2SCAN",
}

# 官方 chgnet 模型缓存 + GPU。CHGNet 是纯 PyTorch(不依赖 DGL)，可直接上 GPU：
# 实测 cuda:1 比 cpu 快 ~10x，结果一致(Δ≈2e-7)。设备由 CHGNET_DEVICE 控制
# (默认 cuda:1，避开 vLLM 占用的 2/3/4/5)；无 GPU 时自动回退 cpu。只加载一次。
_CHGNET = None
_CHGNET_LOCK = threading.Lock()
_LOCAL_CHGNET_MODEL_KEY = "CHGNET_MPTRJ"
_LOCAL_CHGNET_MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "CHGNet-MPtrj-2024.2.13-11M-PES"
_CUDA_TRIM_AFTER_TOOL = os.getenv("MATGL_CUDA_TRIM_AFTER_TOOL", "1").strip().lower() not in {"0", "false", "no"}


def _normalize_device_env(raw: str | None, default: str) -> str:
    value = (raw or default).strip()
    if " - " in value:
        value = value.split(" - ", 1)[0].strip()
    return value or default


def _get_chgnet():
    global _CHGNET
    if _CHGNET is None:
        with _CHGNET_LOCK:
            if _CHGNET is None:
                dev = _normalize_device_env(os.getenv("CHGNET_DEVICE"), "cuda:1")
                if dev.startswith("cuda") and not torch.cuda.is_available():
                    dev = "cpu"
                if not _LOCAL_CHGNET_MODEL_DIR.exists():
                    raise FileNotFoundError(
                        f"Offline CHGNet model directory not found: {_LOCAL_CHGNET_MODEL_DIR}"
                    )
                potential = load_model(_LOCAL_CHGNET_MODEL_KEY)
                core_model = getattr(potential, "model", potential)
                if hasattr(core_model, "to"):
                    core_model = core_model.to(dev)
                if hasattr(core_model, "eval"):
                    core_model.eval()
                if hasattr(potential, "model"):
                    potential.model = core_model
                _CHGNET = potential
    return _CHGNET


def _tensor_to_float(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def _inference_mode():
    if hasattr(torch, "inference_mode"):
        return torch.inference_mode()
    return torch.no_grad()


def _detach_ase_calculator(atoms) -> None:
    if atoms is not None and getattr(atoms, "calc", None) is not None:
        atoms.calc = None


def _trim_cuda_cache() -> None:
    if _CUDA_TRIM_AFTER_TOOL and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _finalize_tool_cleanup(*atoms_objects) -> None:
    for atoms in atoms_objects:
        _detach_ase_calculator(atoms)
    gc.collect()
    _trim_cuda_cache()


async def _run_blocking(func, /, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)


def _normalize_state_attr(state_attr: Optional[list[int] | list[float]]) -> Optional[torch.Tensor]:
    if state_attr is None:
        return None
    if not isinstance(state_attr, list) or not state_attr:
        raise ValueError("state_attr must be a non-empty list when provided")
    dtype = torch.long if all(float(x).is_integer() for x in state_attr) else torch.float32
    return torch.tensor(state_attr, dtype=dtype)


def _model_device(model: Any) -> torch.device:
    for candidate in (getattr(model, "model", None), model):
        if candidate is None:
            continue
        try:
            return next(candidate.parameters()).device
        except (AttributeError, StopIteration, TypeError):
            continue
    return matgl_config.device


def _prepare_state_attr_tensor(state_attr: Any, device: torch.device) -> torch.Tensor:
    if isinstance(state_attr, torch.Tensor):
        tensor = state_attr
    elif state_attr is None:
        tensor = torch.tensor([0.0], dtype=matgl.float_th)
    else:
        dtype = torch.long if all(float(x).is_integer() for x in state_attr) else matgl.float_th
        tensor = torch.tensor(state_attr, dtype=dtype)
    return tensor.to(device)


def _predict_structure_device_aware(
    model: Any,
    structure: Structure,
    state_attr: Optional[torch.Tensor] = None,
) -> Any:
    if not hasattr(model, "predict_structure"):
        raise ValueError("Model does not expose predict_structure()")

    device = _model_device(model)
    if device.type == "cpu":
        with _inference_mode():
            return (
                model.predict_structure(structure, state_attr=state_attr)
                if state_attr is not None
                else model.predict_structure(structure)
            )

    core_model = getattr(model, "model", model)
    element_types = getattr(core_model, "element_types", None)
    cutoff = getattr(core_model, "cutoff", None)
    if element_types is None or cutoff is None:
        with _inference_mode():
            return (
                model.predict_structure(structure, state_attr=state_attr)
                if state_attr is not None
                else model.predict_structure(structure)
            )

    try:
        from matgl.ext.pymatgen import Structure2Graph
    except ImportError:
        from matgl.ext._pymatgen import Structure2Graph

    graph_converter = Structure2Graph(element_types=element_types, cutoff=cutoff)
    g = lat = state_tensor = state_attr_default = None
    try:
        g, lat, state_attr_default = graph_converter.get_graph(structure)
        lat = lat.to(device)
        g = g.to(device)
        g.pbc_offshift = torch.matmul(g.pbc_offset, lat[0])
        g.pos = g.frac_coords @ lat[0]
        state_tensor = _prepare_state_attr_tensor(
            state_attr if state_attr is not None else state_attr_default,
            device,
        )
        with _inference_mode():
            return model(g=g, state_attr=state_tensor).detach()
    finally:
        del state_tensor
        del g
        del lat
        del state_attr_default
        del graph_converter


def _calc_force_metrics(forces) -> dict[str, float]:
    force_tensor = torch.as_tensor(forces, dtype=torch.float32)
    magnitudes = torch.linalg.norm(force_tensor, dim=1)
    return {
        "max_force_eV_per_A": float(torch.max(magnitudes).item()),
        "rms_force_eV_per_A": float(torch.sqrt(torch.mean(magnitudes.square())).item()),
    }


def _resolve_qet_model_ref(variant: str) -> tuple[str, str]:
    normalized = variant.upper()
    if normalized not in _QET_VARIANT_TO_MODEL_KEY:
        raise ValueError(f"variant must be one of {sorted(_QET_VARIANT_TO_MODEL_KEY)}, got {variant}")
    model_key = _QET_VARIANT_TO_MODEL_KEY[normalized]
    return normalized, get_model_ref(model_key)


def _calculate_pes_observables_for_structure(structure: Structure, model_ref: str) -> dict[str, Any]:
    potential = load_model(model_ref)
    atoms = AseAtomsAdaptor().get_atoms(structure)
    try:
        atoms.calc = PESCalculator(potential)
        total_energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        payload = {
            "model_ref": resolve_model_ref(model_ref),
            "total_energy_eV": total_energy,
            "energy_per_atom_eV": total_energy / len(structure),
            **_calc_force_metrics(forces),
        }
        try:
            stress = atoms.get_stress(voigt=False)
            payload["stress_eV_per_A3"] = torch.as_tensor(stress).tolist()
        except Exception:
            payload["stress_eV_per_A3"] = None
        return payload
    finally:
        _finalize_tool_cleanup(atoms)


def _normalize_pes_model_ref(model_ref: str | None) -> str:
    raw = str(model_ref or "PES").strip()
    key = raw.upper().replace("-", "").replace("_", "")
    alias_map = {
        "": "PES",
        "PES": "PES",
        "TENSORNET": "PES",
        "MATPES": "PES",
        "M3GNET": "PES",
        "CHGNET": "PES",
        "QET": "QET_PES_PBE",
        "QETPES": "QET_PES_PBE",
        "QETPESPBE": "QET_PES_PBE",
        "QETPESR2SCAN": "QET_PES_R2SCAN",
        "R2SCAN": "QET_PES_R2SCAN",
    }
    return alias_map.get(key, raw)


@llm_tool(
    name="list_pretrained_models_MatGL",
    description="列出官方或社区 MatGL 预训练模型，便于后续通用模型工具直接调用"
)
async def list_pretrained_models_MatGL(
    search: str = "",
    limit: int = 20,
    official_only: bool = True,
) -> str:
    limit = int(validate_numeric_parameter(limit, "limit", min_val=1, max_val=100))
    payload = await _run_blocking(_list_pretrained_models_sync, search, limit, official_only)
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _list_pretrained_models_sync(search: str, limit: int, official_only: bool) -> dict[str, Any]:
    from huggingface_hub import HfApi

    try:
        api = HfApi()
        models = list(api.list_models(filter="matgl", search=search or None, limit=limit, full=False))
        items = []
        for model in models:
            model_id = getattr(model, "id", None) or getattr(model, "modelId", None)
            if official_only and model_id and not model_id.lower().startswith("materialyze/"):
                continue
            items.append(
                {
                    "id": model_id,
                    "downloads": getattr(model, "downloads", None),
                    "likes": getattr(model, "likes", None),
                    "tags": list(getattr(model, "tags", []) or []),
                }
            )
        return {
            "ok": True,
            "tool": "list_pretrained_models_MatGL",
            "search": search,
            "official_only": bool(official_only),
            "count": len(items),
            "models": items,
        }
    except Exception as exc:
        fallback = [{"id": v} for v in matgl_config.MODEL_REFS.values()]
        return {
            "ok": False,
            "tool": "list_pretrained_models_MatGL",
            "search": search,
            "official_only": bool(official_only),
            "error": str(exc),
            "fallback_models": fallback,
        }


@llm_tool(
    name="clear_matgl_cache_MatGL",
    description="清理 MatGL 下载缓存和当前服务进程内模型缓存，适合模型升级后强制刷新"
)
async def clear_matgl_cache_MatGL() -> str:
    payload = await _run_blocking(_clear_matgl_cache_sync)
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _clear_matgl_cache_sync() -> dict[str, Any]:
    from matgl.config import clear_cache

    in_memory_cleared = matgl_config.clear_model_cache()
    clear_cache(confirm=False)
    return {
        "ok": True,
        "tool": "clear_matgl_cache_MatGL",
        "cleared_in_memory_models": in_memory_cleared,
    }


@llm_tool(
    name="prewarm_matgl_models_MatGL",
    description="预加载一组 MatGL 模型到当前进程，减少首个请求的冷启动延迟"
)
async def prewarm_matgl_models_MatGL(
    model_refs: Optional[str] = None,
) -> str:
    payload = await _run_blocking(_prewarm_matgl_models_sync, model_refs)
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _parse_model_refs_arg(model_refs: Any) -> list[str]:
    if model_refs is None:
        return list(matgl_config.MODEL_REFS.values())
    if isinstance(model_refs, str):
        value = model_refs.strip()
        if not value:
            return list(matgl_config.MODEL_REFS.values())
        if value.startswith("["):
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError("model_refs JSON string must decode to a list")
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]
    if isinstance(model_refs, (list, tuple)):
        return [str(item).strip() for item in model_refs if str(item).strip()]
    raise TypeError("model_refs must be a comma-separated string, JSON array string, or omitted")


def _prewarm_matgl_models_sync(model_refs: Any) -> dict[str, Any]:
    refs = _parse_model_refs_arg(model_refs)
    loaded = []
    for ref in refs:
        model = load_model(ref)
        loaded.append(
            {
                "model_ref": resolve_model_ref(ref),
                "class_name": model.__class__.__name__,
            }
        )
    return {
        "ok": True,
        "tool": "prewarm_matgl_models_MatGL",
        "count": len(loaded),
        "models": loaded,
    }


@llm_tool(
    name="predict_structure_with_model_MatGL",
    description="对任意支持 predict_structure 的 MatGL 预训练模型做通用晶体属性预测"
)
async def predict_structure_with_model_MatGL(
    cif_string: str,
    model_ref: str,
    state_attr: Optional[list[int]] = None,
    optimize_structure: bool = False,
    relax_model_ref: str = "PES",
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _predict_structure_with_model_sync,
        cif_string,
        model_ref,
        state_attr,
        optimize_structure,
        relax_model_ref,
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _predict_structure_with_model_sync(
    cif_string: str,
    model_ref: str,
    state_attr: Optional[list[int]],
    optimize_structure: bool,
    relax_model_ref: str,
    fmax: float,
) -> dict[str, Any]:
    structure = (
        _relax_structure_with_model_sync(cif_string, relax_model_ref, fmax)
        if optimize_structure
        else load_structure_from_cif_string(cif_string)
    )
    tensor_state = None
    try:
        model = load_model(model_ref)
        if not hasattr(model, "predict_structure"):
            raise ValueError(f"Model {resolve_model_ref(model_ref)} does not expose predict_structure()")
        tensor_state = _normalize_state_attr(state_attr)
        prediction = _predict_structure_device_aware(model, structure, tensor_state)
        value = prediction.tolist() if hasattr(prediction, "tolist") else prediction
        return {
            "ok": True,
            "tool": "predict_structure_with_model_MatGL",
            "model_ref": resolve_model_ref(model_ref),
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "optimize_structure": bool(optimize_structure),
            "prediction": value,
        }
    finally:
        del tensor_state
        gc.collect()
        _trim_cuda_cache()


@llm_tool(
    name="relax_structure_with_model_MatGL",
    description="使用任意 MatGL PES 模型对晶体结构做通用弛豫"
)
async def relax_structure_with_model_MatGL(
    cif_string: str,
    model_ref: str = "PES",
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _relax_structure_with_model_payload_sync,
        cif_string,
        _normalize_pes_model_ref(model_ref),
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _relax_structure_with_model_sync(cif_string: str, model_ref: str, fmax: float) -> Structure:
    structure = load_structure_from_cif_string(cif_string)
    potential = load_model(model_ref)
    relaxer = Relaxer(potential=potential)
    relax_results = None
    try:
        relax_results = relaxer.relax(structure, fmax=fmax)
        return relax_results["final_structure"]
    finally:
        del relax_results
        del relaxer
        gc.collect()
        _trim_cuda_cache()


def _relax_structure_with_model_payload_sync(cif_string: str, model_ref: str, fmax: float) -> dict[str, Any]:
    relaxed = _relax_structure_with_model_sync(cif_string, model_ref, fmax)
    return {
        "ok": True,
        "tool": "relax_structure_with_model_MatGL",
        "model_ref": resolve_model_ref(model_ref),
        "formula": relaxed.composition.reduced_formula,
        "num_atoms": len(relaxed),
        "fmax": float(fmax),
        "structure_info": format_basic_structure_info(relaxed),
    }


@llm_tool(
    name="calculate_pes_observables_MatGL",
    description="使用任意 MatGL PES 模型计算总能、每原子能、受力和应力摘要"
)
async def calculate_pes_observables_MatGL(
    cif_string: str,
    model_ref: str = "PES",
    optimize_structure: bool = False,
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _calculate_pes_observables_sync,
        cif_string,
        _normalize_pes_model_ref(model_ref),
        optimize_structure,
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _calculate_pes_observables_sync(
    cif_string: str,
    model_ref: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    structure = (
        _relax_structure_with_model_sync(cif_string, model_ref, fmax)
        if optimize_structure
        else load_structure_from_cif_string(cif_string)
    )
    payload = {
        "ok": True,
        "tool": "calculate_pes_observables_MatGL",
        "formula": structure.composition.reduced_formula,
        "num_atoms": len(structure),
        "optimize_structure": bool(optimize_structure),
        **_calculate_pes_observables_for_structure(structure, model_ref),
    }
    return payload


@llm_tool(
    name="relax_structure_with_qet_MatGL",
    description="使用官方 QET PES 模型对晶体结构做电荷感知弛豫"
)
async def relax_structure_with_qet_MatGL(
    cif_string: str,
    variant: str = "PBE",
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(_relax_structure_with_qet_sync, cif_string, variant, fmax)
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _relax_structure_with_qet_sync(cif_string: str, variant: str, fmax: float) -> dict[str, Any]:
    normalized, model_ref = _resolve_qet_model_ref(variant)
    payload = _relax_structure_with_model_payload_sync(cif_string, model_ref, fmax)
    payload["tool"] = "relax_structure_with_qet_MatGL"
    payload["qet_variant"] = normalized
    return payload


@llm_tool(
    name="calculate_qet_pes_observables_MatGL",
    description="使用官方 QET PES 模型计算总能、每原子能、受力和应力摘要"
)
async def calculate_qet_pes_observables_MatGL(
    cif_string: str,
    variant: str = "PBE",
    optimize_structure: bool = False,
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _calculate_qet_pes_observables_sync,
        cif_string,
        variant,
        optimize_structure,
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _calculate_qet_pes_observables_sync(
    cif_string: str,
    variant: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    normalized, model_ref = _resolve_qet_model_ref(variant)
    payload = _calculate_pes_observables_sync(cif_string, model_ref, optimize_structure, fmax)
    payload["tool"] = "calculate_qet_pes_observables_MatGL"
    payload["qet_variant"] = normalized
    return payload


@llm_tool(
    name="compare_qet_and_tensornet_pes_MatGL",
    description="对同一结构比较 QET PES 与 TensorNet PES 的能量、受力和应力摘要"
)
async def compare_qet_and_tensornet_pes_MatGL(
    cif_string: str,
    qet_variant: str = "PBE",
    optimize_structure: bool = False,
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _compare_qet_and_tensornet_pes_sync,
        cif_string,
        qet_variant,
        optimize_structure,
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _compare_qet_and_tensornet_pes_sync(
    cif_string: str,
    qet_variant: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    normalized, qet_model_ref = _resolve_qet_model_ref(qet_variant)
    structure = (
        _relax_structure_sync(cif_string, fmax)
        if optimize_structure
        else load_structure_from_cif_string(cif_string)
    )
    qet_payload = _calculate_pes_observables_for_structure(structure, qet_model_ref)
    tensornet_payload = _calculate_pes_observables_for_structure(structure, "PES")
    return {
        "ok": True,
        "tool": "compare_qet_and_tensornet_pes_MatGL",
        "formula": structure.composition.reduced_formula,
        "num_atoms": len(structure),
        "optimize_structure": bool(optimize_structure),
        "qet_variant": normalized,
        "qet": qet_payload,
        "tensornet": tensornet_payload,
        "delta_total_energy_eV": qet_payload["total_energy_eV"] - tensornet_payload["total_energy_eV"],
        "delta_energy_per_atom_eV": qet_payload["energy_per_atom_eV"] - tensornet_payload["energy_per_atom_eV"],
        "delta_max_force_eV_per_A": qet_payload["max_force_eV_per_A"] - tensornet_payload["max_force_eV_per_A"],
        "delta_rms_force_eV_per_A": qet_payload["rms_force_eV_per_A"] - tensornet_payload["rms_force_eV_per_A"],
    }


@llm_tool(
    name="run_qet_molecular_dynamics_MatGL",
    description="使用官方 QET PES 模型运行分子动力学模拟"
)
async def run_qet_molecular_dynamics_MatGL(
    cif_string: str,
    variant: str = "PBE",
    temperature_K: float = 300,
    steps: int = 100,
    optimize_structure: bool = True,
    fmax: float = 0.01,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    temperature_K = validate_numeric_parameter(temperature_K, "temperature_K", min_val=1, max_val=5000)
    steps = int(validate_numeric_parameter(steps, "steps", min_val=1, max_val=10000))
    payload = await _run_blocking(
        _run_qet_molecular_dynamics_sync,
        cif_string,
        variant,
        temperature_K,
        steps,
        optimize_structure,
        fmax,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _run_qet_molecular_dynamics_sync(
    cif_string: str,
    variant: str,
    temperature_K: float,
    steps: int,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    normalized, model_ref = _resolve_qet_model_ref(variant)
    final_structure, final_energy = _run_molecular_dynamics_with_model_sync(
        cif_string,
        model_ref,
        temperature_K,
        steps,
        optimize_structure,
        fmax,
    )
    return {
        "ok": True,
        "tool": "run_qet_molecular_dynamics_MatGL",
        "qet_variant": normalized,
        "model_ref": model_ref,
        "formula": final_structure.composition.reduced_formula,
        "num_atoms": len(final_structure),
        "temperature_K": float(temperature_K),
        "steps": int(steps),
        "optimize_structure": bool(optimize_structure),
        "final_potential_energy_eV": float(final_energy),
        "structure_info": format_basic_structure_info(final_structure),
    }


@llm_tool(name="relax_crystal_structure_MatGL",
          description="使用MatGL最新TensorNet势能优化晶体结构几何构型")
async def relax_crystal_structure_MatGL(
    cif_string: str,
    fmax: float = 0.01
) -> str:
    """
    优化晶体结构几何构型以找到平衡构型。

    使用最新 TensorNet MatPES 势能进行快速结构优化，无需DFT计算。
    接受CIF格式的结构字符串或文件内容。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        fmax: 力收敛阈值，单位eV/Å (默认: 0.01)

    Returns:
        包含优化结果的Markdown格式字符串
    """
    # 参数验证
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)

    relaxed_structure = await _run_blocking(_relax_structure_sync, cif_string, fmax)

    reduced_formula = relaxed_structure.composition.reduced_formula

    # 格式化基本结构信息
    structure_info = format_basic_structure_info(relaxed_structure)

    return (f"## 结构优化结果\n\n"
            f"- **分子式**: `{reduced_formula}`\n"
            f"- **力收敛阈值**: `{fmax} eV/Å`\n"
            f"- **优化状态**: `成功优化`\n\n"
            f"{structure_info}")


def _relax_structure_sync(cif_string: str, fmax: float) -> Structure:
    structure = load_structure_from_cif_string(cif_string)
    potential = load_model("PES")
    relaxer = Relaxer(potential=potential)
    relax_results = None
    try:
        relax_results = relaxer.relax(structure, fmax=fmax)
        return relax_results["final_structure"]
    finally:
        del relax_results
        del relaxer
        gc.collect()
        _trim_cuda_cache()


async def _relax_crystal_structure_internal(cif_string: str, fmax: float = 0.01) -> Structure:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    return await _run_blocking(_relax_structure_sync, cif_string, fmax)


@llm_tool(name="predict_formation_energy_MatGL",
          description="使用MatGL M3GNet模型预测晶体结构的形成能")
async def predict_formation_energy_MatGL(
    cif_string: str,
    optimize_structure: bool = False,
    fmax: float = 0.01
) -> str:
    """
    使用M3GNet形成能模型预测晶体结构的形成能。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        optimize_structure: 是否在预测前优化结构 (默认: False)
        fmax: 结构优化的力收敛阈值，单位eV/Å (默认: 0.01)

    Returns:
        包含预测形成能的Markdown格式字符串
    """
    # 参数验证
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)

    payload = await _predict_formation_energy_payload(
        cif_string=cif_string,
        optimize_structure=optimize_structure,
        fmax=fmax,
    )

    return (
        f"## 形成能预测结果\n\n"
        f"- **分子式**: `{payload['formula']}`\n"
        f"- **结构状态**: `{payload['optimization_status']}`\n"
        f"- **形成能**: `{payload['formation_energy_per_atom_eV']:.3f} eV/atom`\n\n"
        f"{payload['structure_info']}\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )


@llm_tool(name="run_molecular_dynamics_MatGL",
          description="使用MatGL最新TensorNet势能运行分子动力学模拟")
async def run_molecular_dynamics_MatGL(
    cif_string: str,
    temperature_K: float = 300,
    steps: int = 100,
    optimize_structure: bool = True,
    fmax: float = 0.01
) -> str:
    """
    使用 TensorNet MatPES 势能运行分子动力学模拟。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        temperature_K: MD模拟温度，单位K (默认: 300)
        steps: MD模拟步数 (默认: 100)
        optimize_structure: 是否在模拟前优化结构 (默认: True)
        fmax: 结构优化的力收敛阈值，单位eV/Å (默认: 0.01)

    Returns:
        包含模拟结果的Markdown格式字符串
    """
    # 参数验证
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    temperature_K = validate_numeric_parameter(temperature_K, "temperature_K", min_val=1, max_val=5000)
    steps = int(validate_numeric_parameter(steps, "steps", min_val=1, max_val=10000))

    final_structure, final_energy = await _run_blocking(
        _run_molecular_dynamics_sync,
        cif_string,
        temperature_K,
        steps,
        optimize_structure,
        fmax,
    )
    reduced_formula = final_structure.composition.reduced_formula

    # 格式化优化状态和结构信息
    optimization_status = format_optimization_status(optimize_structure)
    structure_info = format_basic_structure_info(final_structure)

    return (f"## 分子动力学模拟结果\n\n"
            f"- **分子式**: `{reduced_formula}`\n"
            f"- **结构状态**: `{optimization_status}`\n"
            f"- **模拟温度**: `{temperature_K} K`\n"
            f"- **模拟步数**: `{steps}`\n"
            f"- **最终势能**: `{float(final_energy):.3f} eV`\n\n"
            f"{structure_info}")


def _run_molecular_dynamics_sync(
    cif_string: str,
    temperature_K: float,
    steps: int,
    optimize_structure: bool,
    fmax: float,
) -> tuple[Structure, float]:
    return _run_molecular_dynamics_with_model_sync(
        cif_string,
        "PES",
        temperature_K,
        steps,
        optimize_structure,
        fmax,
    )


def _run_molecular_dynamics_with_model_sync(
    cif_string: str,
    model_ref: str,
    temperature_K: float,
    steps: int,
    optimize_structure: bool,
    fmax: float,
) -> tuple[Structure, float]:
    structure = (
        _relax_structure_with_model_sync(cif_string, model_ref, fmax)
        if optimize_structure
        else load_structure_from_cif_string(cif_string)
    )
    potential = load_model(model_ref)
    ase_adaptor = AseAtomsAdaptor()
    atoms = ase_adaptor.get_atoms(structure)
    driver = None
    try:
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K)
        driver = MolecularDynamics(atoms, potential=potential, temperature=temperature_K)
        driver.run(steps)
        final_energy = float(atoms.get_potential_energy())
        return ase_adaptor.get_structure(atoms), final_energy
    finally:
        del driver
        _finalize_tool_cleanup(atoms)


@llm_tool(name="calculate_single_point_energy_MatGL",
          description="使用MatGL最新TensorNet势能计算晶体结构的单点能")
async def calculate_single_point_energy_MatGL(
    cif_string: str,
    optimize_structure: bool = True,
    fmax: float = 0.01
) -> str:
    """
    使用 TensorNet MatPES 势能计算晶体结构的单点能。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        optimize_structure: 是否在计算前优化结构 (默认: True)
        fmax: 结构优化的力收敛阈值，单位eV/Å (默认: 0.01)

    Returns:
        包含计算势能的Markdown格式字符串
    """
    # 参数验证
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)

    structure, energy = await _run_blocking(
        _calculate_single_point_energy_sync,
        cif_string,
        optimize_structure,
        fmax,
    )
    reduced_formula = structure.composition.reduced_formula

    # 格式化优化状态和结构信息
    optimization_status = format_optimization_status(optimize_structure)
    structure_info = format_basic_structure_info(structure)

    return (f"## 单点能计算结果\n\n"
            f"- **分子式**: `{reduced_formula}`\n"
            f"- **结构状态**: `{optimization_status}`\n"
            f"- **势能**: `{float(energy):.3f} eV`\n\n"
            f"{structure_info}")


def _calculate_single_point_energy_sync(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> tuple[Structure, float]:
    structure = _relax_structure_sync(cif_string, fmax) if optimize_structure else load_structure_from_cif_string(cif_string)
    potential = load_model("PES")
    ase_adaptor = AseAtomsAdaptor()
    atoms = ase_adaptor.get_atoms(structure)
    try:
        atoms.calc = PESCalculator(potential)
        energy = float(atoms.get_potential_energy())
        return structure, energy
    finally:
        _finalize_tool_cleanup(atoms)

@llm_tool(name="predict_multi_fidelity_band_gap_MatGL",
          description="使用MatGL MEGNet多保真度模型预测晶体结构的带隙")
async def predict_multi_fidelity_band_gap_MatGL(
    cif_string: str,
    optimize_structure: bool = False,
    fmax: float = 0.01
) -> str:
    """
    使用MEGNet多保真度带隙模型预测晶体结构的带隙。

    使用多种DFT方法（PBE、GLLB-SC、HSE、SCAN）预测带隙，
    可选择在预测前优化结构以获得更准确的结果。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        optimize_structure: 是否在预测前优化结构 (默认: False)
        fmax: 结构优化的力收敛阈值，单位eV/Å (默认: 0.01)

    Returns:
        包含各种方法预测带隙的Markdown格式字符串
    """
    # 参数验证
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)

    payload = await _predict_multi_fidelity_band_gap_payload(
        cif_string=cif_string,
        optimize_structure=optimize_structure,
        fmax=fmax,
    )

    # 构建预测结果表格
    predictions_table = "| 方法      | 带隙 (eV) |\n"
    predictions_table += "|-----------|----------|\n"
    for method in payload["methods"]:
        predictions_table += f"| {method:9s} | {payload['predictions_eV'][method]:8.3f} |\n"

    return (
        f"## 多保真度带隙预测结果\n\n"
        f"- **分子式**: `{payload['formula']}`\n"
        f"- **结构状态**: `{payload['optimization_status']}`\n"
        f"- **模型**: `{payload['model_ref']}`\n\n"
        f"### 预测带隙\n\n"
        f"{predictions_table}\n"
        f"{payload['structure_info']}\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )


async def _predict_formation_energy_payload(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    return await _run_blocking(_predict_formation_energy_payload_sync, cif_string, optimize_structure, fmax)


async def _predict_multi_fidelity_band_gap_payload(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    return await _run_blocking(_predict_multi_fidelity_band_gap_payload_sync, cif_string, optimize_structure, fmax)


def _predict_formation_energy_payload_sync(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    structure = _relax_structure_sync(cif_string, fmax) if optimize_structure else load_structure_from_cif_string(cif_string)
    try:
        model = load_model("EFORM")
        eform = _tensor_to_float(_predict_structure_device_aware(model, structure))
        reduced_formula = structure.composition.reduced_formula
        optimization_status = format_optimization_status(optimize_structure)
        structure_info = format_basic_structure_info(structure)

        return {
            "ok": True,
            "tool": "predict_formation_energy_MatGL",
            "formula": reduced_formula,
            "formation_energy_per_atom_eV": eform,
            "num_atoms": len(structure),
            "optimize_structure": bool(optimize_structure),
            "optimization_status": optimization_status,
            "structure_info": structure_info,
            "model_ref": get_model_ref("EFORM"),
        }
    finally:
        gc.collect()
        _trim_cuda_cache()


def _predict_multi_fidelity_band_gap_payload_sync(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    structure = _relax_structure_sync(cif_string, fmax) if optimize_structure else load_structure_from_cif_string(cif_string)
    try:
        model = load_model("BAND_GAP_MFI")

        methods = ["PBE", "GLLB-SC", "HSE", "SCAN"]
        method_map = {"PBE": 0, "GLLB-SC": 1, "HSE": 2, "SCAN": 3}
        predictions: dict[str, float] = {}
        for method in methods:
            state_attr = torch.tensor([method_map[method]], dtype=torch.long)
            try:
                bandgap = _predict_structure_device_aware(model, structure, state_attr)
                predictions[method] = _tensor_to_float(bandgap)
            finally:
                del state_attr

        reduced_formula = structure.composition.reduced_formula
        optimization_status = format_optimization_status(optimize_structure)
        structure_info = format_basic_structure_info(structure)

        return {
            "ok": True,
            "tool": "predict_multi_fidelity_band_gap_MatGL",
            "formula": reduced_formula,
            "predictions_eV": predictions,
            "methods": methods,
            "num_atoms": len(structure),
            "optimize_structure": bool(optimize_structure),
            "optimization_status": optimization_status,
            "structure_info": structure_info,
            "recommended_fidelity": "SCAN",
            "model_ref": get_model_ref("BAND_GAP_MFI"),
        }
    finally:
        gc.collect()
        _trim_cuda_cache()


def _flatten_chgnet_magmoms(raw_magmoms: Any) -> tuple[list[float], float]:
    """Convert CHGNet magnetic-moment outputs into per-site magnitudes and a total moment.

    CHGNet docs expose magnetic moments through the ``m`` field, but the exact
    tensor shape can vary across versions. We therefore normalize the output into:
    1. site-wise magnetic-moment magnitudes in mu_B
    2. a single total magnetic-moment magnitude in mu_B
    """
    if hasattr(raw_magmoms, "detach"):
        raw_magmoms = raw_magmoms.detach().cpu().tolist()
    elif hasattr(raw_magmoms, "tolist"):
        raw_magmoms = raw_magmoms.tolist()

    if raw_magmoms is None:
        return [], 0.0

    if not isinstance(raw_magmoms, list):
        raw_magmoms = [float(raw_magmoms)]

    if not raw_magmoms:
        return [], 0.0

    if isinstance(raw_magmoms[0], list):
        site_magnitudes: list[float] = []
        total_vector = [0.0, 0.0, 0.0]
        for item in raw_magmoms:
            values = [float(x) for x in item]
            if len(values) == 1:
                site_magnitudes.append(abs(values[0]))
                total_vector[0] += values[0]
            else:
                while len(values) < 3:
                    values.append(0.0)
                site_magnitudes.append(math.sqrt(sum(v * v for v in values[:3])))
                for idx, value in enumerate(values[:3]):
                    total_vector[idx] += value
        total_moment = math.sqrt(sum(v * v for v in total_vector))
        return site_magnitudes, total_moment

    scalar_moments = [float(x) for x in raw_magmoms]
    site_magnitudes = [abs(x) for x in scalar_moments]
    total_moment = abs(sum(scalar_moments))
    return site_magnitudes, total_moment


@llm_tool(
    name="classify_metallicity_from_band_gap",
    description="使用MatGL多保真度带隙预测结果对材料进行金属性分类"
)
async def classify_metallicity_from_band_gap(
    cif_string: str,
    fidelity: str = "SCAN",
    threshold_eV: float = 0.05,
    optimize_structure: bool = False,
    fmax: float = 0.01,
) -> str:
    """
    基于多保真度带隙预测结果对材料进行金属性分类。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        fidelity: 采用哪种带隙保真度进行分类，可选 PBE/GLLB-SC/HSE/SCAN
        threshold_eV: 视为金属的带隙阈值，默认 0.05 eV
        optimize_structure: 是否在预测前优化结构
        fmax: 结构优化力阈值

    Returns:
        包含金属性分类结果的 Markdown + JSON
    """
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    threshold_eV = validate_numeric_parameter(threshold_eV, "threshold_eV", min_val=0.0, max_val=1.0)
    fidelity = fidelity.upper().replace("-", "_")
    fidelity_aliases = {
        "MULTI": "SCAN",
        "MULTI_FIDELITY": "SCAN",
        "MULTIFIDELITY": "SCAN",
        "HIGH": "HSE",
        "DEFAULT": "SCAN",
    }
    fidelity = fidelity_aliases.get(fidelity, fidelity).replace("_", "-")
    allowed = {"PBE", "GLLB-SC", "HSE", "SCAN"}
    if fidelity not in allowed:
        raise ValueError(f"fidelity must be one of {sorted(allowed)}, got {fidelity}")

    payload = await _predict_multi_fidelity_band_gap_payload(
        cif_string=cif_string,
        optimize_structure=optimize_structure,
        fmax=fmax,
    )
    band_gap = float(payload["predictions_eV"][fidelity])
    is_metal = band_gap <= threshold_eV
    classification = "metal" if is_metal else "non-metal"

    result_payload = {
        "ok": True,
        "tool": "classify_metallicity_from_band_gap",
        "formula": payload["formula"],
        "selected_fidelity": fidelity,
        "band_gap_eV": band_gap,
        "threshold_eV": float(threshold_eV),
        "is_metal": bool(is_metal),
        "classification": classification,
        "all_band_gaps_eV": payload["predictions_eV"],
        "optimize_structure": bool(optimize_structure),
    }

    return (
        f"## 金属性分类结果\n\n"
        f"- **分子式**: `{payload['formula']}`\n"
        f"- **选定保真度**: `{fidelity}`\n"
        f"- **预测带隙**: `{band_gap:.4f} eV`\n"
        f"- **金属阈值**: `{threshold_eV:.4f} eV`\n"
        f"- **分类结果**: `{classification}`\n\n"
        f"```json\n{json.dumps(result_payload, ensure_ascii=False)}\n```"
    )


@llm_tool(
    name="screen_magnetic_moments_CHGNet",
    description="使用CHGNet对晶体结构做局域磁矩screening，给出磁性候选线索而非最终磁序判定"
)
async def screen_magnetic_moments_CHGNet(
    cif_string: str,
    moment_threshold_muB: float = 0.5,
    optimize_structure: bool = False,
    fmax: float = 0.01,
) -> str:
    """
    使用 CHGNet 对晶体结构做磁性 screening。

    这个工具只输出局域磁矩线索和“是否值得进一步做 spin-polarized DFT”的筛查结果，
    不直接宣称 FM / AFM / NM 等最终磁序。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        moment_threshold_muB: 视为显著局域磁矩的阈值，默认 0.5 mu_B
        optimize_structure: 是否先用 CHGNet 进行快速几何优化后再筛查
        fmax: 结构优化力阈值

    Returns:
        包含磁性 screening 结果的 Markdown + JSON
    """
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    moment_threshold_muB = validate_numeric_parameter(
        moment_threshold_muB, "moment_threshold_muB", min_val=0.0, max_val=10.0
    )

    payload = await _run_blocking(
        _screen_magnetic_moments_sync,
        cif_string,
        moment_threshold_muB,
        optimize_structure,
        fmax,
    )

    reduced_formula = payload["formula"]
    optimization_status = payload["optimization_status"]
    has_nonzero_local_moments = payload["has_nonzero_local_moments"]
    total_magnetic_moment = payload["total_magnetic_moment_muB"]
    magnetic_sites = payload["magnetic_sites"]
    magnetic_species = payload["magnetic_species"]
    interpretation = payload["screening_interpretation"]

    return (
        f"## CHGNet磁性筛查结果\n\n"
        f"- **分子式**: `{reduced_formula}`\n"
        f"- **结构状态**: `{optimization_status}`\n"
        f"- **磁矩阈值**: `{moment_threshold_muB:.3f} μB`\n"
        f"- **检测到显著局域磁矩**: `{has_nonzero_local_moments}`\n"
        f"- **显著磁矩位点数**: `{len(magnetic_sites)}`\n"
        f"- **总磁矩幅值**: `{total_magnetic_moment:.4f} μB`\n"
        f"- **磁性元素**: `{', '.join(magnetic_species) if magnetic_species else 'none detected'}`\n\n"
        f"{interpretation}\n\n"
        f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    )


def _screen_magnetic_moments_sync(
    cif_string: str,
    moment_threshold_muB: float,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    structure = load_structure_from_cif_string(cif_string)
    potential = _get_chgnet()
    optimization_status = format_optimization_status(optimize_structure)
    relaxer = None
    result = None
    atoms = None
    if optimize_structure:
        relaxer = Relaxer(potential=potential)
        result = relaxer.relax(structure, fmax=fmax)
        structure = result.get("final_structure", structure)

    try:
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.calc = PESCalculator(potential)
        try:
            site_moments_raw = [abs(float(v)) for v in np.asarray(atoms.get_magnetic_moments()).reshape(-1).tolist()]
        except Exception as exc:
            raise RuntimeError(f"Offline CHGNet magnetic-moment prediction failed: {exc}") from exc
        try:
            total_moment_raw = abs(float(atoms.get_magnetic_moment()))
        except Exception:
            total_moment_raw = float(np.sum(site_moments_raw))
        site_moment_magnitudes = [round(v, 6) for v in site_moments_raw]
        total_magnetic_moment = round(total_moment_raw, 6)

        magnetic_sites: list[dict[str, Any]] = []
        magnetic_species: list[str] = []
        for idx, (site, moment) in enumerate(zip(structure.sites, site_moment_magnitudes), start=1):
            if moment >= moment_threshold_muB:
                specie = str(site.specie)
                magnetic_sites.append({"site_index": idx, "element": specie, "moment_muB": moment})
                if specie not in magnetic_species:
                    magnetic_species.append(specie)

        has_nonzero_local_moments = bool(magnetic_sites)
        if has_nonzero_local_moments:
            interpretation = (
                "CHGNet detects nonzero local moments, so this structure should be treated as a magnetic candidate "
                "and validated with spin-polarized DFT before making any final magnetic-ordering claim."
            )
        else:
            interpretation = (
                "CHGNet does not detect substantial local moments above the screening threshold. "
                "This is weak evidence against strong magnetism, but not a final non-magnetic ground-state proof."
            )

        return {
            "ok": True,
            "tool": "screen_magnetic_moments_CHGNet",
            "formula": structure.composition.reduced_formula,
            "num_atoms": len(structure),
            "moment_threshold_muB": float(moment_threshold_muB),
            "optimize_structure": bool(optimize_structure),
            "optimization_status": optimization_status,
            "site_moment_magnitudes_muB": site_moment_magnitudes,
            "num_magnetic_sites": len(magnetic_sites),
            "magnetic_sites": magnetic_sites,
            "magnetic_species": magnetic_species,
            "has_nonzero_local_moments": has_nonzero_local_moments,
            "total_magnetic_moment_muB": total_magnetic_moment,
            "screening_interpretation": interpretation,
        }
    finally:
        del result
        del relaxer
        _finalize_tool_cleanup(atoms)


@llm_tool(name="predict_internal_energy_MatGL",
          description="使用CHGNet模型预测晶体结构的内部能量")
async def predict_internal_energy_MatGL(
    cif_string: str,
    optimize_structure: bool = True,
    fmax: float = 0.01
) -> str:
    """
    用官方 chgnet v0.3.0 预测晶体结构的 DFT 总能量（per-atom），并可选结构优化。

    背景：之前尝试过两条路径
      1) potential.model.predict_structure(...)  - 跳过 element_refs，输出 ~0.02 eV/atom 量级残差，错；
      2) PESCalculator(potential).get_potential_energy() - 走 matgl Potential 完整管线，输出 ~-3 eV/atom，
         但和 MP 真 DFT 总能仍系统性差 280-900 meV/atom（matgl 的 MPtrj-11M 模型权重和官方实际跑出来差异大）。
    都用 Li2ZrCl6 hull above 验证后发现：第 2 条仍然给出 ~0.9 eV/atom 的离谱 hull above。
    最终切到官方 chgnet 包：CsPbBr3 hull 0.026，Li2ZrCl6 hull 0.041，和 MP/论文报告完美吻合。

    Args:
        cif_string: CIF 格式晶体结构字符串
        optimize_structure: 是否先用 chgnet 的 StructOptimizer 优化结构 (默认 True)
        fmax: 结构优化的力收敛阈值，单位 eV/Å (默认 0.01)

    Returns:
        Markdown 格式的能量预测结果（per-atom 总能与 MP-PBE 体系兼容）
    """
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(_predict_internal_energy_sync, cif_string, optimize_structure, fmax)
    reduced_formula = payload["formula"]
    energy = payload["total_internal_energy_eV"]
    energy_per_atom = payload["energy_per_atom"]
    num_atoms = payload["num_atoms"]

    # 格式化优化状态和结构信息
    optimization_status = format_optimization_status(optimize_structure)

    return (f"## CHGNet内部能量预测结果\n\n"
            f"- **分子式**: `{reduced_formula}`\n"
            f"- **总内部能量**: `{energy:.6f} eV`\n"
            f"- **单原子能量**: `{energy_per_atom:.6f} eV/atom`\n"
            f"- **原子数**: `{num_atoms}`\n"
            f"- **结构状态**: `{optimization_status}`\n\n"
            f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```")


def _predict_internal_energy_sync(
    cif_string: str,
    optimize_structure: bool,
    fmax: float,
) -> dict[str, Any]:
    structure = load_structure_from_cif_string(cif_string)
    potential = _get_chgnet()
    relaxer = None
    result = None
    atoms = None
    if optimize_structure:
        relaxer = Relaxer(potential=potential)
        result = relaxer.relax(structure, fmax=fmax)
        structure = result.get("final_structure", structure)

    try:
        atoms = AseAtomsAdaptor().get_atoms(structure)
        atoms.calc = PESCalculator(potential)
        num_atoms = len(structure)
        total_energy = float(atoms.get_potential_energy())
        energy_per_atom = total_energy / num_atoms
        return {
            "ok": True,
            "tool": "predict_internal_energy_MatGL",
            "formula": structure.composition.reduced_formula,
            "total_internal_energy_eV": total_energy,
            "energy_per_atom": energy_per_atom,
            "num_atoms": num_atoms,
            "optimize_structure": bool(optimize_structure),
        }
    finally:
        del result
        del relaxer
        _finalize_tool_cleanup(atoms)


@llm_tool(name="predict_bulk_modulus_MatGL",
          description="使用MatGL最新TensorNet模型预测晶体结构的体积模量")
async def predict_bulk_modulus_MatGL(
    cif_string: str,
    relax_structure: bool = False,
    fmax: float = 0.01
) -> str:
    """
    使用最新 TensorNet MatPES PBE 模型 + matcalc.ElasticityCalc 预测体积模量。

    基于 Materials Potential Energy Surface (MatPES) 数据集训练的 TensorNet 模型，
    用于计算晶体结构的弹性性质。输出 Voigt-Reuss-Hill 平均的体积模量 K_VRH（GPa）。

    Args:
        cif_string: CIF格式的晶体结构字符串或文件内容
        relax_structure: 是否让 matcalc 在弹性计算前先优化平衡结构 (默认: False)。
            注意：ElasticityCalc 会在内部复用同一计算器完成平衡态弛豫与形变态应力计算。
        fmax: 结构优化的力收敛阈值，单位 eV/Å (默认: 0.01)

    Returns:
        包含体积模量预测结果的Markdown格式字符串
    """
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(_predict_elastic_properties_sync, cif_string, relax_structure, fmax, False, False)

    return (f"## 体积模量预测结果\n\n"
            f"- **分子式**: `{payload['formula']}`\n"
            f"- **结构状态**: `{payload['optimization_status']}`\n"
            f"- **体积模量 (K_VRH)**: `{payload['bulk_modulus_vrh_GPa']:.2f} GPa`\n\n"
            f"{payload['structure_info']}")


def _predict_bulk_modulus_sync(
    cif_string: str,
    relax_structure: bool,
    fmax: float,
) -> tuple[Structure, float]:
    payload = _predict_elastic_properties_sync(cif_string, relax_structure, fmax, False, False)
    structure = load_structure_from_cif_string(cif_string)
    return structure, float(payload["bulk_modulus_vrh_GPa"])


@llm_tool(
    name="predict_elastic_properties_MatGL",
    description="使用 MatGL TensorNet 势能和 matcalc 预测完整弹性性质，包括体积模量、剪切模量、杨氏模量和泊松比"
)
async def predict_elastic_properties_MatGL(
    cif_string: str,
    relax_structure: bool = False,
    fmax: float = 0.01,
    symmetry: bool = False,
    include_tensors: bool = False,
) -> str:
    fmax = validate_numeric_parameter(fmax, "fmax", min_val=0.001, max_val=1.0)
    payload = await _run_blocking(
        _predict_elastic_properties_sync,
        cif_string,
        relax_structure,
        fmax,
        symmetry,
        include_tensors,
    )
    return f"```json\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n```"


def _predict_elastic_properties_sync(
    cif_string: str,
    relax_structure: bool,
    fmax: float,
    symmetry: bool,
    include_tensors: bool,
) -> dict[str, Any]:
    structure = load_structure_from_cif_string(cif_string)
    calculator = _load_matcalc_pes_calculator()
    elasticity = None
    props = None
    try:
        elasticity = mtc.ElasticityCalc(
            calculator,
            relax_structure=relax_structure,
            fmax=fmax,
            symmetry=symmetry,
        )
        props = elasticity.calc(structure)
        equilibrium_structure = props.get("structure", structure)
        elastic_tensor = props["elastic_tensor"]
        payload = {
            "ok": True,
            "tool": "predict_elastic_properties_MatGL",
            "formula": equilibrium_structure.composition.reduced_formula,
            "num_atoms": len(equilibrium_structure),
            "optimize_structure": bool(relax_structure),
            "optimization_status": format_optimization_status(relax_structure),
            "structure_info": format_basic_structure_info(equilibrium_structure),
            "bulk_modulus_vrh_GPa": float(props["bulk_modulus_vrh"]) * _GPA_PER_EV_A3,
            "shear_modulus_vrh_GPa": float(props["shear_modulus_vrh"]) * _GPA_PER_EV_A3,
            "youngs_modulus_GPa": float(props["youngs_modulus"]) * _GPA_PER_EV_A3,
            "poisson_ratio": float(elastic_tensor.homogeneous_poisson),
            "residuals_sum": float(props["residuals_sum"]),
            "symmetry_reduction": bool(symmetry),
            "model_ref": get_model_ref("PES"),
        }
        if include_tensors:
            payload["elastic_tensor_GPa"] = (np.asarray(elastic_tensor.voigt) * _GPA_PER_EV_A3).tolist()
            payload["compliance_tensor_GPa_inverse"] = (
                np.asarray(elastic_tensor.compliance_tensor) / _GPA_PER_EV_A3
            ).tolist()
        return payload
    finally:
        del props
        del elasticity
        gc.collect()
        _trim_cuda_cache()


def _load_matcalc_pes_calculator():
    potential = load_model("PES")
    return PESCalculator(potential)
