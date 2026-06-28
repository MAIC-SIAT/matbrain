"""PyMatGen Tools - 基于PyMatGen的材料科学工具函数

包含化学式验证、价态检验、结构几何分析等材料分析工具
"""

import pandas as pd
import numpy as np
import asyncio
import json
from typing import Dict, Any, Union, List, Annotated, Tuple
from pymatgen.core import Composition, Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.analysis.magnetism.analyzer import CollinearMagneticStructureAnalyzer
try:
    from pymatgen.analysis.magnetism.analyzer import MagneticStructureEnumerator
except Exception:
    MagneticStructureEnumerator = None
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Molecule
from pymatgen.io.cif import CifWriter
from pymatgen.io.vasp import Poscar, Incar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.transformations.standard_transformations import OrderDisorderedStructureTransformation
from pymatgen.core.periodic_table import Element
from pymatgen.ext.matproj import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import re
import warnings
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import io
import base64
import tempfile
import os

from core import llm_tool
from core.utils import validate_chemical_formula, load_structure_from_cif_string, validate_numeric_parameter,_call_mcp_tool
from config import pymatgen_config, server_config

warnings.filterwarnings('ignore')


def _extract_json_block(text: str) -> Dict[str, Any] | None:
    """Extract a fenced or raw JSON object from tool text output."""
    if not text:
        return None

    stripped = text.strip()
    candidates: list[str] = []

    fence_match = re.search(r"```json\s*(\{.*?\}|\[.*?\])\s*```", stripped, flags=re.S)
    if fence_match:
        candidates.append(fence_match.group(1).strip())

    if stripped.startswith("{") or stripped.startswith("["):
        candidates.append(stripped)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


# ============================================================================
# Materials Project数据获取函数
# ============================================================================

def get_materials_project_entries_by_elements(api_key: str, elements: List[str]) -> List[PDEntry]:
    """
    根据元素列表从Materials Project获取化学系统的所有化合物条目

    Args:
        api_key: Materials Project API密钥
        elements: 元素符号列表，例如['Fe', 'V', 'S']

    Returns:
        List[PDEntry]: 相图条目列表
    """
    try:
        with MPRester(api_key) as mpr:
            # 限定 GGA_GGA+U 同一泛函族条目。MP 自 2025 起向 chemsys 返回的默认结果
            # 中混入了 R2SCAN 条目；其能量基准与 PBE/GGA+U 不在同一参考态，混进同
            # 一个 PhaseDiagram 会让 e_hull 给出 2-3 eV/atom 的垃圾值。matgl-mcp 的
            # CHGNet 走 MPtrj (PBE) 训练，必须只对 GGA_GGA+U 这一族条目做相图。
            entries = mpr.get_entries_in_chemsys(
                elements,
                additional_criteria={"thermo_types": ["GGA_GGA+U"]},
            )
        return entries
    except Exception as e:
        raise ValueError(f"从Materials Project获取数据失败: {str(e)}")


def extract_elements_from_cif(cif_string: str) -> List[str]:
    """
    从CIF字符串中提取元素列表

    Args:
        cif_string: CIF格式的结构字符串

    Returns:
        List[str]: 元素符号列表
    """
    try:
        structure = load_structure_from_cif_string(cif_string)
        elements = [str(element) for element in structure.composition.elements]
        return sorted(elements)  # 排序以保证一致性
    except Exception as e:
        raise ValueError(f"从CIF中提取元素失败: {str(e)}")


def create_custom_pdentry(composition_str: str, energy_per_atom: float) -> PDEntry:
    """
    根据组分和单原子能量创建PDEntry对象

    Args:
        composition_str: 化学式字符串
        energy_per_atom: 单原子能量 (eV/atom)

    Returns:
        PDEntry: 相图条目对象
    """
    try:
        composition = Composition(composition_str)
        total_energy = energy_per_atom * composition.num_atoms
        entry = PDEntry(composition, total_energy)
        return entry
    except Exception as e:
        raise ValueError(f"创建PDEntry失败: {str(e)}")

async def call_matgl_energy_prediction(cif_string: str, optimize_structure: bool) -> Dict[str, Any]:
    """
    调用MatGL能量预测工具并返回简化的结果

    Args:
        cif_string: CIF格式的结构字符串
        optimize_structure: 是否优化结构

    Returns:
        Dict包含energy_per_atom和num_atoms两个字段
    """
    sse_url = server_config.MATGL_MCP_SSE_URL
    result = await _call_mcp_tool(sse_url,'predict_internal_energy_MatGL',{
        'cif_string': cif_string,
        'optimize_structure': optimize_structure
    })

    if not result.get("success", False):
        return {
            "energy_per_atom": None,
            "num_atoms": None,
            "error": result.get("content", "Unknown error")
        }

    content = result.get("content", "")

    # 优先读取机器可读负载，避免依赖 Markdown 文本格式。
    payload = _extract_json_block(content)
    if isinstance(payload, dict):
        energy_per_atom = payload.get("energy_per_atom")
        num_atoms = payload.get("num_atoms")
        if energy_per_atom is not None and num_atoms is not None:
            try:
                return {
                    "energy_per_atom": float(energy_per_atom),
                    "num_atoms": int(num_atoms),
                }
            except Exception as exc:
                return {
                    "energy_per_atom": None,
                    "num_atoms": None,
                    "error": f"Failed to cast MatGL JSON payload: {str(exc)}",
                }

    # 回退到 Markdown 文本解析，兼容旧容器输出。
    try:
        energy_match = re.search(r'- \*\*单原子能量\*\*: `([\d.-]+) eV/atom`', content)
        if energy_match:
            energy_per_atom = float(energy_match.group(1))
        else:
            energy_per_atom = None

        atoms_match = re.search(r'- \*\*原子数\*\*: `(\d+)`', content)
        if atoms_match:
            num_atoms = int(atoms_match.group(1))
        else:
            num_atoms = None

        return {
            "energy_per_atom": energy_per_atom,
            "num_atoms": num_atoms
        }

    except Exception as e:
        return {
            "energy_per_atom": None,
            "num_atoms": None,
            "error": f"Failed to parse MatGL result: {str(e)}"
        }


async def call_matgl_formation_energy_prediction(cif_string: str, optimize_structure: bool) -> Dict[str, Any]:
    """
    调用 MatGL 形成能预测工具并返回结构化结果。

    Returns:
        Dict 包含 formation_energy_per_atom、num_atoms 等字段
    """
    sse_url = server_config.MATGL_MCP_SSE_URL
    result = await _call_mcp_tool(
        sse_url,
        'predict_formation_energy_MatGL',
        {
            'cif_string': cif_string,
            'optimize_structure': optimize_structure
        }
    )

    if not result.get("success", False):
        return {
            "formation_energy_per_atom": None,
            "num_atoms": None,
            "error": result.get("content", "Unknown error")
        }

    content = result.get("content", "")
    payload = _extract_json_block(content)
    if isinstance(payload, dict):
        value = payload.get("formation_energy_per_atom_eV")
        num_atoms = payload.get("num_atoms")
        if value is not None and num_atoms is not None:
            try:
                return {
                    "formation_energy_per_atom": float(value),
                    "num_atoms": int(num_atoms),
                    "formula": payload.get("formula"),
                }
            except Exception as exc:
                return {
                    "formation_energy_per_atom": None,
                    "num_atoms": None,
                    "error": f"Failed to cast MatGL formation-energy JSON payload: {str(exc)}",
                }

    # 回退：兼容旧输出，仅解析 markdown 主体
    try:
        energy_match = re.search(r'- \*\*形成能\*\*: `([\d.-]+) eV/atom`', content)
        value = float(energy_match.group(1)) if energy_match else None
        num_atoms = None
        return {
            "formation_energy_per_atom": value,
            "num_atoms": num_atoms,
        }
    except Exception as exc:
        return {
            "formation_energy_per_atom": None,
            "num_atoms": None,
            "error": f"Failed to parse MatGL formation-energy result: {str(exc)}",
        }

# ============================================================================
# 热力学稳定性分析函数
# ============================================================================

def analyze_phase_diagram_stability(entries: List[PDEntry], target_entry: PDEntry) -> Tuple[float, Dict, PhaseDiagram]:
    """
    分析材料在相图中的热力学稳定性

    Args:
        entries: Materials Project中的参考条目列表
        target_entry: 目标材料的条目

    Returns:
        Tuple[float, Dict, PhaseDiagram]: (energy_above_hull, decomposition, phase_diagram)
    """
    try:
        # 构建包含目标材料的完整条目列表
        all_entries = entries + [target_entry]

        # 构建相图
        phase_diagram = PhaseDiagram(all_entries)

        # 计算目标材料的分解反应和hull上方能量
        decomp_and_hull = phase_diagram.get_decomp_and_e_above_hull(target_entry)
        decomposition = decomp_and_hull[0]
        energy_above_hull = decomp_and_hull[1]

        return energy_above_hull, decomposition, phase_diagram

    except Exception as e:
        raise ValueError(f"相图稳定性分析失败: {str(e)}")


def formation_energy_to_total_energy(
    composition: Composition,
    formation_energy_per_atom: float,
    entries: List[PDEntry],
) -> float:
    """
    将 formation energy/atom 转回与 MP 相图同一参考基准下的总能量。

    这里使用同一 chemical system 相图中的元素参考相（el_refs）作为基准：
        E_total/atom = E_form/atom + sum_i x_i * mu_i(element_i)
    再乘原胞总原子数，得到可放进 PDEntry 的 total energy。
    """
    phase_diagram = PhaseDiagram(entries)
    el_ref_map = {str(el): ref.energy_per_atom for el, ref in phase_diagram.el_refs.items()}

    missing = [str(el) for el in composition.elements if str(el) not in el_ref_map]
    if missing:
        raise ValueError(f"缺少元素参考相: {missing}")

    total_atoms = composition.num_atoms
    elemental_reference_per_atom = 0.0
    for el, amount in composition.as_dict().items():
        elemental_reference_per_atom += (amount / total_atoms) * el_ref_map[el]

    target_energy_per_atom = float(formation_energy_per_atom) + elemental_reference_per_atom
    return target_energy_per_atom * total_atoms


@llm_tool(name="estimate_energy_above_hull_pymatgen",
          description="基于MatGL形成能预测和Materials Project相图估计材料的Energy Above Hull与稳定性")
async def estimate_energy_above_hull_pymatgen(
    cif_string: str,
    energy_threshold: float = 0.025,
    optimize_structure: bool = False
) -> str:
    """
    使用 MatGL 形成能预测 + Materials Project chemical-system phase diagram 估计稳定性。

    注意：该工具不直接检索目标材料自身的 e_above_hull，而是：
    1. 用 MatGL 预测目标结构 formation energy per atom
    2. 从 MP 获取同一化学系统的参考相
    3. 构建相图并估计目标结构的 hull distance
    """
    try:
        energy_threshold = validate_numeric_parameter(energy_threshold, "energy_threshold", 0.0, 1.0)
        structure = load_structure_from_cif_string(cif_string)
        composition = structure.composition
        formula = composition.reduced_formula
        elements = sorted(str(el) for el in composition.elements)

        result = "# 基于形成能的稳定性估计结果\n\n"
        result += f"**分子式**: {formula}\n"
        result += f"**化学系统**: {'-'.join(elements)}\n"
        result += f"**稳定性阈值**: {energy_threshold:.3f} eV/atom\n\n"

        formation_result = await call_matgl_formation_energy_prediction(cif_string, optimize_structure)
        if "error" in formation_result:
            return f"Error: 形成能预测失败: {formation_result['error']}"

        formation_energy_per_atom = formation_result.get("formation_energy_per_atom")
        if formation_energy_per_atom is None:
            return "Error: 形成能预测失败: 无法从 MatGL 结果中提取 formation energy per atom"

        result += "## 1. MatGL形成能预测\n\n"
        result += f"**formation energy**: {formation_energy_per_atom:.6f} eV/atom\n"
        result += f"**结构优化**: {'是' if optimize_structure else '否'}\n\n"

        mp_entries = get_materials_project_entries_by_elements(
            pymatgen_config.MP_API_KEY,
            elements,
        )
        result += "## 2. Materials Project参考相图\n\n"
        result += f"**参考条目数**: {len(mp_entries)}\n\n"

        total_energy = formation_energy_to_total_energy(
            composition=composition,
            formation_energy_per_atom=formation_energy_per_atom,
            entries=mp_entries,
        )
        target_entry = PDEntry(composition, total_energy)

        energy_above_hull, decomposition, _ = analyze_phase_diagram_stability(mp_entries, target_entry)
        is_stable = bool(energy_above_hull <= energy_threshold)

        payload = {
            "ok": True,
            "tool": "estimate_energy_above_hull_pymatgen",
            "formula": formula,
            "chemsys": "-".join(elements),
            "formation_energy_per_atom_eV": float(formation_energy_per_atom),
            "energy_above_hull_eV_per_atom": float(energy_above_hull),
            "energy_threshold_eV_per_atom": float(energy_threshold),
            "is_stable": is_stable,
            "decomposition": {
                entry.composition.reduced_formula: float(amount)
                for entry, amount in decomposition.items()
            },
            "reference_entries": len(mp_entries),
            "basis": "MatGL formation energy + MP GGA/GGA+U convex hull",
        }

        result += "## 3. 稳定性结论\n\n"
        result += f"**Energy Above Hull**: {energy_above_hull:.6f} eV/atom\n"
        result += f"**is_stable**: {'true' if is_stable else 'false'}\n\n"
        if decomposition:
            result += "### 可能分解产物\n\n"
            for entry, amount in decomposition.items():
                result += f"- {entry.composition.reduced_formula}: {float(amount):.6f}\n"
            result += "\n"

        result += f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        return result
    except Exception as e:
        return f"Error: 基于形成能的稳定性估计失败: {str(e)}"


@llm_tool(name="analyze_thermodynamic_stability_pymatgen",
          description="分析材料的热力学稳定性")
async def analyze_thermodynamic_stability_pymatgen(
    cif_string: str,
    energy_threshold: float= 0.025,
    optimize_structure: bool = False
) -> str:
    """
    分析材料的热力学稳定性

    功能包括：
    1. 从输入CIF中自动提取元素，动态构建化学系统
    2. 使用CHGNet模型预测材料的内部能量
    3. 从Materials Project数据库获取相应化学系统的参考数据
    4. 构建相图并计算energy above hull
    5. 根据阈值判断材料的热力学稳定性

    Args:
        cif_string: CIF格式的晶体结构内容字符串
        energy_threshold: 能量阈值 (eV/atom)，默认使用配置中的0.025
        optimize_structure: 是否在能量预测前优化结构，默认False

    Returns:
        包含热力学稳定性分析结果的格式化Markdown文本
    """
    try:
        # 使用配置中的默认阈值
        if energy_threshold is None:
            energy_threshold = pymatgen_config.DEFAULT_ENERGY_THRESHOLD
        else:
            energy_threshold = validate_numeric_parameter(energy_threshold, "energy_threshold", 0.0, 1.0)

        result = "# 热力学稳定性分析结果\n\n"

        # 步骤1: 从CIF中提取元素
        try:
            elements = extract_elements_from_cif(cif_string)
            result += f"## 1. 化学系统识别\n\n"
            result += f"**检测到的元素**: {', '.join(elements)}\n"
            result += f"**化学系统**: {'-'.join(elements)}\n\n"
        except Exception as e:
            return f"Error: 元素提取失败: {str(e)}"

        # 获取结构基本信息
        try:
            structure = load_structure_from_cif_string(cif_string)
            formula = structure.composition.reduced_formula
            num_atoms = len(structure)
            result += f"**分子式**: {formula}\n"
            result += f"**原子数量**: {num_atoms}\n\n"
        except Exception as e:
            return f"Error: 结构解析失败: {str(e)}"

        # 步骤2: 使用CHGNet预测能量
        result += f"## 2. 能量预测 (CHGNet)\n\n"
        try:
            energy_result = await call_matgl_energy_prediction(cif_string, optimize_structure)

            # 检查是否有错误
            if "error" in energy_result:
                return f"Error: 能量预测失败: {energy_result['error']}"

            energy_per_atom = energy_result["energy_per_atom"]
            predicted_num_atoms = energy_result["num_atoms"]

            # 检查是否成功获取到数值
            if energy_per_atom is None or predicted_num_atoms is None:
                if "error" in energy_result:
                    return f"Error: 能量预测失败:无法从MatGL结果中提取能量或原子数信息, {energy_result['error']}"
                return f"Error: 能量预测失败: 无法从MatGL结果中提取能量或原子数信息"

            result += f"**预测方法**: CHGNet\n"
            result += f"**结构优化**: {'是' if optimize_structure else '否'}\n"
            result += f"**单原子能量**: {energy_per_atom:.6f} eV/atom\n"
            result += f"**原子数量**: {predicted_num_atoms}\n\n"

        except Exception as e:
            return f"Error: 能量预测过程失败: {str(e)}"

        # 步骤3: 获取Materials Project参考数据
        result += f"## 3. Materials Project数据获取\n\n"
        try:
            mp_entries = get_materials_project_entries_by_elements(
                pymatgen_config.MP_API_KEY,
                elements
            )
            result += f"**获取条目数量**: {len(mp_entries)}\n"
            result += f"**化学系统**: {'-'.join(elements)}\n\n"

        except Exception as e:
            return f"Error: Materials Project数据获取失败: {str(e)}"

        # 步骤4: 创建目标材料条目
        try:
            target_entry = create_custom_pdentry(formula, energy_per_atom)
        except Exception as e:
            return f"Error: 目标材料条目创建失败: {str(e)}"

        # 步骤5: 相图分析
        result += f"## 4. 相图稳定性分析\n\n"
        try:
            energy_above_hull, decomposition, phase_diagram = analyze_phase_diagram_stability(
                mp_entries, target_entry
            )

            is_stable = energy_above_hull <= energy_threshold

            result += f"**Energy Above Hull**: {energy_above_hull:.6f} eV/atom\n"
            result += f"**稳定性阈值**: {energy_threshold:.3f} eV/atom\n"
            result += f"**热力学稳定性**: {'稳定' if is_stable else '不稳定'}\n\n"

            # 分解反应信息
            if energy_above_hull > 0:
                result += f"### 分解反应\n\n"
                result += f"目标材料将分解为以下相:\n\n"

                for phase, amount in decomposition.items():
                    phase_formula = phase.composition.reduced_formula
                    result += f"- **{phase_formula}**: {amount:.3f}\n"
                result += "\n"

        except Exception as e:
            return f"Error: 相图分析失败: {str(e)}"

        # 步骤6: 结果总结
        result += f"## 5. 分析总结\n\n"

        stability_status = "热力学稳定" if is_stable else "热力学不稳定"
        result += f"**稳定性结论**: {stability_status}\n\n"

        result += f"### 关键参数\n\n"
        result += f"- **化学系统**: {'-'.join(elements)}\n"
        result += f"- **分子式**: {formula}\n"
        result += f"- **预测能量**: {energy_per_atom:.6f} eV/atom\n"
        result += f"- **Hull上方能量**: {energy_above_hull:.6f} eV/atom\n"
        result += f"- **稳定性阈值**: {energy_threshold:.3f} eV/atom\n"
        result += f"- **参考条目数量**: {len(mp_entries)}\n"
        result += f"- **结构优化**: {'是' if optimize_structure else '否'}\n\n"

        if is_stable:
            result += f"### 稳定性评价\n\n"
            result += f"该材料的energy above hull ({energy_above_hull:.6f} eV/atom) "
            result += f"低于设定阈值 ({energy_threshold:.3f} eV/atom)，"
            result += f"表明该材料在热力学上是稳定的，有望在实验中合成。\n"
        else:
            result += f"### 不稳定性分析\n\n"
            result += f"该材料的energy above hull ({energy_above_hull:.6f} eV/atom) "
            result += f"高于设定阈值 ({energy_threshold:.3f} eV/atom)，"
            result += f"表明该材料在热力学上不稳定，倾向于分解为更稳定的相。\n"

        return result

    except Exception as e:
        return f"Error: 热力学稳定性分析失败: {str(e)}"


def _ternary_to_cartesian(a: float, b: float, c: float) -> Tuple[float, float]:
    """
    将三元组分转换为笛卡尔坐标（用于Gibbs三角图）

    Args:
        a, b, c: 三个组分的摩尔分数（应满足 a + b + c = 1）

    Returns:
        Tuple[float, float]: (x, y) 笛卡尔坐标
    """
    # 标准化确保总和为1
    total = a + b + c
    if total > 0:
        a, b, c = a/total, b/total, c/total

    # 转换为笛卡尔坐标
    # 三角形顶点坐标：A(0,0), B(1,0), C(0.5, sqrt(3)/2)
    x = 0.5 * (2*b + c)
    y = (np.sqrt(3)/2) * c

    return x, y


def _draw_ternary_axes(ax, elements: List[str]) -> None:
    """
    绘制Gibbs三角图的坐标轴和刻度

    Args:
        ax: matplotlib轴对象
        elements: 三个元素的列表 [A, B, C]
    """
    # 三角形顶点坐标
    triangle_x = [0, 1, 0.5, 0]
    triangle_y = [0, 0, np.sqrt(3)/2, 0]

    # 绘制三角形边框
    ax.plot(triangle_x, triangle_y, 'k-', linewidth=2)

    # 绘制网格线和刻度
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for tick in ticks:
        if tick == 0.0 or tick == 1.0:
            continue

        # 平行于底边的线（C组分恒定）
        x1, y1 = _ternary_to_cartesian(1-tick, tick, 0)
        x2, y2 = _ternary_to_cartesian(0, 1-tick, tick)
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)

        # 平行于左边的线（A组分恒定）
        x1, y1 = _ternary_to_cartesian(tick, 1-tick, 0)
        x2, y2 = _ternary_to_cartesian(tick, 0, 1-tick)
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)

        # 平行于右边的线（B组分恒定）
        x1, y1 = _ternary_to_cartesian(1-tick, 0, tick)
        x2, y2 = _ternary_to_cartesian(0, tick, 1-tick)
        ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=0.5)

    # 添加刻度标签
    for i, tick in enumerate(ticks):
        # 底边刻度（B组分）
        x, y = _ternary_to_cartesian(1-tick, tick, 0)
        ax.text(x, y-0.05, f'{tick:.1f}', ha='center', va='top', fontsize=10)

        # 左边刻度（A组分）
        x, y = _ternary_to_cartesian(tick, 0, 1-tick)
        ax.text(x-0.05, y, f'{1-tick:.1f}', ha='right', va='center', fontsize=10, rotation=60)

        # 右边刻度（C组分）
        x, y = _ternary_to_cartesian(0, 1-tick, tick)
        ax.text(x+0.05, y, f'{tick:.1f}', ha='left', va='center', fontsize=10, rotation=-60)

    # 添加元素标签
    ax.text(0, -0.1, elements[0], ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(1, -0.1, elements[1], ha='center', va='top', fontsize=14, fontweight='bold')
    ax.text(0.5, np.sqrt(3)/2 + 0.05, elements[2], ha='center', va='bottom', fontsize=14, fontweight='bold')

    # 添加百分比标签
    ax.text(-0.1, 0.05, f'{elements[0]} (%)', ha='center', va='bottom', fontsize=12, rotation=60)
    ax.text(1.1, 0.05, f'{elements[2]} (%)', ha='center', va='bottom', fontsize=12, rotation=-60)
    ax.text(0.5, -0.15, f'{elements[1]} (%)', ha='center', va='top', fontsize=12)


def _generate_phase_diagram_plot(phase_diagram: PhaseDiagram, target_entry: PDEntry,
                                decomposition: Dict, energy_above_hull: float) -> str:
    """
    生成相图的可视化图像并返回base64编码

    Args:
        phase_diagram: PyMatGen相图对象
        target_entry: 目标材料条目
        decomposition: 分解产物字典
        energy_above_hull: hull上方能量

    Returns:
        str: 相图的base64编码字符串
    """
    try:
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图形
        fig, ax = plt.subplots(figsize=(11, 6), dpi=150)

        # 获取所有稳定相的组成和能量
        stable_entries = phase_diagram.stable_entries
        unstable_entries = [e for e in phase_diagram.all_entries if e not in stable_entries]

        # 如果是二元或三元系统，绘制相图
        elements = list(phase_diagram.elements)
        n_elements = len(elements)

        if n_elements == 2:
            # 二元相图
            compositions = []
            energies = []

            for entry in stable_entries:
                comp = entry.composition
                x = comp.get_atomic_fraction(elements[1])  # 第二个元素的摩尔分数
                energy = phase_diagram.get_form_energy_per_atom(entry)
                compositions.append(x)
                energies.append(energy)

            # 排序以便绘制连线
            sorted_data = sorted(zip(compositions, energies))
            compositions, energies = zip(*sorted_data)

            # 绘制凸包线
            ax.plot(compositions, energies, 'b-', linewidth=2, label='Convex Hull')
            ax.scatter(compositions, energies, c='blue', s=50, zorder=5)

            # 标记目标材料
            target_comp = target_entry.composition
            target_x = target_comp.get_atomic_fraction(elements[1])
            target_energy = phase_diagram.get_form_energy_per_atom(target_entry)

            ax.scatter([target_x], [target_energy], c='red', s=100, marker='*',
                      label=f'Target: {target_entry.composition.reduced_formula}', zorder=10)

            # 如果材料不稳定，绘制到凸包的距离
            if energy_above_hull > 0:
                hull_energy = target_energy - energy_above_hull
                ax.plot([target_x, target_x], [hull_energy, target_energy],
                       'r--', linewidth=2, alpha=0.7)
                # ax.annotate(f'E_hull = {energy_above_hull:.3f} eV/atom',
                #            xy=(target_x, target_energy), xytext=(10, 10),
                #            textcoords='offset points', fontsize=10,
                #            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

            ax.set_xlabel(f'Atomic fraction of {elements[1]}')
            ax.set_ylabel('Formation Energy (eV/atom)')

        elif n_elements == 3:
            # 三元相图 - 使用Gibbs三角图
            _draw_ternary_axes(ax, [str(el) for el in elements])

            # 绘制稳定相
            stable_x = []
            stable_y = []
            stable_labels = []

            for entry in stable_entries:
                comp = entry.composition
                # 获取三个组分的摩尔分数
                fractions = [comp.get_atomic_fraction(el) for el in elements]
                x, y = _ternary_to_cartesian(*fractions)
                stable_x.append(x)
                stable_y.append(y)
                stable_labels.append(entry.composition.reduced_formula)

            # 绘制稳定相点
            ax.scatter(stable_x, stable_y, c='blue', s=60, alpha=0.8,
                      label='Stable phases', zorder=5, edgecolors='darkblue')

            # 标记目标材料
            target_comp = target_entry.composition
            target_fractions = [target_comp.get_atomic_fraction(el) for el in elements]
            target_x, target_y = _ternary_to_cartesian(*target_fractions)

            ax.scatter([target_x], [target_y], c='red', s=120, marker='*',
                      label=f'Target: {target_entry.composition.reduced_formula}',
                      zorder=10, edgecolors='darkred')

            # 添加化合物标签（只标记主要的稳定相）
            for i, (x, y, label) in enumerate(zip(stable_x, stable_y, stable_labels)):
                if i < 10:  # 只标记前10个以避免过于拥挤
                    ax.annotate(label, (x, y), xytext=(5, 5),
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

            # 标记目标材料
            # ax.annotate(f'{target_entry.composition.reduced_formula}\n(E_hull = {energy_above_hull:.3f} eV/atom)',
            #            (target_x, target_y), xytext=(10, 10),
            #            textcoords='offset points', fontsize=10,
            #            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

            # 设置坐标轴
            ax.set_xlim(-0.2, 1.2)
            ax.set_ylim(-0.2, np.sqrt(3)/2 + 0.2)
            ax.set_aspect('equal')
            ax.axis('off')  # 隐藏默认坐标轴

        else:
            # 多元系统 - 使用能量条形图
            formulas = []
            energies = []

            for entry in stable_entries[:20]:  # 只显示前20个
                formulas.append(entry.composition.reduced_formula)
                energies.append(phase_diagram.get_form_energy_per_atom(entry))

            # 添加目标材料
            formulas.append(target_entry.composition.reduced_formula)
            energies.append(phase_diagram.get_form_energy_per_atom(target_entry))

            bars = ax.bar(range(len(formulas)), energies, alpha=0.7)

            # 高亮目标材料
            bars[-1].set_color('red')
            bars[-1].set_label(f'Target: {target_entry.composition.reduced_formula}')

            ax.set_xlabel('Compounds')
            ax.set_ylabel('Formation Energy (eV/atom)')
            ax.set_xticks(range(len(formulas)))
            ax.set_xticklabels(formulas, rotation=45, ha='right')

        # 添加标题和图例
        system_name = '-'.join([str(el) for el in elements])
        if n_elements == 3:
            ax.set_title(f'Ternary Phase Diagram: {system_name} System\n'
                        f'Target: {target_entry.composition.reduced_formula} '
                        f'(E_hull = {energy_above_hull:.3f} eV/atom)',
                        fontsize=14, pad=20)
        else:
            ax.set_title(f'Phase Diagram: {system_name} System\n'
                        f'Target: {target_entry.composition.reduced_formula} '
                        f'(E_hull = {energy_above_hull:.3f} eV/atom)')

        if n_elements != 3:  # 三元图不需要网格和常规图例
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))

        # 调整布局
        plt.tight_layout()

        # 保存到内存缓冲区
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)

        # 转换为base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # 清理
        plt.close(fig)
        buffer.close()

        return image_base64

    except Exception as e:
        # 如果绘图失败，返回空字符串
        plt.close('all')  # 确保清理所有图形
        return ""


@llm_tool(name="analyze_thermodynamic_stability_with_phase_diagram_pymatgen",
          description="分析材料的热力学稳定性并生成相图的base64编码")
async def analyze_thermodynamic_stability_with_phase_diagram_pymatgen(
    cif_string: str,
    energy_threshold: float = 0.025,
    optimize_structure: bool = False
) -> str:
    """
    分析材料的热力学稳定性并生成相图的可视化

    功能包括：
    1. 从输入CIF中自动提取元素，动态构建化学系统
    2. 使用CHGNet模型预测材料的内部能量
    3. 从Materials Project数据库获取相应化学系统的参考数据
    4. 构建相图并计算energy above hull
    5. 根据阈值判断材料的热力学稳定性
    6. 生成相图的可视化图像并转换为base64编码

    Args:
        cif_string: CIF格式的晶体结构内容字符串
        energy_threshold: 能量阈值 (eV/atom)，默认使用配置中的0.025
        optimize_structure: 是否在能量预测前优化结构，默认False

    Returns:
        包含热力学稳定性分析结果和相图base64编码的格式化Markdown文本
    """
    try:
        # 使用配置中的默认阈值
        if energy_threshold is None:
            energy_threshold = pymatgen_config.DEFAULT_ENERGY_THRESHOLD
        else:
            energy_threshold = validate_numeric_parameter(energy_threshold, "energy_threshold", 0.0, 1.0)

        result = "# 热力学稳定性分析结果（含相图）\n\n"

        # 步骤1: 从CIF中提取元素
        try:
            elements = extract_elements_from_cif(cif_string)
            result += f"## 1. 化学系统识别\n\n"
            result += f"**检测到的元素**: {', '.join(elements)}\n"
            result += f"**化学系统**: {'-'.join(elements)}\n\n"
        except Exception as e:
            return f"Error: 元素提取失败: {str(e)}"

        # 获取结构基本信息
        try:
            structure = load_structure_from_cif_string(cif_string)
            formula = structure.composition.reduced_formula
            num_atoms = len(structure)
            result += f"**分子式**: {formula}\n"
            result += f"**原子数量**: {num_atoms}\n\n"
        except Exception as e:
            return f"Error: 结构解析失败: {str(e)}"

        # 步骤2: 使用CHGNet预测能量
        result += f"## 2. 能量预测 (CHGNet)\n\n"
        try:
            energy_result = await call_matgl_energy_prediction(cif_string, optimize_structure)

            # 检查是否有错误
            if "error" in energy_result:
                return f"Error: 能量预测失败: {energy_result['error']}"

            energy_per_atom = energy_result["energy_per_atom"]
            predicted_num_atoms = energy_result["num_atoms"]

            # 检查是否成功获取到数值
            if energy_per_atom is None or predicted_num_atoms is None:
                if "error" in energy_result:
                    return f"Error: 能量预测失败:无法从MatGL结果中提取能量或原子数信息, {energy_result['error']}"
                return f"Error: 能量预测失败: 无法从MatGL结果中提取能量或原子数信息"

            result += f"**预测方法**: CHGNet\n"
            result += f"**结构优化**: {'是' if optimize_structure else '否'}\n"
            result += f"**单原子能量**: {energy_per_atom:.6f} eV/atom\n"
            result += f"**原子数量**: {predicted_num_atoms}\n\n"

        except Exception as e:
            return f"Error: 能量预测过程失败: {str(e)}"

        # 步骤3: 获取Materials Project参考数据
        result += f"## 3. Materials Project数据获取\n\n"
        try:
            mp_entries = get_materials_project_entries_by_elements(
                pymatgen_config.MP_API_KEY,
                elements
            )
            result += f"**获取条目数量**: {len(mp_entries)}\n"
            result += f"**化学系统**: {'-'.join(elements)}\n\n"

        except Exception as e:
            return f"Error: Materials Project数据获取失败: {str(e)}"

        # 步骤4: 创建目标材料条目
        try:
            target_entry = create_custom_pdentry(formula, energy_per_atom)
        except Exception as e:
            return f"Error: 目标材料条目创建失败: {str(e)}"

        # 步骤5: 相图分析
        result += f"## 4. 相图稳定性分析\n\n"
        try:
            energy_above_hull, decomposition, phase_diagram = analyze_phase_diagram_stability(
                mp_entries, target_entry
            )

            is_stable = energy_above_hull <= energy_threshold

            result += f"**Energy Above Hull**: {energy_above_hull:.6f} eV/atom\n"
            result += f"**稳定性阈值**: {energy_threshold:.3f} eV/atom\n"
            result += f"**热力学稳定性**: {'稳定' if is_stable else '不稳定'}\n\n"

            # 分解反应信息
            if energy_above_hull > 0:
                result += f"### 分解反应\n\n"
                result += f"目标材料将分解为以下相:\n\n"

                for phase, amount in decomposition.items():
                    phase_formula = phase.composition.reduced_formula
                    result += f"- **{phase_formula}**: {amount:.3f}\n"
                result += "\n"

        except Exception as e:
            return f"Error: 相图分析失败: {str(e)}"

        # 步骤6: 生成相图可视化
        result += f"## 5. 相图可视化\n\n"
        try:
            phase_diagram_base64 = _generate_phase_diagram_plot(
                phase_diagram, target_entry, decomposition, energy_above_hull
            )

            if phase_diagram_base64:
                result += f"### 相图图像\n\n"
                result += f"![Phase Diagram](data:image/png;base64,{phase_diagram_base64})\n\n"
                result += f"**图像说明**:\n"
                result += f"- 蓝色线/点表示热力学稳定相（凸包）\n"
                result += f"- 红色星号标记目标材料位置\n"
                result += f"- 如果材料不稳定，红色虚线显示到凸包的距离\n"
                result += f"- Energy above hull值显示在目标材料附近\n\n"

                result += f"**Base64编码**: \n```\n{phase_diagram_base64}\n```\n\n"
            else:
                result += f"**注意**: 相图生成失败，可能是由于系统复杂度过高或数据不足\n\n"

        except Exception as e:
            result += f"**相图生成警告**: {str(e)}\n\n"

        # 步骤7: 结果总结
        result += f"## 6. 分析总结\n\n"

        stability_status = "热力学稳定" if is_stable else "热力学不稳定"
        result += f"**稳定性结论**: {stability_status}\n\n"

        result += f"### 关键参数\n\n"
        result += f"- **化学系统**: {'-'.join(elements)}\n"
        result += f"- **分子式**: {formula}\n"
        result += f"- **预测能量**: {energy_per_atom:.6f} eV/atom\n"
        result += f"- **Hull上方能量**: {energy_above_hull:.6f} eV/atom\n"
        result += f"- **稳定性阈值**: {energy_threshold:.3f} eV/atom\n"
        result += f"- **参考条目数量**: {len(mp_entries)}\n"
        result += f"- **结构优化**: {'是' if optimize_structure else '否'}\n\n"

        if is_stable:
            result += f"### 稳定性评价\n\n"
            result += f"该材料的energy above hull ({energy_above_hull:.6f} eV/atom) "
            result += f"低于设定阈值 ({energy_threshold:.3f} eV/atom)，"
            result += f"表明该材料在热力学上是稳定的，有望在实验中合成。\n\n"
            result += f"从相图中可以看出，该材料位于或接近凸包线上，表明其为热力学稳定相。\n"
        else:
            result += f"### 不稳定性分析\n\n"
            result += f"该材料的energy above hull ({energy_above_hull:.6f} eV/atom) "
            result += f"高于设定阈值 ({energy_threshold:.3f} eV/atom)，"
            result += f"表明该材料在热力学上不稳定，倾向于分解为更稳定的相。\n\n"
            result += f"从相图中可以看出，该材料位于凸包线上方，红色虚线显示了其到稳定相的能量距离。\n"

        return result

    except Exception as e:
        return f"Error: 热力学稳定性分析失败: {str(e)}"


def _check_valence_core(formula: str) -> tuple:
    """
    检测化学式的价态合理性核心函数

    Args:
        formula (str): 化学式字符串

    Returns:
        tuple: (is_valid, details)
            - is_valid (bool): 价态是否合理
            - details: 氧化态详情或错误信息
    """
    if pd.isna(formula) or not isinstance(formula, str):
        return False, "化学式为空或格式错误"

    try:
        comp = Composition(formula)
        # 获取可能的氧化态组合
        oxidation_guesses = comp.oxi_state_guesses()

        if not oxidation_guesses:
            return False, "无合理的氧化态组合"

        # 检查所有可能性中是否存在电荷平衡的组合
        valid = False
        valid_guess = None
        for guess in oxidation_guesses:
            total_charge = sum(comp[element] * guess[element.symbol] for element in comp.elements)
            if abs(total_charge) < 1e-6:  # 考虑浮点数精度
                valid = True
                valid_guess = guess
                break

        if valid:
            return True, valid_guess
        else:
            return False, "无电荷平衡的组合"

    except Exception as e:
        return False, f"解析化学式失败: {str(e)}"


@llm_tool(name="check_chemical_formula_valence_pymatgen",
          description="使用PyMatGen检验化学式的价态合理性和氧化态信息")
async def check_chemical_formula_valence_pymatgen(
    formula:str
) -> str:
    """
    使用PyMatGen库检验化学式的价态合理性，并返回详细的氧化态信息

    Args:
        formula: 要检验的化学式字符串 (例如: "Fe2O3", "LiFePO4", "CaCO3")

    Returns:
        包含价态验证结果和氧化态详细信息的格式化Markdown文本
    """
    try:
        # 使用统一的化学式验证函数进行输入验证和清理
        try:
            cleaned_formula = validate_chemical_formula(formula)
        except ValueError as e:
            return f"Error: {str(e)}"

        # 执行价态检验
        is_valid, details = _check_valence_core(cleaned_formula)

        # 构建结果输出
        result = f"# 化学式价态验证结果\n\n"
        result += f"**化学式**: {cleaned_formula}\n\n"

        if is_valid:
            result += f"**验证结果**:  价态合理\n\n"
            result += f"## 氧化态信息\n\n"

            # 格式化氧化态信息
            if isinstance(details, dict):
                result += "| 元素 | 氧化态 |\n"
                result += "|------|--------|\n"
                for element_symbol, oxidation_state in details.items():
                    result += f"| {element_symbol} | {oxidation_state:+} |\n"
            else:
                result += f"氧化态详情: {details}\n"

        else:
            result += f"**验证结果**:  价态不合理\n\n"
            result += f"**错误信息**: {details}\n\n"

            # 尝试提供更多信息
            try:
                comp = Composition(cleaned_formula)
                result += f"## 化学式组成信息\n\n"
                result += f"**分子式**: {comp.formula}\n"
                result += f"**简化分子式**: {comp.reduced_formula}\n"
                result += f"**元素列表**: {', '.join([str(el) for el in comp.elements])}\n"
            except:
                pass

        return result

    except Exception as e:
        return f"Error: 处理化学式时发生意外错误: {str(e)}"


# ============================================================================
# 原子几何检查模块
# ============================================================================

def _get_element_symbol_safe(obj) -> str:
    """
    从pymatgen对象中安全获取元素符号
    支持Site和PeriodicNeighbor等不同类型的对象

    Args:
        obj: pymatgen对象（Site、PeriodicNeighbor等）

    Returns:
        str: 元素符号
    """
    try:
        # 方法1: 尝试 obj.specie.symbol
        if hasattr(obj, 'specie') and hasattr(obj.specie, 'symbol'):
            return obj.specie.symbol
    except AttributeError:
        pass

    try:
        # 方法2: 尝试 obj.specie (直接返回元素符号)
        if hasattr(obj, 'specie'):
            specie = obj.specie
            if isinstance(specie, str):
                return specie
    except AttributeError:
        pass

    try:
        # 方法3: 尝试 obj.species_string
        if hasattr(obj, 'species_string'):
            return obj.species_string
    except AttributeError:
        pass

    try:
        # 方法4: 尝试从 obj.species 中提取
        if hasattr(obj, 'species'):
            species = obj.species
            if hasattr(species, 'symbol'):
                return species.symbol
            elif isinstance(species, str):
                # 从 "S1" 这样的字符串中提取 "S"
                match = re.match(r'([A-Z][a-z]?)', species)
                if match:
                    return match.group(1)
    except AttributeError:
        pass

    # 如果所有方法都失败，返回未知
    return "Unknown"


def _get_atomic_radius(element_symbol: str) -> float:
    """
    获取元素的原子半径（埃）

    Args:
        element_symbol: 元素符号

    Returns:
        float: 原子半径（埃）
    """
    try:
        element = Element(element_symbol)
        # 优先使用共价半径，如果没有则使用原子半径
        if hasattr(element, 'atomic_radius') and element.atomic_radius:
            return element.atomic_radius
        elif hasattr(element, 'van_der_waals_radius') and element.van_der_waals_radius:
            return element.van_der_waals_radius * 0.8  # 估算共价半径
        else:
            # 使用默认值
            default_radii = {
                'H': 0.31, 'He': 0.28, 'Li': 1.28, 'Be': 0.96, 'B': 0.84,
                'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57, 'Ne': 0.58,
                'Na': 1.66, 'Mg': 1.41, 'Al': 1.21, 'Si': 1.11, 'P': 1.07,
                'S': 1.05, 'Cl': 1.02, 'Ar': 1.06, 'K': 2.03, 'Ca': 1.76,
                'Sc': 1.70, 'Ti': 1.60, 'V': 1.53, 'Cr': 1.39, 'Mn': 1.39,
                'Fe': 1.32, 'Co': 1.26, 'Ni': 1.24, 'Cu': 1.32, 'Zn': 1.22
            }
            return default_radii.get(element_symbol, 1.5)  # 默认1.5埃
    except:
        return 1.5  # 默认值


def _check_atomic_overlap(structure: Structure, min_distance_factor: float, max_neighbors_distance: float) -> tuple:
    """
    检查原子重叠

    Args:
        structure: pymatgen Structure对象
        min_distance_factor: 最小距离因子
        max_neighbors_distance: 搜索近邻的最大距离

    Returns:
        tuple: (is_valid, details_dict)
    """
    try:
        overlap_details = []
        total_pairs_checked = 0
        min_distance_found = float('inf')
        min_distance_pair = None

        # 获取所有原子的近邻
        for i, site in enumerate(structure.sites):
            neighbors = structure.get_neighbors(site, max_neighbors_distance)

            element1 = _get_element_symbol_safe(site)
            radius1 = _get_atomic_radius(element1)

            for neighbor in neighbors:
                total_pairs_checked += 1
                distance = neighbor.nn_distance
                element2 = _get_element_symbol_safe(neighbor)
                radius2 = _get_atomic_radius(element2)

                # 计算最小允许距离
                min_distance = (radius1 + radius2) * min_distance_factor

                # 记录最短距离
                if distance < min_distance_found:
                    min_distance_found = distance
                    min_distance_pair = (element1, element2, distance, min_distance)

                if distance < min_distance:
                    overlap_details.append({
                        'pair': f"{element1}-{element2}",
                        'actual_distance': distance,
                        'min_allowed_distance': min_distance,
                        'overlap_severity': (min_distance - distance) / min_distance * 100
                    })

        is_valid = len(overlap_details) == 0

        details = {
            'is_valid': is_valid,
            'total_pairs_checked': total_pairs_checked,
            'overlapping_pairs': len(overlap_details),
            'overlap_details': overlap_details,
            'shortest_distance': {
                'pair': f"{min_distance_pair[0]}-{min_distance_pair[1]}" if min_distance_pair else "N/A",
                'distance': min_distance_found if min_distance_found != float('inf') else 0,
                'min_allowed': min_distance_pair[3] if min_distance_pair else 0
            }
        }

        return is_valid, details

    except Exception as e:
        raise ValueError(f"原子重叠检查失败: {str(e)}")


def _check_bond_lengths(structure: Structure, bond_tolerance: float) -> tuple:
    """
    检查键长合理性

    Args:
        structure: pymatgen Structure对象
        bond_tolerance: 键长容差

    Returns:
        tuple: (is_valid, details_dict)
    """
    try:
        crystal_nn = CrystalNN()
        bond_details = []
        total_bonds = 0
        unreasonable_bonds = 0

        # 使用CrystalNN识别化学键
        for i, site in enumerate(structure.sites):
            try:
                # 获取该原子的配位环境
                nn_info = crystal_nn.get_nn_info(structure, i)

                element1 = _get_element_symbol_safe(site)

                for neighbor_info in nn_info:
                    total_bonds += 1
                    neighbor_site = neighbor_info['site']
                    # 使用正确的距离计算方法
                    distance = structure[i].distance(neighbor_site)

                    element2 = _get_element_symbol_safe(neighbor_site)

                    # 估算期望键长（基于原子半径）
                    radius1 = _get_atomic_radius(element1)
                    radius2 = _get_atomic_radius(element2)
                    expected_bond_length = radius1 + radius2

                    # 检查键长是否在合理范围内
                    ratio = distance / expected_bond_length
                    min_ratio = 1.0 - bond_tolerance
                    max_ratio = 1.0 + bond_tolerance

                    deviation_percent = (ratio - 1.0) * 100
                    is_reasonable = min_ratio <= ratio <= max_ratio

                    if not is_reasonable:
                        unreasonable_bonds += 1

                    bond_details.append({
                        'bond_type': f"{element1}-{element2}",
                        'actual_length': distance,
                        'expected_length': expected_bond_length,
                        'deviation_percent': deviation_percent,
                        'is_reasonable': is_reasonable
                    })

            except Exception as e:
                # CrystalNN可能对某些结构失败，这种情况下跳过
                continue

        is_valid = unreasonable_bonds == 0

        details = {
            'is_valid': is_valid,
            'total_bonds': total_bonds,
            'unreasonable_bonds': unreasonable_bonds,
            'bond_details': bond_details
        }

        return is_valid, details

    except Exception as e:
        raise ValueError(f"键长检查失败: {str(e)}")


@llm_tool(name="check_structure_atomic_geometry_pymatgen",
          description="使用PyMatGen检查晶体结构的原子几何合理性，包括原子重叠和键长分析")
async def check_structure_atomic_geometry_pymatgen(
    cif_string:str,
    min_distance_factor: float=0.5,
    bond_tolerance:float = 0.3,
    max_neighbors_distance: float = 5.0
) -> str:
    """
    使用PyMatGen库检查晶体结构的原子几何合理性

    检查内容包括：
    1. 原子重叠检查：确保原子间距离不小于原子半径和的指定倍数
    2. 键长合理性检查：使用CrystalNN算法识别化学键并验证键长是否在合理范围内

    Args:
        cif_string: CIF格式的晶体结构内容字符串
        min_distance_factor: 最小距离因子，相对于原子半径和 (默认: 0.5)
        bond_tolerance: 键长容差比例 (默认: 0.3，即±30%)
        max_neighbors_distance: 搜索近邻的最大距离 (埃) (默认: 5.0)

    Returns:
        包含原子几何检查结果的格式化Markdown文本
    """
    try:
        # 参数验证
        min_distance_factor = validate_numeric_parameter(min_distance_factor, "min_distance_factor", 0.1, 1.0)
        bond_tolerance = validate_numeric_parameter(bond_tolerance, "bond_tolerance", 0.05, 1.0)
        max_neighbors_distance = validate_numeric_parameter(max_neighbors_distance, "max_neighbors_distance", 2.0, 20.0)

        # 解析CIF结构
        try:
            structure = load_structure_from_cif_string(cif_string)
        except Exception as e:
            return f"Error: CIF结构解析失败: {str(e)}"

        # 获取结构基本信息
        reduced_formula = structure.composition.reduced_formula
        symmetry = structure.get_space_group_info()
        num_sites = len(structure.sites)
        volume = structure.volume
        density = structure.density

        # 执行原子重叠检查
        try:
            overlap_valid, overlap_details = _check_atomic_overlap(structure, min_distance_factor, max_neighbors_distance)
        except Exception as e:
            return f"Error: 原子重叠检查失败: {str(e)}"

        # 执行键长合理性检查
        try:
            bond_valid, bond_details = _check_bond_lengths(structure, bond_tolerance)
        except Exception as e:
            return f"Error: 键长合理性检查失败: {str(e)}"

        # 综合检查结果
        overall_valid = overlap_valid and bond_valid

        # 构建结果输出
        result = "# 晶体结构原子几何检查结果\n\n"

        # 总体状态
        status = "通过" if overall_valid else "未通过"
        result += f"**检查状态**: {status}\n\n"

        # 结构基本信息
        result += "## 结构基本信息\n\n"
        result += f"- **分子式**: {reduced_formula}\n"
        result += f"- **空间群**: {symmetry[0]} (#{symmetry[1]})\n"
        result += f"- **原子数量**: {num_sites}\n"
        result += f"- **体积**: {volume:.2f} Å³\n"
        result += f"- **密度**: {density:.2f} g/cm³\n\n"

        # 原子重叠检查结果
        result += "## 原子重叠检查\n\n"
        overlap_status = "无原子重叠" if overlap_valid else "存在原子重叠"
        result += f"**结果**: {overlap_status}\n\n"
        result += f"- 检查了 {overlap_details['total_pairs_checked']} 对原子间距离\n"
        result += f"- 最小距离因子: {min_distance_factor}\n"

        if overlap_details['shortest_distance']['pair'] != "N/A":
            result += f"- 最短原子间距: {overlap_details['shortest_distance']['pair']} "
            result += f"{overlap_details['shortest_distance']['distance']:.3f} Å "
            result += f"(最小允许: {overlap_details['shortest_distance']['min_allowed']:.3f} Å)\n"

        if not overlap_valid:
            result += f"- 发现 {overlap_details['overlapping_pairs']} 对重叠原子\n"
            result += "\n**重叠详情**:\n\n"
            result += "| 原子对 | 实际距离 | 最小允许距离 | 重叠程度 |\n"
            result += "|--------|----------|--------------|----------|\n"
            for detail in overlap_details['overlap_details'][:5]:  # 最多显示5个
                result += f"| {detail['pair']} | {detail['actual_distance']:.3f} Å | "
                result += f"{detail['min_allowed_distance']:.3f} Å | {detail['overlap_severity']:.1f}% |\n"

        result += "\n"

        # 键长合理性检查结果
        result += "## 键长合理性检查\n\n"
        bond_status = "键长合理" if bond_valid else "键长不合理"
        result += f"**结果**: {bond_status}\n\n"

        if bond_details['total_bonds'] > 0:
            result += f"- 识别化学键数量: {bond_details['total_bonds']}\n"
            result += f"- 键长容差: ±{bond_tolerance*100:.0f}%\n"

            if not bond_valid:
                result += f"- 不合理键长数量: {bond_details['unreasonable_bonds']}\n"

            result += "\n**键长分析详情**:\n\n"
            result += "| 键类型 | 实际长度 | 期望长度 | 偏差比例 | 状态 |\n"
            result += "|--------|----------|----------|----------|------|\n"

            # 显示前10个键的详情
            for detail in bond_details['bond_details'][:10]:
                status_text = "合理" if detail['is_reasonable'] else "不合理"
                result += f"| {detail['bond_type']} | {detail['actual_length']:.3f} Å | "
                result += f"{detail['expected_length']:.3f} Å | {detail['deviation_percent']:+.1f}% | {status_text} |\n"

            if len(bond_details['bond_details']) > 10:
                result += f"\n*注: 仅显示前10个键的详情，总共识别 {bond_details['total_bonds']} 个键*\n"
        else:
            result += "- 未能识别到化学键（可能是CrystalNN算法限制）\n"
            result += "- 建议检查结构的合理性或尝试其他分析方法\n"

        result += "\n"

        # 检查参数
        result += "## 检查参数\n\n"
        result += f"- 最小距离因子: {min_distance_factor}\n"
        result += f"- 键长容差: ±{bond_tolerance*100:.0f}%\n"
        result += f"- 近邻搜索距离: {max_neighbors_distance} Å\n"

        return result

    except Exception as e:
        return f"Error: 晶体结构原子几何检查失败: {str(e)}"


# ============================================================================
# XRD衍射图谱模拟模块
# ============================================================================

def _format_miller_indices(hkl_list: List[Tuple[int, ...]]) -> str:
    """
    格式化Miller指数列表为字符串表示

    Args:
        hkl_list: Miller指数元组列表

    Returns:
        str: 格式化的Miller指数字符串
    """
    if not hkl_list:
        return "N/A"

    formatted_indices = []
    for hkl in hkl_list:
        # 处理负数指数的显示
        hkl_str = "".join([str(h) if h >= 0 else f"̄{abs(h)}" for h in hkl])
        formatted_indices.append(f"({hkl_str})")

    return ", ".join(formatted_indices)


def _get_peak_statistics(intensities: List[float], two_thetas: List[float]) -> Dict[str, Any]:
    """
    计算衍射峰的统计信息

    Args:
        intensities: 强度列表
        two_thetas: 2θ角度列表

    Returns:
        Dict: 包含统计信息的字典
    """
    if not intensities:
        return {
            'total_peaks': 0,
            'max_intensity': 0,
            'min_intensity': 0,
            'avg_intensity': 0,
            'strong_peaks_count': 0,
            'angle_range': (0, 0)
        }

    intensities_array = np.array(intensities)
    two_thetas_array = np.array(two_thetas)

    # 定义强峰阈值（相对强度>50%）
    strong_peak_threshold = np.max(intensities_array) * 0.5
    strong_peaks_count = np.sum(intensities_array >= strong_peak_threshold)

    return {
        'total_peaks': len(intensities),
        'max_intensity': float(np.max(intensities_array)),
        'min_intensity': float(np.min(intensities_array)),
        'avg_intensity': float(np.mean(intensities_array)),
        'strong_peaks_count': int(strong_peaks_count),
        'angle_range': (float(np.min(two_thetas_array)), float(np.max(two_thetas_array)))
    }


@llm_tool(name="simulate_xrd_pattern_pymatgen",
          description="使用PyMatGen模拟晶体结构的XRD衍射图谱，提供完整的衍射峰信息")
async def simulate_xrd_pattern_pymatgen(
    cif_string: str,
    wavelength: str = "CuKa",
    two_theta_range: Tuple[float, float] = (10.0, 80.0),
    min_intensity_threshold: float = 0.5
) -> str:
    """
    使用PyMatGen库模拟晶体结构的XRD衍射图谱

    功能包括：
    1. 基于晶体结构计算理论XRD衍射图谱
    2. 考虑原子散射因子和Lorentz偏振因子
    3. 提供完整的衍射峰信息（角度、强度、d间距、Miller指数）
    4. 支持多种常用X射线源
    5. 可自定义角度范围和强度阈值

    Args:
        cif_string: CIF格式的晶体结构内容字符串
        wavelength: X射线波长类型，支持CuKa, MoKa, CrKa, FeKa, CoKa, AgKa等 (默认: "CuKa")
        two_theta_range: 2θ角度扫描范围，格式为(min_angle, max_angle) (默认: (10.0, 80.0))
        min_intensity_threshold: 最小强度阈值，相对于最强峰的百分比 (默认: 0.5)

    Returns:
        包含XRD衍射图谱完整信息的格式化Markdown文本
    """
    try:
        # 参数验证
        if not isinstance(two_theta_range, (tuple, list)) or len(two_theta_range) != 2:
            return "Error: two_theta_range必须是包含两个数值的元组或列表"

        min_angle, max_angle = two_theta_range
        min_angle = validate_numeric_parameter(min_angle, "min_angle", 0.0, 180.0)
        max_angle = validate_numeric_parameter(max_angle, "max_angle", 0.0, 180.0)

        if min_angle >= max_angle:
            return "Error: 最小角度必须小于最大角度"

        min_intensity_threshold = validate_numeric_parameter(min_intensity_threshold, "min_intensity_threshold", 0.0, 100.0)

        # 验证波长参数
        available_wavelengths = [
            "CuKa", "CuKa1", "CuKa2", "CuKb1",
            "MoKa", "MoKa1", "MoKa2", "MoKb1",
            "CrKa", "CrKa1", "CrKa2", "CrKb1",
            "FeKa", "FeKa1", "FeKa2", "FeKb1",
            "CoKa", "CoKa1", "CoKa2", "CoKb1",
            "AgKa", "AgKa1", "AgKa2", "AgKb1"
        ]

        if wavelength not in available_wavelengths:
            return f"Error: 不支持的波长类型 '{wavelength}'。支持的类型: {', '.join(available_wavelengths)}"

        # 解析CIF结构
        try:
            structure = load_structure_from_cif_string(cif_string)
        except Exception as e:
            return f"Error: CIF结构解析失败: {str(e)}"

        # 获取结构基本信息
        reduced_formula = structure.composition.reduced_formula
        symmetry = structure.get_space_group_info()
        num_sites = len(structure.sites)
        volume = structure.volume
        density = structure.density
        lattice_params = structure.lattice.parameters

        # 创建XRD计算器
        try:
            xrd_calculator = XRDCalculator(wavelength=wavelength, symprec=0.01)
        except Exception as e:
            return f"Error: XRD计算器初始化失败: {str(e)}"

        # 计算XRD图谱
        try:
            xrd_pattern = xrd_calculator.get_pattern(
                structure=structure,
                scaled=True,  # 归一化强度
                two_theta_range=(min_angle, max_angle)
            )
        except Exception as e:
            return f"Error: XRD图谱计算失败: {str(e)}"

        # 提取衍射峰数据
        two_thetas = xrd_pattern.x
        intensities = xrd_pattern.y
        d_spacings = xrd_pattern.d_hkls
        hkl_data = xrd_pattern.hkls

        # 应用强度阈值过滤
        filtered_peaks = []
        max_intensity = float(max(intensities)) if len(intensities) > 0 else 0.0
        threshold_value = max_intensity * min_intensity_threshold / 100.0

        for i, (two_theta, intensity, d_spacing, hkl_info) in enumerate(zip(two_thetas, intensities, d_spacings, hkl_data)):
            if float(intensity) >= threshold_value:
                # 提取Miller指数和多重度信息
                miller_indices = []
                total_multiplicity = 0

                for hkl_entry in hkl_info:
                    hkl = hkl_entry.get('hkl', ())
                    multiplicity = hkl_entry.get('multiplicity', 1)
                    miller_indices.append(hkl)
                    total_multiplicity += multiplicity

                filtered_peaks.append({
                    'two_theta': two_theta,
                    'intensity': intensity,
                    'd_spacing': d_spacing,
                    'miller_indices': miller_indices,
                    'multiplicity': total_multiplicity,
                    'relative_intensity': (intensity / max_intensity) * 100 if max_intensity > 0 else 0
                })

        # 按强度排序（从高到低）
        filtered_peaks.sort(key=lambda x: x['intensity'], reverse=True)

        # 计算统计信息
        peak_stats = _get_peak_statistics([p['intensity'] for p in filtered_peaks],
                                        [p['two_theta'] for p in filtered_peaks])

        # 构建结果输出
        result = "# XRD衍射图谱模拟结果\n\n"

        # 基本信息（简化）
        result += "## 结构信息\n\n"
        result += f"- **分子式**: {reduced_formula}\n"
        result += f"- **空间群**: {symmetry[0]} (#{symmetry[1]})\n"
        result += f"- **晶格参数**: a={lattice_params[0]:.3f} Å, b={lattice_params[1]:.3f} Å, c={lattice_params[2]:.3f} Å\n"
        result += f"- **X射线源**: {wavelength}\n\n"

        # 图谱统计（简化）
        result += "## 衍射峰统计\n\n"
        result += f"- **检测峰数**: {peak_stats['total_peaks']}\n"
        result += f"- **强峰数** (>50%): {peak_stats['strong_peaks_count']}\n"
        result += f"- **最强峰强度**: {peak_stats['max_intensity']:.1f}\n\n"

        # 衍射峰详细信息
        if filtered_peaks:
            result += "## 衍射峰详细信息\n\n"
            result += "| 序号 | 2θ (°) | 相对强度 (%) | d间距 (Å) | Miller指数 | 多重度 |\n"
            result += "|------|--------|--------------|-----------|------------|--------|\n"

            for i, peak in enumerate(filtered_peaks, 1):
                miller_str = _format_miller_indices(peak['miller_indices'])
                result += f"| {i:2d} | {peak['two_theta']:6.2f} | {peak['relative_intensity']:8.1f} | "
                result += f"{peak['d_spacing']:7.3f} | {miller_str:10s} | {peak['multiplicity']:6d} |\n"

            result += "\n"

            # 主要衍射峰分析（前5个最强峰）
            result += "## 主要衍射峰分析\n\n"
            main_peaks = filtered_peaks[:5]

            for i, peak in enumerate(main_peaks, 1):
                result += f"### 峰 {i}: 2θ = {peak['two_theta']:.2f}°\n\n"
                result += f"- **强度**: {peak['intensity']:.1f} (相对强度: {peak['relative_intensity']:.1f}%)\n"
                result += f"- **d间距**: {peak['d_spacing']:.3f} Å\n"
                result += f"- **Miller指数**: {_format_miller_indices(peak['miller_indices'])}\n"
                result += f"- **多重度**: {peak['multiplicity']}\n\n"
        else:
            result += "## 衍射峰信息\n\n"
            result += f"在设定的强度阈值 ({min_intensity_threshold}%) 下未检测到衍射峰。\n"
            result += "建议降低强度阈值或检查结构的合理性。\n\n"

        # 简化的质量评估
        if peak_stats['total_peaks'] > 0:
            # 计算关键指标
            angle_span = peak_stats['angle_range'][1] - peak_stats['angle_range'][0]
            peak_density = peak_stats['total_peaks'] / angle_span if angle_span > 0 else 0
            intensity_ratio = peak_stats['strong_peaks_count'] / peak_stats['total_peaks'] * 100

            result += "## 图谱质量指标\n\n"
            result += f"- **峰密度**: {peak_density:.2f} 峰/度\n"
            result += f"- **强峰比例**: {intensity_ratio:.1f}%\n"

            if len(filtered_peaks) >= 2:
                min_separation = min(abs(filtered_peaks[i]['two_theta'] - filtered_peaks[i+1]['two_theta'])
                                   for i in range(len(filtered_peaks)-1))
                result += f"- **最小峰间距**: {min_separation:.2f}°\n"
        else:
            result += "## 注意事项\n\n"
            result += "未检测到有效衍射峰，建议降低强度阈值或扩大角度范围。\n"

        return result

    except Exception as e:
        return f"Error: XRD衍射图谱模拟失败: {str(e)}"


# ============================================================================
# 元素替位 / 同形替换 (prototype substitution)
# ============================================================================

@llm_tool(name="substitute_elements_in_structure_pymatgen",
          description="对 CIF 结构做一次性元素替换（如 {'In':'Ga'} 或 {'In':'Ga','S':'Se'}），保留分数坐标、晶格、空间群。用于从已知母体结构（如 ICSD prototype）生成同形的化学类似物 CIF。")
async def substitute_elements_in_structure_pymatgen(
    cif_string: str,
    substitutions: Dict[str, str],
) -> str:
    """
    使用 pymatgen.Structure.replace_species 对 CIF 做整体元素替换。
    所有 Wyckoff 位点保留，只换原子种类，化学式、晶格、对称性元数据
    由 pymatgen 自动重算。

    Args:
        cif_string: 原始 CIF 字符串
        substitutions: 替换映射，如 {"In":"Ga"} 或 {"In":"Ga","S":"Se"}。
            key=待替换元素，value=目标元素，二者均为元素符号字符串。

    Returns:
        Markdown 格式：替换前后对比表 + 新 CIF。
    """
    try:
        if not isinstance(substitutions, dict) or not substitutions:
            return "Error: substitutions 必须为非空字典，如 {'In': 'Ga'}"

        for old_sym, new_sym in substitutions.items():
            try:
                Element(old_sym)
            except Exception:
                return f"Error: 待替换元素符号 '{old_sym}' 不是合法元素"
            try:
                Element(new_sym)
            except Exception:
                return f"Error: 目标元素符号 '{new_sym}' 不是合法元素"

        try:
            structure = load_structure_from_cif_string(cif_string)
        except Exception as e:
            return f"Error: CIF 结构解析失败: {str(e)}"

        old_formula = structure.composition.reduced_formula
        try:
            old_sg = structure.get_space_group_info()
        except Exception:
            old_sg = ("?", 0)
        old_volume = structure.volume

        present_elements = {str(el) for el in structure.composition.elements}
        missing = [k for k in substitutions if k not in present_elements]
        if missing:
            return (f"Error: 结构中不含待替换元素 {missing}; "
                    f"当前结构含元素 {sorted(present_elements)}")

        try:
            structure.replace_species(substitutions)
        except Exception as e:
            return f"Error: 元素替换执行失败: {str(e)}"

        new_formula = structure.composition.reduced_formula
        try:
            new_sg = structure.get_space_group_info()
        except Exception:
            new_sg = ("?", 0)
        new_volume = structure.volume
        new_cif = structure.to(fmt="cif")

        sub_pairs = ", ".join(f"{k}→{v}" for k, v in substitutions.items())
        result = "# 元素替换后的晶体结构\n\n"
        result += f"**替换映射**: {sub_pairs}\n\n"
        result += "## 替换前后对比\n\n"
        result += "| | 替换前 | 替换后 |\n"
        result += "|---|---|---|\n"
        result += f"| 化学式 | {old_formula} | {new_formula} |\n"
        result += f"| 空间群 | {old_sg[0]} (#{old_sg[1]}) | {new_sg[0]} (#{new_sg[1]}) |\n"
        result += f"| 体积 (Å³) | {old_volume:.2f} | {new_volume:.2f} |\n\n"

        if new_sg[1] != old_sg[1]:
            result += ("⚠️ **空间群在替换后发生变化** —— 通常说明新元素对原 Wyckoff "
                       "位点的对称性不兼容（离子半径差异过大或电子构型不同）。"
                       "建议后续做 ML 弛豫或 DFT 优化以重新确认对称性。\n\n")

        result += "## 替换后 CIF\n\n"
        result += "```\n"
        result += new_cif.strip() + "\n"
        result += "```\n"

        return result

    except Exception as e:
        return f"Error: 元素替换工具失败: {str(e)}"


# ============================================================================
# Lightweight RL-oriented structure tools
# ============================================================================

def _safe_space_group_info(structure: Structure, symprec: float = 0.1) -> Tuple[str, int]:
    try:
        sga = SpacegroupAnalyzer(structure, symprec=symprec)
        return sga.get_space_group_symbol(), int(sga.get_space_group_number())
    except Exception:
        try:
            sg = structure.get_space_group_info()
            return str(sg[0]), int(sg[1])
        except Exception:
            return "unknown", 0


def _structure_summary_dict(structure: Structure, symprec: float = 0.1) -> Dict[str, Any]:
    sg_symbol, sg_number = _safe_space_group_info(structure, symprec=symprec)
    lattice = structure.lattice
    comp = structure.composition
    return {
        "formula": comp.formula,
        "reduced_formula": comp.reduced_formula,
        "anonymous_formula": comp.anonymized_formula,
        "num_sites": len(structure),
        "num_elements": len(comp.elements),
        "elements": [str(el) for el in comp.elements],
        "space_group_symbol": sg_symbol,
        "space_group_number": sg_number,
        "crystal_system": SpacegroupAnalyzer(structure, symprec=symprec).get_crystal_system()
            if sg_number else "unknown",
        "lattice": {
            "a": float(lattice.a),
            "b": float(lattice.b),
            "c": float(lattice.c),
            "alpha": float(lattice.alpha),
            "beta": float(lattice.beta),
            "gamma": float(lattice.gamma),
            "volume": float(lattice.volume),
        },
        "density_g_cm3": float(structure.density),
        "volume_per_atom": float(structure.volume / max(len(structure), 1)),
    }


def _json_block(payload: Dict[str, Any]) -> str:
    return "```json\n" + json.dumps(payload, ensure_ascii=False, indent=2) + "\n```"


@llm_tool(name="analyze_cif_structure_pymatgen",
          description="Parse a CIF and return compact crystallographic metadata for validation and RL rewards.")
async def analyze_cif_structure_pymatgen(
    cif_string: str,
    symprec: float = 0.1,
    include_cif: bool = False
) -> str:
    """
    Parse a CIF and summarize formula, lattice, space group, density, and site count.
    This is a low-cost deterministic validator for structure-generation rollouts.
    """
    try:
        symprec = validate_numeric_parameter(symprec, "symprec", 1e-5, 1.0)
        structure = load_structure_from_cif_string(cif_string)
        summary = _structure_summary_dict(structure, symprec=symprec)
        summary["parse_success"] = True

        result = "# CIF Structure Analysis\n\n"
        result += _json_block(summary)
        if include_cif:
            result += "\n\n## Standardized CIF\n\n```\n"
            result += str(CifWriter(structure, symprec=symprec)).strip()
            result += "\n```"
        return result
    except Exception as e:
        return "Error: CIF analysis failed: " + str(e)


@llm_tool(name="compare_structures_pymatgen",
          description="Compare two CIF structures using pymatgen StructureMatcher and return match metrics.")
async def compare_structures_pymatgen(
    cif_string_a: str,
    cif_string_b: str,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0,
    primitive_cell: bool = True,
    scale: bool = True,
    attempt_supercell: bool = False
) -> str:
    """
    Compare two structures with StructureMatcher. Useful for duplicate audits,
    novelty checks, and target-similarity rewards.
    """
    try:
        ltol = validate_numeric_parameter(ltol, "ltol", 0.0, 1.0)
        stol = validate_numeric_parameter(stol, "stol", 0.0, 2.0)
        angle_tol = validate_numeric_parameter(angle_tol, "angle_tol", 0.0, 30.0)
        struct_a = load_structure_from_cif_string(cif_string_a)
        struct_b = load_structure_from_cif_string(cif_string_b)
        matcher = StructureMatcher(
            ltol=ltol,
            stol=stol,
            angle_tol=angle_tol,
            primitive_cell=primitive_cell,
            scale=scale,
            attempt_supercell=attempt_supercell,
        )
        is_match = bool(matcher.fit(struct_a, struct_b))
        rms = None
        try:
            rms = matcher.get_rms_dist(struct_a, struct_b)
        except Exception:
            rms = None
        anon_match = False
        try:
            anon_match = bool(matcher.fit_anonymous(struct_a, struct_b))
        except Exception:
            anon_match = False

        payload = {
            "match": is_match,
            "anonymous_match": anon_match,
            "formula_a": struct_a.composition.reduced_formula,
            "formula_b": struct_b.composition.reduced_formula,
            "num_sites_a": len(struct_a),
            "num_sites_b": len(struct_b),
            "space_group_a": _safe_space_group_info(struct_a),
            "space_group_b": _safe_space_group_info(struct_b),
            "rms_distance": rms,
            "parameters": {
                "ltol": ltol,
                "stol": stol,
                "angle_tol": angle_tol,
                "primitive_cell": primitive_cell,
                "scale": scale,
                "attempt_supercell": attempt_supercell,
            }
        }
        return "# Structure Comparison\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: structure comparison failed: " + str(e)


@llm_tool(name="score_generated_cif_against_target_pymatgen",
          description="Score a generated CIF against target composition, space group, site count, and optional reference CIF.")
async def score_generated_cif_against_target_pymatgen(
    generated_cif: str,
    target_formula: str = "",
    target_space_group_number: int = 0,
    target_space_group_symbol: str = "",
    target_num_sites: int = 0,
    reference_cif: str = "",
    symprec: float = 0.1,
    ltol: float = 0.2,
    stol: float = 0.3,
    angle_tol: float = 5.0
) -> str:
    """
    Deterministic scoring utility for RL rewards. It intentionally avoids database
    lookup and only checks the generated CIF against provided targets.
    """
    try:
        symprec = validate_numeric_parameter(symprec, "symprec", 1e-5, 1.0)
        structure = load_structure_from_cif_string(generated_cif)
        summary = _structure_summary_dict(structure, symprec=symprec)

        checks = {}
        scores = []
        if target_formula:
            try:
                target_comp = Composition(target_formula)
                formula_match = structure.composition.reduced_composition == target_comp.reduced_composition
            except Exception:
                formula_match = False
            checks["formula_match"] = bool(formula_match)
            scores.append(1.0 if formula_match else 0.0)

        if target_space_group_number:
            sg_match = int(summary["space_group_number"]) == int(target_space_group_number)
            checks["space_group_number_match"] = bool(sg_match)
            scores.append(1.0 if sg_match else 0.0)

        if target_space_group_symbol:
            sg_symbol_match = summary["space_group_symbol"].replace(" ", "") == target_space_group_symbol.replace(" ", "")
            checks["space_group_symbol_match"] = bool(sg_symbol_match)
            scores.append(1.0 if sg_symbol_match else 0.0)

        if target_num_sites:
            site_match = int(summary["num_sites"]) == int(target_num_sites)
            checks["num_sites_match"] = bool(site_match)
            scores.append(1.0 if site_match else 0.0)

        reference_match = None
        if reference_cif:
            ref = load_structure_from_cif_string(reference_cif)
            matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
            reference_match = bool(matcher.fit(structure, ref))
            checks["reference_structure_match"] = reference_match
            scores.append(1.0 if reference_match else 0.0)

        overall_score = float(sum(scores) / len(scores)) if scores else 1.0
        payload = {
            "parse_success": True,
            "overall_score": overall_score,
            "checks": checks,
            "generated_summary": summary,
            "reference_match": reference_match,
        }
        return "# Generated CIF Target Score\n\n" + _json_block(payload)
    except Exception as e:
        payload = {"parse_success": False, "overall_score": 0.0, "error": str(e)}
        return "# Generated CIF Target Score\n\n" + _json_block(payload)


@llm_tool(name="apply_vegards_law_refinement_pymatgen",
          description="Refine lattice parameters for a mixed-composition structure using Vegard's law endmember data.")
async def apply_vegards_law_refinement_pymatgen(
    cif_string: str,
    endmember_lattices: Dict[str, Dict[str, float]],
    fractions: Dict[str, float],
    preserve_angles: bool = True,
    symprec: float = 0.1
) -> str:
    """
    Apply a transparent Vegard's-law lattice interpolation. The keys in fractions
    must match keys in endmember_lattices. Each lattice dict should include a,b,c
    and optionally alpha,beta,gamma.
    """
    try:
        if not endmember_lattices or not fractions:
            return "Error: endmember_lattices and fractions must be non-empty dictionaries"
        missing = [k for k in fractions if k not in endmember_lattices]
        if missing:
            return "Error: fractions include keys missing from endmember_lattices: " + str(missing)

        total = float(sum(float(v) for v in fractions.values()))
        if total <= 0:
            return "Error: fractions must sum to a positive value"
        norm = {k: float(v) / total for k, v in fractions.items()}

        params = {}
        for p in ["a", "b", "c", "alpha", "beta", "gamma"]:
            vals = []
            for key, frac in norm.items():
                lattice_data = endmember_lattices[key]
                if p in lattice_data:
                    vals.append(frac * float(lattice_data[p]))
            if vals:
                params[p] = sum(vals)

        structure = load_structure_from_cif_string(cif_string)
        old_lattice = structure.lattice
        if preserve_angles:
            alpha, beta, gamma = old_lattice.alpha, old_lattice.beta, old_lattice.gamma
        else:
            alpha = params.get("alpha", old_lattice.alpha)
            beta = params.get("beta", old_lattice.beta)
            gamma = params.get("gamma", old_lattice.gamma)

        new_lattice = Lattice.from_parameters(
            params.get("a", old_lattice.a),
            params.get("b", old_lattice.b),
            params.get("c", old_lattice.c),
            alpha,
            beta,
            gamma,
        )
        refined = Structure(new_lattice, structure.species, structure.frac_coords,
                            site_properties=structure.site_properties)

        payload = {
            "formula": refined.composition.reduced_formula,
            "fractions_normalized": norm,
            "old_lattice": {
                "a": old_lattice.a, "b": old_lattice.b, "c": old_lattice.c,
                "alpha": old_lattice.alpha, "beta": old_lattice.beta, "gamma": old_lattice.gamma,
                "volume": old_lattice.volume,
            },
            "new_lattice": {
                "a": refined.lattice.a, "b": refined.lattice.b, "c": refined.lattice.c,
                "alpha": refined.lattice.alpha, "beta": refined.lattice.beta, "gamma": refined.lattice.gamma,
                "volume": refined.lattice.volume,
            },
            "space_group": _safe_space_group_info(refined, symprec=symprec),
        }
        result = "# Vegard Lattice Refinement\n\n" + _json_block(payload)
        result += "\n\n## Refined CIF\n\n```\n" + str(CifWriter(refined, symprec=symprec)).strip() + "\n```"
        return result
    except Exception as e:
        return "Error: Vegard refinement failed: " + str(e)


@llm_tool(name="enumerate_fractional_occupancy_structure_pymatgen",
          description="Create ordered approximants from a fractional-occupancy CIF using pymatgen order_disordered_structure.")
async def enumerate_fractional_occupancy_structure_pymatgen(
    cif_string: str,
    max_structures: int = 4,
    symprec: float = 0.1
) -> str:
    """
    Convert a disordered/fractional-occupancy structure into ordered approximants.
    This gives an auditable algorithm for mixed-site structures.
    """
    try:
        max_structures = int(validate_numeric_parameter(max_structures, "max_structures", 1, 20))
        structure = load_structure_from_cif_string(cif_string)
        if structure.is_ordered:
            ordered_structures = [structure]
        else:
            transformer = OrderDisorderedStructureTransformation()
            transformed = transformer.apply_transformation(structure, return_ranked_list=max_structures)
            ordered_structures = [item["structure"] for item in transformed]
        ordered_structures = ordered_structures[:max_structures]

        payload = {
            "input_formula": structure.composition.formula,
            "input_reduced_formula": structure.composition.reduced_formula,
            "input_is_ordered": bool(structure.is_ordered),
            "num_ordered_structures": len(ordered_structures),
            "structures": [
                _structure_summary_dict(s, symprec=symprec) for s in ordered_structures
            ],
        }
        result = "# Fractional Occupancy Enumeration\n\n" + _json_block(payload)
        for i, s in enumerate(ordered_structures, 1):
            result += f"\n\n## Ordered CIF {i}\n\n```\n"
            result += str(CifWriter(s, symprec=symprec)).strip()
            result += "\n```"
        return result
    except Exception as e:
        return "Error: fractional occupancy enumeration failed: " + str(e)


@llm_tool(name="build_bulk_supercell_or_slab_pymatgen",
          description="Build a simple bulk structure, supercell, or surface slab for structure-design rollouts.")
async def build_bulk_supercell_or_slab_pymatgen(
    mode: str,
    lattice_type: str = "cubic",
    species: List[str] = None,
    frac_coords: List[List[float]] = None,
    lattice_parameters: Dict[str, float] = None,
    input_cif: str = "",
    supercell_matrix: List[List[int]] = None,
    miller_index: List[int] = None,
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 10.0,
    symprec: float = 0.1
) -> str:
    """
    Lightweight builder for three common operations:
    - mode='bulk': build a Structure from species, fractional coordinates, and lattice parameters.
    - mode='supercell': expand input_cif by a 3x3 integer matrix.
    - mode='slab': generate a slab from input_cif and a Miller index.
    """
    try:
        mode = mode.lower().strip()
        if mode == "bulk":
            if not species or not frac_coords:
                return "Error: bulk mode requires species and frac_coords"
            lattice_parameters = lattice_parameters or {}
            if lattice_type.lower() == "cubic":
                a = float(lattice_parameters.get("a", 3.0))
                lattice = Lattice.cubic(a)
            elif lattice_type.lower() == "tetragonal":
                lattice = Lattice.tetragonal(
                    float(lattice_parameters.get("a", 3.0)),
                    float(lattice_parameters.get("c", 5.0)),
                )
            elif lattice_type.lower() == "orthorhombic":
                lattice = Lattice.orthorhombic(
                    float(lattice_parameters.get("a", 3.0)),
                    float(lattice_parameters.get("b", 4.0)),
                    float(lattice_parameters.get("c", 5.0)),
                )
            elif lattice_type.lower() == "hexagonal":
                lattice = Lattice.hexagonal(
                    float(lattice_parameters.get("a", 3.0)),
                    float(lattice_parameters.get("c", 5.0)),
                )
            else:
                lattice = Lattice.from_parameters(
                    float(lattice_parameters.get("a", 3.0)),
                    float(lattice_parameters.get("b", 3.0)),
                    float(lattice_parameters.get("c", 3.0)),
                    float(lattice_parameters.get("alpha", 90.0)),
                    float(lattice_parameters.get("beta", 90.0)),
                    float(lattice_parameters.get("gamma", 90.0)),
                )
            structure = Structure(lattice, species, frac_coords)

        elif mode == "supercell":
            if not input_cif:
                return "Error: supercell mode requires input_cif"
            structure = load_structure_from_cif_string(input_cif)
            if supercell_matrix is None:
                supercell_matrix = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            structure.make_supercell(supercell_matrix)

        elif mode == "slab":
            if not input_cif:
                return "Error: slab mode requires input_cif"
            structure = load_structure_from_cif_string(input_cif)
            if miller_index is None:
                miller_index = [1, 0, 0]
            generator = SlabGenerator(
                initial_structure=structure,
                miller_index=tuple(int(x) for x in miller_index),
                min_slab_size=float(min_slab_size),
                min_vacuum_size=float(min_vacuum_size),
                center_slab=True,
            )
            slabs = generator.get_slabs(symmetrize=False)
            if not slabs:
                return "Error: no slab generated"
            structure = slabs[0]
        else:
            return "Error: mode must be one of bulk, supercell, slab"

        payload = {
            "mode": mode,
            "summary": _structure_summary_dict(structure, symprec=symprec),
        }
        result = "# Structure Builder Result\n\n" + _json_block(payload)
        result += "\n\n## CIF\n\n```\n" + str(CifWriter(structure, symprec=symprec)).strip() + "\n```"
        return result
    except Exception as e:
        return "Error: structure builder failed: " + str(e)


@llm_tool(name="standardize_structure_pymatgen",
          description="Standardize a CIF structure with pymatgen/spglib and return primitive or conventional CIF.")
async def standardize_structure_pymatgen(
    cif_string: str,
    mode: str = "conventional",
    symprec: float = 0.1,
    angle_tolerance: float = 5.0
) -> str:
    """Return a standardized structure for symmetry-aware design rollouts."""
    try:
        symprec = validate_numeric_parameter(symprec, "symprec", 1e-5, 1.0)
        angle_tolerance = validate_numeric_parameter(angle_tolerance, "angle_tolerance", 0.0, 30.0)
        structure = load_structure_from_cif_string(cif_string)
        analyzer = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        mode_l = mode.lower().strip()
        if mode_l == "primitive":
            standardized = analyzer.get_primitive_standard_structure()
        elif mode_l == "refined":
            standardized = analyzer.get_refined_structure()
        else:
            standardized = analyzer.get_conventional_standard_structure()

        payload = {
            "input_summary": _structure_summary_dict(structure, symprec=symprec),
            "standardized_summary": _structure_summary_dict(standardized, symprec=symprec),
            "mode": mode_l,
            "symmetry_dataset_available": analyzer.get_symmetry_dataset() is not None,
        }
        result = "# Standardized Structure\n\n" + _json_block(payload)
        result += "\n\n## Standardized CIF\n\n```\n"
        result += str(CifWriter(standardized, symprec=symprec)).strip()
        result += "\n```"
        return result
    except Exception as e:
        return "Error: structure standardization failed: " + str(e)


@llm_tool(name="extract_symmetry_and_wyckoff_pymatgen",
          description="Extract space-group, equivalent atoms, and Wyckoff labels from a CIF using pymatgen/spglib.")
async def extract_symmetry_and_wyckoff_pymatgen(
    cif_string: str,
    symprec: float = 0.1,
    angle_tolerance: float = 5.0
) -> str:
    """Expose Wyckoff/symmetry constraints as a deterministic low-level tool."""
    try:
        symprec = validate_numeric_parameter(symprec, "symprec", 1e-5, 1.0)
        angle_tolerance = validate_numeric_parameter(angle_tolerance, "angle_tolerance", 0.0, 30.0)
        structure = load_structure_from_cif_string(cif_string)
        analyzer = SpacegroupAnalyzer(
            structure, symprec=symprec, angle_tolerance=angle_tolerance
        )
        dataset = analyzer.get_symmetry_dataset()
        if hasattr(dataset, "wyckoffs"):
            wyckoffs = list(dataset.wyckoffs)
        else:
            wyckoffs = list(dataset.get("wyckoffs", []))
        if hasattr(dataset, "equivalent_atoms"):
            equivalent_atoms = list(dataset.equivalent_atoms)
        else:
            equivalent_atoms = list(dataset.get("equivalent_atoms", []))
        sites = []
        for i, site in enumerate(structure):
            sites.append({
                "index": i,
                "element": site.specie.symbol if hasattr(site, "specie") else str(site.species_string),
                "frac_coords": [float(x) for x in site.frac_coords],
                "wyckoff": wyckoffs[i] if i < len(wyckoffs) else None,
                "equivalent_atom": int(equivalent_atoms[i]) if i < len(equivalent_atoms) else None,
            })

        payload = {
            "summary": _structure_summary_dict(structure, symprec=symprec),
            "space_group_symbol": analyzer.get_space_group_symbol(),
            "space_group_number": int(analyzer.get_space_group_number()),
            "crystal_system": analyzer.get_crystal_system(),
            "point_group_symbol": analyzer.get_point_group_symbol(),
            "sites": sites,
        }
        return "# Symmetry and Wyckoff Analysis\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: symmetry/Wyckoff extraction failed: " + str(e)


@llm_tool(name="estimate_oxidation_states_pymatgen",
          description="Estimate oxidation states from composition and optionally decorate a CIF with bond-valence oxidation states.")
async def estimate_oxidation_states_pymatgen(
    formula: str = "",
    cif_string: str = "",
    max_guesses: int = 8
) -> str:
    """Estimate oxidation states without database lookup."""
    try:
        max_guesses = int(validate_numeric_parameter(max_guesses, "max_guesses", 1, 50))
        payload: Dict[str, Any] = {}
        if formula:
            comp = Composition(formula)
            payload["formula"] = comp.reduced_formula
            payload["composition_oxidation_guesses"] = comp.oxi_state_guesses(max_sites=-1)[:max_guesses]

        if cif_string:
            structure = load_structure_from_cif_string(cif_string)
            payload["structure_formula"] = structure.composition.reduced_formula
            try:
                decorated = BVAnalyzer().get_oxi_state_decorated_structure(structure)
                site_oxi = []
                for i, site in enumerate(decorated):
                    specie = site.specie
                    site_oxi.append({
                        "index": i,
                        "species": str(specie),
                        "element": specie.symbol,
                        "oxidation_state": float(specie.oxi_state),
                    })
                payload["bond_valence_success"] = True
                payload["site_oxidation_states"] = site_oxi
            except Exception as exc:
                payload["bond_valence_success"] = False
                payload["bond_valence_error"] = str(exc)

        if not payload:
            return "Error: provide at least formula or cif_string"
        return "# Oxidation State Estimate\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: oxidation-state estimation failed: " + str(e)


@llm_tool(name="compute_bond_valence_pymatgen",
          description="Compute bond-valence oxidation-state diagnostics for a CIF using pymatgen BVAnalyzer.")
async def compute_bond_valence_pymatgen(cif_string: str) -> str:
    """Check whether a candidate structure admits plausible bond valences."""
    try:
        structure = load_structure_from_cif_string(cif_string)
        analyzer = BVAnalyzer()
        decorated = analyzer.get_oxi_state_decorated_structure(structure)
        site_data = []
        by_element: Dict[str, List[float]] = {}
        for i, site in enumerate(decorated):
            specie = site.specie
            oxi = float(specie.oxi_state)
            by_element.setdefault(specie.symbol, []).append(oxi)
            site_data.append({
                "index": i,
                "species": str(specie),
                "element": specie.symbol,
                "oxidation_state": oxi,
                "frac_coords": [float(x) for x in site.frac_coords],
            })
        payload = {
            "success": True,
            "formula": structure.composition.reduced_formula,
            "site_oxidation_states": site_data,
            "element_oxidation_states": {
                el: sorted(set(vals)) for el, vals in by_element.items()
            },
            "charge_sum": float(sum(site.specie.oxi_state for site in decorated)),
        }
        return "# Bond Valence Analysis\n\n" + _json_block(payload)
    except Exception as e:
        payload = {"success": False, "error": str(e)}
        return "# Bond Valence Analysis\n\n" + _json_block(payload)


@llm_tool(name="analyze_coordination_environment_pymatgen",
          description="Analyze local coordination environments, nearest neighbors, and short bonds in a CIF.")
async def analyze_coordination_environment_pymatgen(
    cif_string: str,
    max_neighbors: int = 12,
    short_bond_factor: float = 0.65
) -> str:
    """Return local coordination fingerprints for candidate repair and scoring."""
    try:
        max_neighbors = int(validate_numeric_parameter(max_neighbors, "max_neighbors", 1, 32))
        short_bond_factor = validate_numeric_parameter(short_bond_factor, "short_bond_factor", 0.1, 1.2)
        structure = load_structure_from_cif_string(cif_string)
        cnn = CrystalNN()
        sites = []
        short_contacts = []
        for i, site in enumerate(structure):
            try:
                nn_info = cnn.get_nn_info(structure, i)[:max_neighbors]
            except Exception:
                nn_info = []
            neighbors = []
            for item in nn_info:
                nn_site = item["site"]
                distance = float(site.distance(nn_site))
                neighbors.append({
                    "element": nn_site.specie.symbol if hasattr(nn_site, "specie") else nn_site.species_string,
                    "distance": distance,
                    "weight": float(item.get("weight", 0.0)),
                    "site_index": int(item.get("site_index", -1)),
                })
            sites.append({
                "index": i,
                "element": site.specie.symbol if hasattr(site, "specie") else site.species_string,
                "coordination_number": len(neighbors),
                "neighbors": neighbors,
            })

        for i in range(len(structure)):
            for j in range(i + 1, len(structure)):
                dist = float(structure.get_distance(i, j))
                try:
                    r_i = Element(structure[i].specie.symbol).atomic_radius or Element(structure[i].specie.symbol).average_ionic_radius
                    r_j = Element(structure[j].specie.symbol).atomic_radius or Element(structure[j].specie.symbol).average_ionic_radius
                    threshold = float(short_bond_factor) * float(r_i + r_j)
                except Exception:
                    threshold = 0.7
                if dist < threshold:
                    short_contacts.append({
                        "site_i": i,
                        "site_j": j,
                        "elements": [structure[i].specie.symbol, structure[j].specie.symbol],
                        "distance": dist,
                        "threshold": threshold,
                    })

        payload = {
            "formula": structure.composition.reduced_formula,
            "num_sites": len(structure),
            "sites": sites,
            "short_contact_count": len(short_contacts),
            "short_contacts": short_contacts[:50],
        }
        return "# Coordination Environment Analysis\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: coordination analysis failed: " + str(e)


@llm_tool(name="perturb_lattice_or_positions_pymatgen",
          description="Create a locally perturbed CIF by scaling lattice lengths and/or shifting selected fractional coordinates.")
async def perturb_lattice_or_positions_pymatgen(
    cif_string: str,
    lattice_scale: Dict[str, float] = None,
    site_shifts: Dict[str, List[float]] = None,
    wrap: bool = True,
    symprec: float = 0.1
) -> str:
    """A small deterministic structure-editing primitive for iterative repair."""
    try:
        structure = load_structure_from_cif_string(cif_string)
        lattice_scale = lattice_scale or {}
        a = structure.lattice.a * float(lattice_scale.get("a", 1.0))
        b = structure.lattice.b * float(lattice_scale.get("b", 1.0))
        c = structure.lattice.c * float(lattice_scale.get("c", 1.0))
        new_lattice = Lattice.from_parameters(
            a, b, c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
        )
        new_frac = np.array(structure.frac_coords, dtype=float)
        for raw_idx, shift in (site_shifts or {}).items():
            idx = int(raw_idx)
            if idx < 0 or idx >= len(structure):
                continue
            if len(shift) != 3:
                continue
            new_frac[idx] = new_frac[idx] + np.array([float(x) for x in shift])
        if wrap:
            new_frac = new_frac % 1.0

        mutated = Structure(
            new_lattice,
            structure.species,
            new_frac,
            site_properties=structure.site_properties,
        )
        payload = {
            "input_summary": _structure_summary_dict(structure, symprec=symprec),
            "mutated_summary": _structure_summary_dict(mutated, symprec=symprec),
            "lattice_scale": lattice_scale,
            "site_shifts": site_shifts or {},
        }
        result = "# Perturbed Structure\n\n" + _json_block(payload)
        result += "\n\n## Mutated CIF\n\n```\n" + str(CifWriter(mutated, symprec=symprec)).strip() + "\n```"
        return result
    except Exception as e:
        return "Error: structure perturbation failed: " + str(e)


@llm_tool(name="convert_cif_to_vasp_inputs_pymatgen",
          description="Generate POSCAR, INCAR, and KPOINTS text from a CIF without running VASP.")
async def convert_cif_to_vasp_inputs_pymatgen(
    cif_string: str,
    task: str = "static",
    kppa: int = 1000,
    user_incar_settings: Dict[str, Any] = None
) -> str:
    """Prepare auditable VASP input files; this does not execute VASP."""
    try:
        kppa = int(validate_numeric_parameter(kppa, "kppa", 10, 100000))
        structure = load_structure_from_cif_string(cif_string)
        task_l = task.lower().strip()
        incar_defaults: Dict[str, Any] = {
            "ENCUT": 520,
            "EDIFF": 1e-5,
            "ISMEAR": 0,
            "SIGMA": 0.05,
            "LASPH": True,
            "LREAL": False,
        }
        if task_l in {"relax", "optimization", "opt"}:
            incar_defaults.update({"IBRION": 2, "ISIF": 3, "NSW": 99, "EDIFFG": -0.02})
        elif task_l in {"band", "band_structure"}:
            incar_defaults.update({"ICHARG": 11, "ISMEAR": 0, "SIGMA": 0.05})
        elif task_l in {"dos", "density_of_states"}:
            incar_defaults.update({"NEDOS": 2000, "LORBIT": 11})
        else:
            incar_defaults.update({"IBRION": -1, "NSW": 0})

        incar_defaults.update(user_incar_settings or {})
        poscar = Poscar(structure)
        incar = Incar(incar_defaults)
        kpoints = Kpoints.automatic_density(structure, kppa)
        payload = {
            "formula": structure.composition.reduced_formula,
            "task": task_l,
            "kppa": kppa,
            "num_sites": len(structure),
            "incar": dict(incar),
        }
        result = "# VASP Input Files\n\n" + _json_block(payload)
        result += "\n\n## POSCAR\n\n```text\n" + str(poscar).strip() + "\n```"
        result += "\n\n## INCAR\n\n```text\n" + str(incar).strip() + "\n```"
        result += "\n\n## KPOINTS\n\n```text\n" + str(kpoints).strip() + "\n```"
        return result
    except Exception as e:
        return "Error: VASP input generation failed: " + str(e)


@llm_tool(name="parse_vasp_outputs_pymatgen",
          description="Parse uploaded vasprun.xml and/or OUTCAR text and summarize convergence, energy, band gap, and magnetization.")
async def parse_vasp_outputs_pymatgen(
    vasprun_xml: str = "",
    outcar_text: str = "",
    parse_dos: bool = False
) -> str:
    """Parse existing VASP outputs from text; this does not run VASP."""
    try:
        payload: Dict[str, Any] = {"vasprun_parsed": False, "outcar_parsed": False}
        if vasprun_xml:
            with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
                handle.write(vasprun_xml)
                vasprun_path = handle.name
            try:
                vasprun = Vasprun(
                    vasprun_path,
                    parse_dos=parse_dos,
                    parse_eigen=False,
                    parse_projected_eigen=False,
                    exception_on_bad_xml=False,
                )
                payload.update({
                    "vasprun_parsed": True,
                    "converged": bool(vasprun.converged),
                    "final_energy": float(vasprun.final_energy),
                    "final_energy_per_atom": float(vasprun.final_energy / max(len(vasprun.final_structure), 1)),
                    "formula": vasprun.final_structure.composition.reduced_formula,
                    "num_sites": len(vasprun.final_structure),
                })
                try:
                    payload["band_gap"] = vasprun.eigenvalue_band_properties[0]
                    payload["cbm"] = vasprun.eigenvalue_band_properties[1]
                    payload["vbm"] = vasprun.eigenvalue_band_properties[2]
                    payload["is_gap_direct"] = bool(vasprun.eigenvalue_band_properties[3])
                except Exception:
                    pass
            finally:
                try:
                    os.unlink(vasprun_path)
                except Exception:
                    pass

        if outcar_text:
            with tempfile.NamedTemporaryFile("w", suffix=".OUTCAR", delete=False) as handle:
                handle.write(outcar_text)
                outcar_path = handle.name
            try:
                outcar = Outcar(outcar_path)
                payload["outcar_parsed"] = True
                if getattr(outcar, "magnetization", None) is not None:
                    payload["magnetization"] = outcar.magnetization
                if getattr(outcar, "run_stats", None):
                    payload["run_stats"] = outcar.run_stats
                if getattr(outcar, "final_energy", None) is not None:
                    payload["outcar_final_energy"] = outcar.final_energy
            finally:
                try:
                    os.unlink(outcar_path)
                except Exception:
                    pass

        if not vasprun_xml and not outcar_text:
            return "Error: provide vasprun_xml and/or outcar_text"
        return "# VASP Output Summary\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: VASP output parsing failed: " + str(e)


@llm_tool(name="generate_surface_slabs_pymatgen",
          description="Generate surface slabs from a bulk CIF using pymatgen SlabGenerator without running DFT.")
async def generate_surface_slabs_pymatgen(
    cif_string: str,
    miller_indices: List[List[int]] = None,
    min_slab_size: float = 10.0,
    min_vacuum_size: float = 12.0,
    max_slabs_per_miller: int = 3,
    center_slab: bool = True,
    symprec: float = 0.1
) -> str:
    """Generate slab candidates for catalyst/surface workflows."""
    try:
        structure = load_structure_from_cif_string(cif_string)
        miller_indices = miller_indices or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        max_slabs_per_miller = int(validate_numeric_parameter(max_slabs_per_miller, "max_slabs_per_miller", 1, 10))
        slab_records = []
        result = "# Surface Slab Generation\n\n"
        for miller in miller_indices[:8]:
            generator = SlabGenerator(
                initial_structure=structure,
                miller_index=tuple(int(x) for x in miller),
                min_slab_size=float(min_slab_size),
                min_vacuum_size=float(min_vacuum_size),
                center_slab=center_slab,
            )
            slabs = generator.get_slabs(symmetrize=False)[:max_slabs_per_miller]
            for idx, slab in enumerate(slabs):
                record = {
                    "miller_index": [int(x) for x in miller],
                    "slab_index": idx,
                    "summary": _structure_summary_dict(slab, symprec=symprec),
                    "is_polar": bool(getattr(slab, "is_polar", lambda: False)()),
                }
                slab_records.append(record)
                result += f"\n\n## Slab {len(slab_records)} Miller {record['miller_index']}\n\n```cif\n"
                result += str(CifWriter(slab, symprec=symprec)).strip()
                result += "\n```"
        payload = {
            "bulk_summary": _structure_summary_dict(structure, symprec=symprec),
            "num_slabs": len(slab_records),
            "slabs": slab_records,
        }
        return "# Surface Slab Generation\n\n" + _json_block(payload) + result
    except Exception as e:
        return "Error: surface slab generation failed: " + str(e)


@llm_tool(name="find_adsorption_sites_pymatgen",
          description="Find likely adsorption sites on a slab CIF using pymatgen AdsorbateSiteFinder.")
async def find_adsorption_sites_pymatgen(
    slab_cif: str,
    distance: float = 2.0,
    symm_reduce: float = 0.01,
    near_reduce: float = 0.01
) -> str:
    """Find ontop/bridge/hollow adsorption sites for catalyst workflows."""
    try:
        slab = load_structure_from_cif_string(slab_cif)
        finder = AdsorbateSiteFinder(slab)
        sites = finder.find_adsorption_sites(
            distance=float(distance),
            symm_reduce=float(symm_reduce),
            near_reduce=float(near_reduce),
        )
        payload = {
            "slab_formula": slab.composition.reduced_formula,
            "site_counts": {key: len(val) for key, val in sites.items() if isinstance(val, list)},
            "sites": {
                key: [[float(x) for x in coords] for coords in val[:32]]
                for key, val in sites.items()
                if isinstance(val, list)
            },
        }
        return "# Adsorption Site Analysis\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: adsorption-site analysis failed: " + str(e)


@llm_tool(name="place_adsorbate_on_slab_pymatgen",
          description="Place an atom or small molecule adsorbate on a slab adsorption site and return adsorbed-structure CIF.")
async def place_adsorbate_on_slab_pymatgen(
    slab_cif: str,
    adsorbate: str = "N2",
    site: List[float] = None,
    site_type: str = "ontop",
    distance: float = 2.0,
    reorient: bool = True,
    symprec: float = 0.1
) -> str:
    """Build slab+adsorbate structures for downstream DFT input preparation."""
    try:
        slab = load_structure_from_cif_string(slab_cif)
        finder = AdsorbateSiteFinder(slab)
        molecule_templates = {
            "H": Molecule(["H"], [[0, 0, 0]]),
            "N": Molecule(["N"], [[0, 0, 0]]),
            "N2": Molecule(["N", "N"], [[0, 0, 0], [0, 0, 1.10]]),
            "NH3": Molecule(["N", "H", "H", "H"], [[0, 0, 0], [0.94, 0, -0.34], [-0.47, 0.81, -0.34], [-0.47, -0.81, -0.34]]),
            "O": Molecule(["O"], [[0, 0, 0]]),
            "O2": Molecule(["O", "O"], [[0, 0, 0], [0, 0, 1.21]]),
            "H2O": Molecule(["O", "H", "H"], [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
        }
        ads = molecule_templates.get(adsorbate)
        if ads is None:
            comp = Composition(adsorbate)
            elems = []
            coords = []
            offset = 0.0
            for el, amt in comp.get_el_amt_dict().items():
                for _ in range(int(round(amt))):
                    elems.append(el)
                    coords.append([0, 0, offset])
                    offset += 1.2
            ads = Molecule(elems, coords)

        if site is None:
            sites = finder.find_adsorption_sites(distance=float(distance))
            candidates = sites.get(site_type, []) or sites.get("all", [])
            if not candidates:
                return "Error: no adsorption sites found"
            site = candidates[0]
        ads_struct = finder.add_adsorbate(ads, site, reorient=reorient)
        payload = {
            "slab_formula": slab.composition.reduced_formula,
            "adsorbate": adsorbate,
            "site": [float(x) for x in site],
            "adsorbed_summary": _structure_summary_dict(ads_struct, symprec=symprec),
        }
        result = "# Adsorbate Placement\n\n" + _json_block(payload)
        result += "\n\n## Adsorbed Structure CIF\n\n```cif\n"
        result += str(CifWriter(ads_struct, symprec=symprec)).strip()
        result += "\n```"
        return result
    except Exception as e:
        return "Error: adsorbate placement failed: " + str(e)


@llm_tool(name="generate_nrr_intermediates_on_surface_pymatgen",
          description="Generate common nitrogen-reduction adsorbate structures (*N2, *NNH, *NH2, *NH3) on a slab.")
async def generate_nrr_intermediates_on_surface_pymatgen(
    slab_cif: str,
    site: List[float] = None,
    site_type: str = "ontop",
    distance: float = 2.0,
    symprec: float = 0.1
) -> str:
    """Generate NRR intermediate geometries for catalyst candidate comparison."""
    try:
        slab = load_structure_from_cif_string(slab_cif)
        finder = AdsorbateSiteFinder(slab)
        if site is None:
            sites = finder.find_adsorption_sites(distance=float(distance))
            candidates = sites.get(site_type, []) or sites.get("all", [])
            if not candidates:
                return "Error: no adsorption sites found"
            site = candidates[0]
        templates = {
            "N2": Molecule(["N", "N"], [[0, 0, 0], [0, 0, 1.10]]),
            "NNH": Molecule(["N", "N", "H"], [[0, 0, 0], [0, 0, 1.20], [0.75, 0, 1.55]]),
            "NH": Molecule(["N", "H"], [[0, 0, 0], [0.0, 0.0, 1.02]]),
            "NH2": Molecule(["N", "H", "H"], [[0, 0, 0], [0.94, 0, 0.35], [-0.47, 0.81, 0.35]]),
            "NH3": Molecule(["N", "H", "H", "H"], [[0, 0, 0], [0.94, 0, -0.34], [-0.47, 0.81, -0.34], [-0.47, -0.81, -0.34]]),
        }
        records = []
        result = "# NRR Intermediate Structures\n\n"
        for name, molecule in templates.items():
            ads_struct = finder.add_adsorbate(molecule, site, reorient=True)
            records.append({
                "intermediate": name,
                "site": [float(x) for x in site],
                "summary": _structure_summary_dict(ads_struct, symprec=symprec),
            })
            result += f"\n\n## {name} on slab\n\n```cif\n"
            result += str(CifWriter(ads_struct, symprec=symprec)).strip()
            result += "\n```"
        payload = {
            "slab_formula": slab.composition.reduced_formula,
            "num_intermediates": len(records),
            "intermediates": records,
        }
        return "# NRR Intermediate Structures\n\n" + _json_block(payload) + result
    except Exception as e:
        return "Error: NRR intermediate generation failed: " + str(e)


@llm_tool(name="analyze_collinear_magnetism_pymatgen",
          description="Analyze magnetic moments and ordering in a CIF with pymatgen CollinearMagneticStructureAnalyzer.")
async def analyze_collinear_magnetism_pymatgen(
    cif_string: str,
    overwrite_magmom_mode: str = "replace_all_if_undefined"
) -> str:
    """Analyze magnetic ordering when magmom site properties are present or inferred."""
    try:
        structure = load_structure_from_cif_string(cif_string)
        analyzer = CollinearMagneticStructureAnalyzer(
            structure,
            overwrite_magmom_mode=overwrite_magmom_mode,
        )
        mag_structure = analyzer.get_structure_with_spin()
        payload = {
            "formula": structure.composition.reduced_formula,
            "is_magnetic": bool(analyzer.is_magnetic),
            "ordering": str(analyzer.ordering),
            "total_magmoms": float(analyzer.total_magmoms),
            "magnetic_species_and_magmoms": {
                str(k): float(v) for k, v in analyzer.magnetic_species_and_magmoms.items()
            },
            "num_sites": len(structure),
            "magmom_site_property": [
                float(site.properties.get("magmom", 0.0)) for site in mag_structure
            ],
        }
        return "# Collinear Magnetism Analysis\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: magnetism analysis failed: " + str(e)


@llm_tool(name="suggest_initial_magmoms_pymatgen",
          description="Suggest VASP MAGMOM settings for a CIF using element defaults and optional overrides.")
async def suggest_initial_magmoms_pymatgen(
    cif_string: str,
    default_magmoms: Dict[str, float] = None,
    ordering: str = "ferromagnetic"
) -> str:
    """Generate auditable initial magnetic moments for VASP input preparation."""
    try:
        structure = load_structure_from_cif_string(cif_string)
        defaults = {
            "Sc": 1.0, "Ti": 2.0, "V": 3.0, "Cr": 4.0, "Mn": 5.0,
            "Fe": 5.0, "Co": 3.0, "Ni": 2.0, "Cu": 1.0,
            "Ce": 1.0, "Pr": 2.0, "Nd": 3.0, "Sm": 5.0, "Gd": 7.0,
            "Tb": 6.0, "Dy": 5.0, "Ho": 4.0, "Er": 3.0, "Tm": 2.0,
        }
        defaults.update(default_magmoms or {})
        ordering_l = ordering.lower().strip()
        magmoms = []
        for i, site in enumerate(structure):
            base = float(defaults.get(site.specie.symbol, 0.0))
            if ordering_l in {"afm", "antiferromagnetic"} and base:
                base = base if i % 2 == 0 else -base
            magmoms.append(base)
        incar = {"ISPIN": 2, "MAGMOM": magmoms}
        payload = {
            "formula": structure.composition.reduced_formula,
            "ordering_assumption": ordering_l,
            "magmom_by_site": magmoms,
            "incar_settings": incar,
        }
        return "# Initial Magnetic Moments\n\n" + _json_block(payload)
    except Exception as e:
        return "Error: initial MAGMOM suggestion failed: " + str(e)


@llm_tool(name="enumerate_magnetic_orderings_pymatgen",
          description="Enumerate candidate magnetic orderings with pymatgen when available.")
async def enumerate_magnetic_orderings_pymatgen(
    cif_string: str,
    strategies: List[str] = None,
    max_orderings: int = 8
) -> str:
    """Enumerate FM/AFM-like orderings for later DFT setup; does not run DFT."""
    try:
        if MagneticStructureEnumerator is None:
            return "Error: MagneticStructureEnumerator is unavailable in this pymatgen version"
        structure = load_structure_from_cif_string(cif_string)
        strategies = strategies or ["ferromagnetic", "antiferromagnetic"]
        max_orderings = int(validate_numeric_parameter(max_orderings, "max_orderings", 1, 32))
        enumerator = MagneticStructureEnumerator(structure, strategies=strategies)
        ordered_structures = list(enumerator.ordered_structures)[:max_orderings]
        payload = {
            "formula": structure.composition.reduced_formula,
            "strategies": strategies,
            "num_orderings": len(ordered_structures),
            "orderings": [
                {
                    "index": i,
                    "summary": _structure_summary_dict(s, symprec=0.1),
                    "magmoms": [float(v) for v in s.site_properties.get("magmom", [])],
                }
                for i, s in enumerate(ordered_structures)
            ],
        }
        result = "# Magnetic Ordering Enumeration\n\n" + _json_block(payload)
        for i, s in enumerate(ordered_structures):
            result += f"\n\n## Magnetic ordering {i}\n\n```cif\n"
            result += str(CifWriter(s, symprec=0.1)).strip()
            result += "\n```"
        return result
    except Exception as e:
        return "Error: magnetic ordering enumeration failed: " + str(e)
