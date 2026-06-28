"""
MatterGen 材料生成工具函数 - 适配 MCP 服务器框架
"""

import json
import logging
from typing import Dict, Optional, Union

from core.core import llm_tool
from mattergen_service import MatterGenService

logger = logging.getLogger(__name__)

def _generate_material_base(properties: Optional[Dict] = None, batch_size: int = 2, num_batches: int = 1, diffusion_guidance_factor: float = 2.0) -> str:
    """
    材料生成的基础函数 - 所有具体函数共享的核心逻辑。
    """
    try:
        service = MatterGenService.get_instance()
        logger.info("成功获取 MatterGenService 实例")

        result = service.generate(
            properties=properties,
            batch_size=batch_size,
            num_batches=num_batches,
            diffusion_guidance_factor=diffusion_guidance_factor
        )
        logger.info("材料生成完成")

        if "Error generating structures" in result:
            return f"Error：生成材料失败，属性约束 {properties}"
        else:
            return result

    except Exception as e:
        logger.error(f"材料生成过程中出错: {e}")
        return f"Error：材料生成失败 - {str(e)}"

# ============================================================================
# 无条件生成
# ============================================================================

@llm_tool(
    name="generate_material_unconditional_MatterGen",
    description="使用 MatterGen 模型生成无属性约束的晶体结构"
)
def generate_material_unconditional_MatterGen(
    batch_size: int = 2,
    num_batches: int = 1
) -> str:
    """
    生成无属性约束的晶体结构（无条件生成）。

    此函数生成材料时不针对任何特定属性，允许生成结构的最大多样性。

    Args:
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_unconditional_MatterGen(batch_size=4, num_batches=2)
    """
    return _generate_material_base(
        properties=None,
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=0.0
    )

# ============================================================================
# 单属性条件生成
# ============================================================================

@llm_tool(
    name="generate_material_by_dft_band_gap_MatterGen",
    description="使用 MatterGen 模型生成具有特定 DFT 带隙的晶体结构"
)
def generate_material_by_dft_band_gap_MatterGen(
    dft_band_gap: float,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成目标 DFT 带隙值的晶体结构。

    Args:
        dft_band_gap: 目标 DFT 带隙值，单位 eV（例如：2.5）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标带隙的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_dft_band_gap_MatterGen(dft_band_gap=2.5)
    """
    return _generate_material_base(
        properties={"dft_band_gap": dft_band_gap},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )

@llm_tool(
    name="generate_material_by_chemical_system_MatterGen",
    description="使用 MatterGen 模型生成具有特定化学体系的晶体结构"
)
def generate_material_by_chemical_system_MatterGen(
    chemical_system: str,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成目标化学体系的晶体结构。

    Args:
        chemical_system: 目标化学体系（例如："Li-Fe-P-O", "Si-O"）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标化学体系的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_chemical_system_MatterGen(chemical_system="Li-Fe-P-O")
    """
    return _generate_material_base(
        properties={"chemical_system": chemical_system},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )

@llm_tool(
    name="generate_material_by_space_group_MatterGen",
    description="使用 MatterGen 模型生成具有特定空间群的晶体结构"
)
def generate_material_by_space_group_MatterGen(
    space_group: int,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成目标空间群的晶体结构。

    Args:
        space_group: 目标空间群编号（1-230，例如：225 对应 Fm-3m）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标空间群的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_space_group_MatterGen(space_group=225)
    """
    if space_group < 1 or space_group > 230:
        return f"Error：无效的空间群值：{space_group}。必须在 1 到 230 之间。"

    return _generate_material_base(
        properties={"space_group": space_group},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )

@llm_tool(
    name="generate_material_by_dft_mag_density_MatterGen",
    description="使用 MatterGen 模型生成具有特定 DFT 磁密度的晶体结构"
)
def generate_material_by_dft_mag_density_MatterGen(
    dft_mag_density: float,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成目标 DFT 磁密度的晶体结构。

    Args:
        dft_mag_density: 目标 DFT 磁密度，单位 μB/atom（例如：2.0）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标磁密度的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_dft_mag_density_MatterGen(dft_mag_density=2.0)
    """
    return _generate_material_base(
        properties={"dft_mag_density": dft_mag_density},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )



# ============================================================================
# 多属性条件生成
# ============================================================================


@llm_tool(
    name="generate_material_by_dft_mag_density_and_hhi_score_MatterGen",
    description="使用 MatterGen 模型生成具有特定 DFT 磁密度和 HHI 稀缺性评分的晶体结构"
)
def generate_material_by_dft_mag_density_and_hhi_score_MatterGen(
    dft_mag_density: float,
    hhi_score: float,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成同时满足特定 DFT 磁密度和 HHI 稀缺性评分的晶体结构。

    此函数使用专门训练的多属性模型，能够更好地控制这两个属性的组合。

    Args:
        dft_mag_density: 目标 DFT 磁密度，单位 μB/atom（例如：2.0）
        hhi_score: 目标 HHI 稀缺性评分（例如：0.5）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标属性的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_dft_mag_density_and_hhi_score_MatterGen(dft_mag_density=2.0, hhi_score=0.5)
    """
    return _generate_material_base(
        properties={"dft_mag_density": dft_mag_density, "hhi_score": hhi_score},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )

@llm_tool(
    name="generate_material_by_chemical_system_and_energy_above_hull_MatterGen",
    description="使用 MatterGen 模型生成具有特定化学体系和能量高于凸包的晶体结构"
)
def generate_material_by_chemical_system_and_energy_above_hull_MatterGen(
    chemical_system: str,
    energy_above_hull: float,
    batch_size: int = 2,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 2.0
) -> str:
    """
    生成同时满足特定化学体系和能量高于凸包的晶体结构。

    此函数使用专门训练的多属性模型，能够更好地控制这两个属性的组合。

    Args:
        chemical_system: 目标化学体系（例如："Li-Fe-P-O", "Si-O"）
        energy_above_hull: 目标能量高于凸包值，单位 eV/atom（例如：0.1）
        batch_size: 每批生成的结构数量（默认：2）
        num_batches: 生成的批次数量（默认：1）
        diffusion_guidance_factor: 控制对目标属性的遵循程度（默认：2.0）

    Returns:
        包含生成晶体结构的描述性文本，采用 CIF 格式

    Example:
        generate_material_by_chemical_system_and_energy_above_hull_MatterGen(
            chemical_system="Li-Fe-P-O",
            energy_above_hull=0.1
        )
    """
    return _generate_material_base(
        properties={"chemical_system": chemical_system, "energy_above_hull": energy_above_hull},
        batch_size=batch_size,
        num_batches=num_batches,
        diffusion_guidance_factor=diffusion_guidance_factor
    )
