"""
MatGL 工具函数测试脚本

用于验证重构后的工具函数是否正常工作
"""
import asyncio
from tools.matgl_tools import (
    relax_crystal_structure_MatGL,
    predict_formation_energy_MatGL,
    run_molecular_dynamics_MatGL,
    calculate_single_point_energy_MatGL,
    predict_multi_fidelity_band_gap_MatGL
)

# 简单的CIF格式测试数据 (NaCl结构)
test_cif = """data_NaCl
_cell_length_a 5.64
_cell_length_b 5.64
_cell_length_c 5.64
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_space_group_name_H-M_alt 'F m -3 m'
_space_group_IT_number 225
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.0 0.0
"""

async def test_structure_relaxation():
    """测试结构优化功能"""
    print("=== 测试结构优化 ===")
    try:
        result = await relax_crystal_structure_MatGL(test_cif, fmax=0.1)
        print("✓ 结构优化测试成功")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ 结构优化测试失败: {e}")
    print()

async def test_formation_energy():
    """测试形成能预测功能"""
    print("=== 测试形成能预测 ===")
    try:
        result = await predict_formation_energy_MatGL(test_cif, optimize_structure=False)
        print("✓ 形成能预测测试成功")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ 形成能预测测试失败: {e}")
    print()

async def test_single_point_energy():
    """测试单点能计算功能"""
    print("=== 测试单点能计算 ===")
    try:
        result = await calculate_single_point_energy_MatGL(test_cif, optimize_structure=False)
        print("✓ 单点能计算测试成功")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ 单点能计算测试失败: {e}")
    print()

async def test_molecular_dynamics():
    """测试分子动力学模拟功能"""
    print("=== 测试分子动力学模拟 ===")
    try:
        result = await run_molecular_dynamics_MatGL(
            test_cif,
            temperature_K=300,
            steps=10,  # 使用较少步数进行快速测试
            optimize_structure=False
        )
        print("✓ 分子动力学模拟测试成功")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ 分子动力学模拟测试失败: {e}")
    print()

async def test_band_gap_prediction():
    """测试带隙预测功能"""
    print("=== 测试带隙预测 ===")
    try:
        result = await predict_multi_fidelity_band_gap_MatGL(test_cif, optimize_structure=False)
        print("✓ 带隙预测测试成功")
        print(result[:200] + "..." if len(result) > 200 else result)
    except Exception as e:
        print(f"✗ 带隙预测测试失败: {e}")
    print()

async def main():
    """主测试函数"""
    print("开始测试 MatGL 工具函数...")
    print("=" * 50)

    # 运行所有测试
    await test_structure_relaxation()
    await test_formation_energy()
    await test_single_point_energy()
    await test_molecular_dynamics()
    await test_band_gap_prediction()

    print("=" * 50)
    print("测试完成！")

if __name__ == "__main__":
    asyncio.run(main())
# === 测试结构优化 ===
# ✓ 结构优化测试成功
# ## 结构优化结果

# - **分子式**: `NaCl`
# - **力收敛阈值**: `0.1 eV/Å`
# - **优化状态**: `成功优化`

# ### 结构信息

# - **分子式**: `NaCl`
# - **空间群**: `Fm-3m (#225)`
# - **体积**: `180.18 Å³`
# - **密度**: `2.15 g/cm³`
# - **晶格参数**:
#   - a = `5.64812...

# === 测试形成能预测 ===
# ✓ 形成能预测测试成功
# ## 形成能预测结果

# - **分子式**: `NaCl`
# - **结构状态**: `未优化`
# - **形成能**: `-2.097 eV/atom`

# ### 结构信息

# - **分子式**: `NaCl`
# - **空间群**: `Fm-3m (#225)`
# - **体积**: `179.41 Å³`
# - **密度**: `2.16 g/cm³`
# - **晶格参数**:
#   - a = `5.6...

# === 测试单点能计算 ===
# ✓ 单点能计算测试成功
# ## 单点能计算结果

# - **分子式**: `NaCl`
# - **结构状态**: `未优化`
# - **势能**: `-26.927 eV`

# ### 结构信息

# - **分子式**: `NaCl`
# - **空间群**: `Fm-3m (#225)`
# - **体积**: `179.41 Å³`
# - **密度**: `2.16 g/cm³`
# - **晶格参数**:
#   - a = `5.640000...

# === 测试分子动力学模拟 ===
# ✓ 分子动力学模拟测试成功
# ## 分子动力学模拟结果

# - **分子式**: `NaCl`
# - **结构状态**: `未优化`
# - **模拟温度**: `300.0 K`
# - **模拟步数**: `10`
# - **最终势能**: `-26.901 eV`

# ### 结构信息

# - **分子式**: `NaCl`
# - **空间群**: `P1 (#1)`
# - **体积**: `179.41 Å³`
# - **密度**: `2.1...

# === 测试带隙预测 ===
# ✓ 带隙预测测试成功
# ## 多保真度带隙预测结果

# - **分子式**: `NaCl`
# - **结构状态**: `未优化`
# - **模型**: `MEGNet-MP-2019.4.1-BandGap-mfi`

# ### 预测带隙

# | 方法      | 带隙 (eV) |
# |-----------|----------|
# | PBE       |    1.293 |
# | GLLB-SC   |    1.299 ...
